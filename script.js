// script.js
let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let statusEl = document.getElementById('status');
let saveBtn = document.getElementById('saveBtn');
let toggleBlurWarn = document.getElementById('toggleBlurWarn');
let toggleCameraBtn = document.getElementById('toggleCameraBtn');

let streaming = false;
let src = null, gray = null, blurred = null;
let contours = null, hierarchy = null;
let cap = null;
let bestCropped = null;
let cameraStream = null;
let isProcessing = false;
let capturedFrame = null; // Store the captured frame

// onnxruntime-web session for YOLO
let ortSession = null;
let yoloInputShape = [1, 3, 640, 640]; // default, will try to read from model
let yoloInputName = null;
let yoloProviders = [];
const yoloScoreThresh = 0.4;   // confidence threshold for detections
const yoloNmsIouThresh = 0.45; // NMS IoU threshold

const scoreThreshold = 0.60; // card positioning threshold (our combined score)
const sharpnessThreshold = 35; // Slightly lower for more flexibility
const requiredStableFrames = 15; // Increase to get more samples
const inFrameThreshold = 0.7;
let stableCount = 0, bestScore = 0.0, bestContour = null;
let frameHistory = [];
const historySize = 10; // Increase history size to capture more frames

// Add at top with other variables
let frameSkipCounter = 0;
// Add target FPS for both video and YOLO, and track last YOLO time
const TARGET_FPS = 15;
let lastYoloTimestamp = 0;
// NEW: track when a YOLO result actually arrived and allow waiters
let lastYoloResultTimestamp = 0;
let yoloPendingResolvers = [];
// Process every captured frame (camera will be requested at TARGET_FPS)
let PROCESS_EVERY_N_FRAMES = 1; // adaptive; will tune for mobile
let reusableTempCanvas = null; // Reuse canvas
let yoloCanvas = null; // For letterbox preprocessing

// Worker-based YOLO offload
let yoloWorker = null;
let yoloWorkerReady = false;
let yoloRequestInFlight = false;
let lastDetections = [];
let lastYoloMs = 0;
let modelPath = null; // auto-chosen model

// Simple mobile detection + capabilities
function isMobile() {
  // Prefer modern userAgentData when available
  try {
    if (navigator.userAgentData && typeof navigator.userAgentData.mobile === 'boolean') {
      return navigator.userAgentData.mobile;
    }
  } catch(e){}
  const ua = navigator.userAgent || navigator.vendor || window.opera;
  return /android|iphone|ipad|ipod|iemobile|mobile/i.test(ua);
}

// Frame scheduler using requestVideoFrameCallback when available
function scheduleNextFrame(cb) {
  if (typeof video.requestVideoFrameCallback === 'function') {
    video.requestVideoFrameCallback(() => cb());
  } else {
    requestAnimationFrame(cb);
  }
}

function logStatus(s){ statusEl.innerText = s; }

// wait for OpenCV to be ready
function onOpenCvReady() {
  if (typeof cv === 'undefined') {
    logStatus("Waiting for OpenCV...");
    setTimeout(onOpenCvReady, 100);
    return;
  }
  if (cv.getBuildInformation) {
    // Kick off YOLO model loading in parallel
    initYolo().then(() => {
      logStatus("OpenCV.js + YOLO ready - Click 'Start Camera' to begin");
      toggleCameraBtn.disabled = false;
    }).catch(err => {
      console.error(err);
      logStatus("OpenCV.js loaded, YOLO failed: " + err);
      toggleCameraBtn.disabled = false;
    });
  } else {
    setTimeout(onOpenCvReady, 100);
  }
}

// Wait for DOM to be ready, then wait for OpenCV
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(onOpenCvReady, 100);
  });
} else {
  setTimeout(onOpenCvReady, 100);
}

async function startCamera() {
  try {
    // Pick constraints suited for device
    const mobile = isMobile();
    const constraints = mobile
      ? {
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280, max: 1280 },
            height: { ideal: 720, max: 720 },
            frameRate: { ideal: TARGET_FPS, max: TARGET_FPS }
          },
          audio: false
        }
      : {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "environment",
            frameRate: { ideal: TARGET_FPS, max: TARGET_FPS }
          },
          audio: false
        };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = cameraStream;
    video.play();
    video.onloadedmetadata = () => {
        // Wait for video to have actual dimensions
        setTimeout(() => {
            const w = video.videoWidth;
            const h = video.videoHeight;
            console.log('After timeout - Video size:', w, h);
            if (!w || !h) {
                logStatus("Waiting for video size...");
                setTimeout(video.onloadedmetadata, 100);
                return;
            }
            initializeMats();
            isProcessing = true;
            // Slightly higher skip on mobile to save CPU/GPU
            if (isMobile()) {
              PROCESS_EVERY_N_FRAMES = 1;
            }
            // Give one more frame delay before processing
            setTimeout(() => scheduleNextFrame(processVideo), 100);
        }, 200);
    };
    streaming = true;
    toggleCameraBtn.textContent = "Stop Camera";
    logStatus("Camera started");
  } catch (err) {
    logStatus("Cannot access camera: " + err);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    cameraStream = null;
  }
  isProcessing = false;
  streaming = false;
  toggleCameraBtn.textContent = "Start Camera";
  
  // Clear canvas
  const ctx = canvasOutput.getContext('2d');
  ctx.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
  
  logStatus("Camera stopped");
}

function initializeMats(){
  const h = video.videoHeight;
  const w = video.videoWidth;
  console.log('Video dimensions:', w, 'x', h);
  
  canvasOutput.width = w;
  canvasOutput.height = h;
  
  // Don't use VideoCapture, we'll capture from canvas instead
  cap = null;
  src = null;
  
  gray = new cv.Mat();
  blurred = new cv.Mat();
  contours = new cv.MatVector();
  hierarchy = new cv.Mat();
  
  console.log('Mats initialized');
}

function toDegree(rad){ return Math.abs(rad * 180.0 / Math.PI); }

// --- helper conversions of your Python functions ---

function calculateSharpness(frameMatColor, contour){
  // create mask
  const mask = cv.Mat.zeros(frameMatColor.rows, frameMatColor.cols, cv.CV_8UC1);
  const contourVec = new cv.MatVector();
  contourVec.push_back(contour);
  cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
  // gray
  const grayLocal = new cv.Mat();
  cv.cvtColor(frameMatColor, grayLocal, cv.COLOR_RGBA2GRAY);
  const cardRegion = new cv.Mat();
  cv.bitwise_and(grayLocal, grayLocal, cardRegion, mask);

  // Laplacian variance
  const lap = new cv.Mat();
  cv.Laplacian(cardRegion, lap, cv.CV_64F);
  // compute variance manually
  const mean = new cv.Mat();
  const stddev = new cv.Mat();
  cv.meanStdDev(lap, mean, stddev, mask);
  const variance = Math.pow(stddev.doubleAt(0,0), 2);

  // cleanup
  mask.delete(); contourVec.delete(); grayLocal.delete(); cardRegion.delete(); lap.delete(); mean.delete(); stddev.delete();

  // scale to 0-100 similarly
  return Math.min(variance / 5.0, 100.0);
}

function detectReflection(frameMatColor, contour) {
  // ตรวจจับแสงสะท้อนบนการ์ด (รวมทั้ง spike reflections)
  const mask = cv.Mat.zeros(frameMatColor.rows, frameMatColor.cols, cv.CV_8UC1);
  const contourVec = new cv.MatVector();
  contourVec.push_back(contour);
  cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
  
  // แปลงเป็น grayscale
  const grayLocal = new cv.Mat();
  cv.cvtColor(frameMatColor, grayLocal, cv.COLOR_RGBA2GRAY);
  
  // หาค่าเฉลี่ยความสว่างของการ์ดทั้งหมด
  const cardRegionGray = new cv.Mat();
  cv.bitwise_and(grayLocal, grayLocal, cardRegionGray, mask);
  const meanMat = new cv.Mat();
  const stddevMat = new cv.Mat();
  cv.meanStdDev(cardRegionGray, meanMat, stddevMat, mask);
  const meanBrightness = meanMat.doubleAt(0, 0);
  const stddevBrightness = stddevMat.doubleAt(0, 0);
  
  // หาพื้นที่ที่สว่างมากกว่าค่าเฉลี่ย + 2*std (outliers)
  const spikeThreshold = Math.min(meanBrightness + 2 * stddevBrightness, 245);
  const brightThreshold = new cv.Mat();
  cv.threshold(grayLocal, brightThreshold, spikeThreshold, 255, cv.THRESH_BINARY);
  
  // นับจำนวนพิกเซลที่สว่างเกินไปในพื้นที่การ์ด
  const cardRegion = new cv.Mat();
  cv.bitwise_and(brightThreshold, brightThreshold, cardRegion, mask);
  
  const brightPixels = cv.countNonZero(cardRegion);
  const totalPixels = cv.countNonZero(mask);
  const reflectionRatio = brightPixels / totalPixels;
  
  // ตรวจจับ spike reflections โดยหา connected components (ทำเมื่อมีแสงจ้าในระดับหนึ่งเพื่อลดภาระ)
  let hasSpikeReflection = false;
  let maxSpikeIntensity = 0;
  if (reflectionRatio > 0.05) {
    const labels = new cv.Mat();
    const stats = new cv.Mat();
    const centroids = new cv.Mat();
    const numLabels = cv.connectedComponentsWithStats(cardRegion, labels, stats, centroids);

    // ตรวจสอบแต่ละ component (ข้าม label 0 ซึ่งเป็น background)
    for (let i = 1; i < numLabels; i++) {
      const area = stats.intAt(i, cv.CC_STAT_AREA);
      const x = stats.intAt(i, cv.CC_STAT_LEFT);
      const y = stats.intAt(i, cv.CC_STAT_TOP);
      const width = stats.intAt(i, cv.CC_STAT_WIDTH);
      const height = stats.intAt(i, cv.CC_STAT_HEIGHT);
      
      // คำนวณความเข้มข้นของ spike (area / bounding box)
      const boundingBoxArea = width * height;
      const density = area / boundingBoxArea;
      
      // Spike reflection = พื้นที่เล็ก (0.1% - 5% ของการ์ด) แต่เข้มข้นสูง (density > 0.3)
      const minSpikeArea = totalPixels * 0.001;  // 0.1%
      const maxSpikeArea = totalPixels * 0.05;   // 5%
      
      if (area > minSpikeArea && area < maxSpikeArea && density > 0.3) {
        // ตรวจสอบว่า spike นี้สว่างกว่าค่าเฉลี่ยมากพอหรือไม่
        const spikeROI = grayLocal.roi(new cv.Rect(x, y, width, height));
        const spikeMask = labels.roi(new cv.Rect(x, y, width, height));
        const spikeMaskBinary = new cv.Mat();
        const labelVal = new cv.Mat(height, width, spikeMask.type(), new cv.Scalar(i));
        cv.compare(spikeMask, labelVal, spikeMaskBinary, cv.CMP_EQ);
        
  const spikeMean = new cv.Mat();
  const spikeStd = new cv.Mat();
  cv.meanStdDev(spikeROI, spikeMean, spikeStd, spikeMaskBinary);
        const spikeIntensity = spikeMean.doubleAt(0, 0);
        
        // Spike ต้องสว่างกว่าค่าเฉลี่ยอย่างน้อย 40 units
        if (spikeIntensity - meanBrightness > 40) {
          hasSpikeReflection = true;
          maxSpikeIntensity = Math.max(maxSpikeIntensity, spikeIntensity - meanBrightness);
        }
        
        // Cleanup ROI temporaries
        spikeMaskBinary.delete();
  spikeMean.delete();
  spikeStd.delete();
        spikeROI.delete();
        spikeMask.delete();
        labelVal.delete();
      }
    }

    // cleanup CC mats
    labels.delete();
    stats.delete();
    centroids.delete();
  }

  // cleanup commons
  mask.delete();
  contourVec.delete();
  grayLocal.delete();
  cardRegionGray.delete();
  meanMat.delete();
  stddevMat.delete();
  brightThreshold.delete();
  cardRegion.delete();
  
  // ตรวจพบแสงสะท้อนถ้า:
  // 1. มีพื้นที่สว่างเกินไปมากกว่า 15%
  // 2. หรือมี spike reflection ที่สว่างกว่าค่าเฉลี่ยมาก
  return {
    hasReflection: reflectionRatio > 0.15 || hasSpikeReflection,
    reflectionRatio: reflectionRatio,
    hasSpikeReflection: hasSpikeReflection
  };
}

function contourToMat(contourPts){
  // contourPts = JS array of {x,y}
  const mat = new cv.Mat(contourPts.length, 1, cv.CV_32SC2);
  for (let i=0;i<contourPts.length;i++){
    mat.intPtr(i,0)[0] = contourPts[i].x;
    mat.intPtr(i,0)[1] = contourPts[i].y;
  }
  return mat;
}

function isCardInFrame(cardContour, frameBox) {
  // ตรวจสอบว่าบัตรอยู่ในกรอบหรือไม่
  if (!cardContour || cardContour.rows !== 4) return false;
  
  let pointsInside = 0;
  for (let i = 0; i < 4; i++) {
    const x = cardContour.intAt(i, 0);
    const y = cardContour.intAt(i, 1);
    
    if (x >= frameBox.x && x <= frameBox.x + frameBox.width &&
        y >= frameBox.y && y <= frameBox.y + frameBox.height) {
      pointsInside++;
    }
  }
  
  // อย่างน้อย 3 ใน 4 มุมต้องอยู่ในกรอบ
  return pointsInside >= 3;
}

function calculateCardScore(contour, frameShape){
  const frameArea = frameShape.height * frameShape.width;
  const cardArea = cv.contourArea(contour);
  const rect = cv.boundingRect(contour);
  const x = rect.x, y = rect.y, w = rect.width, h = rect.height;
  const areaRatio = cardArea / frameArea;
  
  // Improved area scoring - prefer 15-50% of frame
  let areaScore = 0;
  if (areaRatio >= 0.15 && areaRatio <= 0.50) {
    // Optimal range
    areaScore = 1.0;
  } else if (areaRatio < 0.15) {
    // Too small
    areaScore = areaRatio / 0.15;
  } else {
    // Too large
    areaScore = Math.max(0, 1.0 - (areaRatio - 0.50) / 0.30);
  }

  const frame_cx = frameShape.width / 2.0, frame_cy = frameShape.height / 2.0;
  const card_cx = x + w/2.0, card_cy = y + h/2.0;
  const distance = Math.hypot(frame_cx - card_cx, frame_cy - card_cy);
  const max_distance = Math.hypot(frame_cx, frame_cy);
  const centerScore = 1.0 - (distance / max_distance);

  const hull = new cv.Mat();
  cv.convexHull(contour, hull, false, true);
  const hullArea = cv.contourArea(hull);
  const straightnessScore = hullArea > 0 ? (cardArea / hullArea) : 0;

  hull.delete();

  // Adjusted weights - prioritize straightness and centering
  return (areaScore * 0.25 + centerScore * 0.35 + straightnessScore * 0.40);
}

// New function to calculate combined quality score
function calculateQualityScore(score, sharpness, reflectionRatio) {
  const normalizedSharpness = Math.min(sharpness / 100.0, 1.0);
  const reflectionPenalty = Math.max(0, 1.0 - (reflectionRatio * 2)); // Penalize reflection
  
  // Combined quality: 40% positioning, 50% sharpness, 10% reflection penalty
  return (score * 0.40) + (normalizedSharpness * 0.50) + (reflectionPenalty * 0.10);
}

function sharpenImage(mat) {
  // Apply sharpening filter เหมือนใน Python
  const kernel = cv.matFromArray(3, 3, cv.CV_32FC1, [
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
  ]);
  const sharpened = new cv.Mat();
  cv.filter2D(mat, sharpened, cv.CV_8U, kernel);
  kernel.delete();
  return sharpened;
}

function extractCardRegion(frameMat, contour){
  // expects contour with 4 points (cv.CV_32SC2)
  if (contour.rows !== 4){
    const r = cv.boundingRect(contour);
    const padx = Math.floor(r.width * 0.05), pady = Math.floor(r.height * 0.05);
    const x = Math.max(0, r.x - padx), y = Math.max(0, r.y - pady);
    const w = Math.min(frameMat.cols - x, r.width + 2*padx), h = Math.min(frameMat.rows - y, r.height + 2*pady);
    return frameMat.roi(new cv.Rect(x,y,w,h)).clone();
  }

  // Get the four corners
  let points = [];
  for (let i=0;i<4;i++){
    points.push([contour.intAt(i,0), contour.intAt(i,1)]);
  }
  
  // Sort points properly: top-left, top-right, bottom-right, bottom-left
  // ใช้วิธีที่แม่นยำกว่า โดยเรียงตาม y ก่อน แล้วค่อยเรียงตาม x
  points.sort((a, b) => a[1] - b[1]); // เรียงตาม y
  
  let rect = new Array(4);
  // 2 จุดบนสุด
  let topPoints = [points[0], points[1]];
  topPoints.sort((a, b) => a[0] - b[0]); // เรียงตาม x
  rect[0] = topPoints[0]; // top-left
  rect[1] = topPoints[1]; // top-right
  
  // 2 จุดล่างสุด
  let bottomPoints = [points[2], points[3]];
  bottomPoints.sort((a, b) => a[0] - b[0]); // เรียงตาม x
  rect[3] = bottomPoints[0]; // bottom-left
  rect[2] = bottomPoints[1]; // bottom-right
  
  // Expand the corners slightly outward (3% expansion)
  const center_x = (rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4.0;
  const center_y = (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4.0;
  const expansion_factor = 1.03;
  
  for (let i = 0; i < 4; i++) {
    const direction_x = rect[i][0] - center_x;
    const direction_y = rect[i][1] - center_y;
    rect[i][0] = center_x + direction_x * expansion_factor;
    rect[i][1] = center_y + direction_y * expansion_factor;
  }
  
  // คำนวณความกว้างและความสูงจริงของการ์ด
  const width_top = Math.sqrt(Math.pow(rect[1][0] - rect[0][0], 2) + Math.pow(rect[1][1] - rect[0][1], 2));
  const width_bottom = Math.sqrt(Math.pow(rect[2][0] - rect[3][0], 2) + Math.pow(rect[2][1] - rect[3][1], 2));
  const width = Math.max(width_top, width_bottom);
  
  const height_left = Math.sqrt(Math.pow(rect[3][0] - rect[0][0], 2) + Math.pow(rect[3][1] - rect[0][1], 2));
  const height_right = Math.sqrt(Math.pow(rect[2][0] - rect[1][0], 2) + Math.pow(rect[2][1] - rect[1][1], 2));
  const height = Math.max(height_left, height_right);
  
  // อัตราส่วนมาตรฐานของการ์ด
  const card_ratio = 86 / 54;
  
  // คำนวณขนาดสุดท้าย
  let finalWidth = Math.floor(width);
  let finalHeight = Math.floor(height);
  
  const current_ratio = width / height;
  
  if (current_ratio > 1) {
    // landscape orientation
    if (current_ratio > card_ratio) {
      finalWidth = Math.floor(finalHeight * card_ratio);
    } else {
      finalHeight = Math.floor(finalWidth / card_ratio);
    }
  } else {
    // portrait orientation
    const portrait_ratio = 54 / 86;
    if (current_ratio > portrait_ratio) {
      finalWidth = Math.floor(finalHeight * portrait_ratio);
    } else {
      finalHeight = Math.floor(finalWidth / portrait_ratio);
    }
  }

  // Define destination points for perspective transform
  const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    finalWidth - 1, 0,
    finalWidth - 1, finalHeight - 1,
    0, finalHeight - 1
  ]);
  
  // Source points (ตรงตามลำดับ: TL, TR, BR, BL)
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    rect[0][0], rect[0][1],  // top-left
    rect[1][0], rect[1][1],  // top-right
    rect[2][0], rect[2][1],  // bottom-right
    rect[3][0], rect[3][1]   // bottom-left
  ]);
  
  // Get perspective transformation matrix
  const M = cv.getPerspectiveTransform(srcPts, dst);
  
  // Apply perspective transformation
  const warped = new cv.Mat();
  cv.warpPerspective(frameMat, warped, M, new cv.Size(finalWidth, finalHeight));

  // cleanup
  srcPts.delete();
  dst.delete();
  M.delete();
  
  // Apply sharpening to improve clarity
  const sharpened = sharpenImage(warped);
  warped.delete();
  
  return sharpened;
}

// ================== YOLO (ONNX Runtime Web) helpers ==================
async function initYolo() {
  // Choose model path (faster on mobile by default)
  modelPath = isMobile() ? 'best480.onnx' : 'best640.onnx';

  // Try to init Web Worker first for multi-thread speedup
  try {
    yoloWorker = new Worker('yoloWorker.js');
  const allowWebGPU = !isMobile() && !!navigator.gpu;
  // Use a single WASM thread unless COOP/COEP enabled (crossOriginIsolated)
  const coi = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
  const numThreads = coi ? Math.min(4, (navigator.hardwareConcurrency || 4)) : 1;
    const readyPromise = new Promise((resolve, reject) => {
      const onMsg = (ev) => {
        const m = ev.data;
        if (!m || !m.type) return;
        if (m.type === 'ready') {
          yoloWorkerReady = true;
          if (m.inputShape) yoloInputShape = m.inputShape;
          resolve();
        } else if (m.type === 'error') {
          reject(new Error(m.error || 'Worker init error'));
        }
      };
      yoloWorker.addEventListener('message', onMsg, { once: true });
      yoloWorker.postMessage({
        type: 'init',
        allowWebGPU,
        numThreads,
        scoreThresh: yoloScoreThresh,
        nmsIouThresh: yoloNmsIouThresh,
        modelPath
      });
    });
    await readyPromise;
    // Hook continuous result listener
    yoloWorker.addEventListener('message', (ev) => {
      const m = ev.data || {};
      if (m.type === 'result') {
        lastDetections = m.dets || [];
        lastYoloMs = m.timeMs || 0;
        yoloRequestInFlight = false;
        // NEW: mark when worker result arrived
        lastYoloResultTimestamp = performance.now();
        // Resolve one pending waiter (if any) with the fresh detections
        if (yoloPendingResolvers.length) {
          try {
            const resolver = yoloPendingResolvers.shift();
            resolver(lastDetections || []);
          } catch(e) { /* ignore */ }
        }
      } else if (m.type === 'error') {
        console.warn('YOLO worker error:', m.error);
        yoloRequestInFlight = false;
        // resolve any waiters with fallback data
        while (yoloPendingResolvers.length) {
          try { yoloPendingResolvers.shift()([]); } catch(e){}
        }
      }
    });
    return; // Worker ready, skip main-thread ORT init
  } catch (e) {
    console.warn('YOLO worker failed, falling back to main thread:', e);
  }

  // Fallback: init ORT on main thread
  if (typeof ort === 'undefined') {
    throw new Error('onnxruntime-web (ort) not found. Make sure index.html includes it.');
  }
  try {
    if (ort.env && ort.env.wasm) {
      // Ensure ORT loads wasm assets from CDN when needed
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
      ort.env.wasm.simd = true;
      const coi = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
      ort.env.wasm.numThreads = coi ? Math.min(4, (navigator.hardwareConcurrency || 4)) : 1;
    }
  } catch (e) { /* ignore */ }

  const ep = [];
  const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
  const mobile = isMobile();
  if (!mobile && navigator.gpu) ep.push('webgpu');
  if (gl) ep.push('webgl');
  ep.push('wasm');
  yoloProviders = ep;

  const so = { executionProviders: ep, graphOptimizationLevel: 'all' };
  ortSession = await ort.InferenceSession.create(modelPath, so);
  yoloInputName = ortSession.inputNames[0];
  try {
    const meta = ortSession.inputMetadata[yoloInputName];
    if (meta && Array.isArray(meta.dimensions)) {
      const dims = meta.dimensions.map(d => (typeof d === 'number' && d > 0 ? d : null));
      if (dims[2] && dims[3]) yoloInputShape = [1, 3, dims[2], dims[3]];
    }
  } catch(e) { console.warn('Could not read input metadata, using default 640x640'); }

  if (!yoloCanvas) {
    if (typeof OffscreenCanvas !== 'undefined') yoloCanvas = new OffscreenCanvas(yoloInputShape[2], yoloInputShape[3]);
    else yoloCanvas = document.createElement('canvas');
  }
  yoloCanvas.width = yoloInputShape[2];
  yoloCanvas.height = yoloInputShape[3];
}

function letterboxToCanvas(srcCanvas, dstCanvas, dstW, dstH, fill = [114,114,114,255]) {
  // Keep aspect ratio by padding
  const srcW = srcCanvas.width, srcH = srcCanvas.height;
  const r = Math.min(dstW / srcW, dstH / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const dw = Math.floor((dstW - newW) / 2);
  const dh = Math.floor((dstH - newH) / 2);

  const ctx = dstCanvas.getContext('2d', { willReadFrequently: true });
  // Fill with gray
  ctx.fillStyle = `rgba(${fill[0]},${fill[1]},${fill[2]},${fill[3]/255})`;
  ctx.fillRect(0, 0, dstW, dstH);
  // Favor speed on mobile
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = isMobile() ? 'low' : 'high';
  ctx.drawImage(srcCanvas, 0, 0, srcW, srcH, dw, dh, newW, newH);
  return { scale: r, padX: dw, padY: dh, newW, newH };
}

function preprocessForYolo(srcCanvas) {
  const [n, c, w, h] = yoloInputShape; // NCHW
  const info = letterboxToCanvas(srcCanvas, yoloCanvas, w, h);
  const ctx = yoloCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data; // RGBA
  const chw = new Float32Array(n * c * w * h);
  // Convert to RGB, normalize 0..1, NCHW
  let p = 0;
  const wh = w * h;
  for (let i = 0; i < wh; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = r;            // R channel
    chw[i + wh] = g;       // G channel
    chw[i + 2 * wh] = b;   // B channel
    p += 3;
  }
  const input = new ort.Tensor('float32', chw, [n, c, h, w]);
  return { input, info };
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function xywh2xyxy(x, y, w, h) {
  const x1 = x - w / 2;
  const y1 = y - h / 2;
  const x2 = x + w / 2;
  const y2 = y + h / 2;
  return [x1, y1, x2, y2];
}

function nms(boxes, scores, iouThresh, topK = 100) {
  const idxs = scores.map((s, i) => [s, i]).sort((a,b)=>b[0]-a[0]).map(v=>v[1]);
  const picked = [];
  while (idxs.length) {
    const i = idxs.shift();
    picked.push(i);
    if (picked.length >= topK) break;
    const rest = [];
    for (const j of idxs) {
      const iou = bboxIoU(boxes[i], boxes[j]);
      if (iou < iouThresh) rest.push(j);
    }
    idxs.splice(0, idxs.length, ...rest);
  }
  return picked;
}

function bboxIoU(a, b) {
  const ax1=a[0], ay1=a[1], ax2=a[2], ay2=a[3];
  const bx1=b[0], by1=b[1], bx2=b[2], by2=b[3];
  const interX1 = Math.max(ax1, bx1);
  const interY1 = Math.max(ay1, by1);
  const interX2 = Math.min(ax2, bx2);
  const interY2 = Math.min(ay2, by2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interA = interW * interH;
  const aA = Math.max(0, ax2-ax1) * Math.max(0, ay2-ay1);
  const bA = Math.max(0, bx2-bx1) * Math.max(0, by2-by1);
  const union = aA + bA - interA;
  return union <= 0 ? 0 : interA / union;
}

function postprocessYolo(output, info, origW, origH) {
  // Supports common YOLOv5/8 heads
  // output: ort.Tensor or array of Tensors. We use the first.
  let outTensor = Array.isArray(output) ? output[0] : output;
  let data = outTensor.data;
  const dims = outTensor.dims; // e.g., [1,25200,85] or [1,84,8400]

  const boxes = [];
  const scores = [];
  const classes = [];

  // map back to original image coords
  const gain = info.scale;
  const padX = info.padX;
  const padY = info.padY;
  const inW = yoloInputShape[2];
  const inH = yoloInputShape[3];

  // Heuristic to detect layout
  if (dims.length === 3 && dims[2] >= 6 && dims[1] > 1000) {
    // [1, N, no] with [cx,cy,w,h,obj,cls...]
    const N = dims[1];
    const no = dims[2];
    const numClasses = no - 5;
    let needSigmoid = false;
    for (let k = 0; k < Math.min(10, N); k++) {
      if (data[k*no + 4] > 1) { needSigmoid = true; break; }
    }
    for (let i = 0; i < N; i++) {
      const off = i * no;
      let cx = data[off + 0];
      let cy = data[off + 1];
      let w = data[off + 2];
      let h = data[off + 3];
      let obj = data[off + 4];
      if (needSigmoid) obj = sigmoid(obj);
      // class probability
      let best = 0, bestIdx = 0;
      for (let c = 0; c < numClasses; c++) {
        let v = data[off + 5 + c];
        if (needSigmoid) v = sigmoid(v);
        if (v > best) { best = v; bestIdx = c; }
      }
      const conf = obj * (numClasses > 0 ? best : 1);
      if (conf < yoloScoreThresh) continue;
      let [x1, y1, x2, y2] = xywh2xyxy(cx, cy, w, h);
      // Undo letterbox
      x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain;
      x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
      // Clip
      x1 = Math.max(0, Math.min(origW-1, x1));
      y1 = Math.max(0, Math.min(origH-1, y1));
      x2 = Math.max(0, Math.min(origW-1, x2));
      y2 = Math.max(0, Math.min(origH-1, y2));
      boxes.push([x1,y1,x2,y2]);
      scores.push(conf);
      classes.push(bestIdx);
    }
  } else if (dims.length === 3 && dims[1] >= 6 && dims[2] > 1000) {
    // [1, no, N] YOLOv8 style, first 4 rows are boxes, remaining are class scores
    const no = dims[1];
    const N = dims[2];
    const numClasses = no - 4;
    // We may need to apply sigmoid to class probs if >1 detected
    let needSigmoid = false;
    for (let k = 0; k < Math.min(10, N); k++) {
      if (data[4*N + k] > 1) { needSigmoid = true; break; }
    }
    for (let i = 0; i < N; i++) {
      const cx = data[0 * N + i];
      const cy = data[1 * N + i];
      const w  = data[2 * N + i];
      const h  = data[3 * N + i];
      let best = 0, bestIdx = 0;
      for (let c = 0; c < numClasses; c++) {
        let v = data[(4 + c) * N + i];
        if (needSigmoid) v = sigmoid(v);
        if (v > best) { best = v; bestIdx = c; }
      }
      const conf = best; // v8 often uses class score directly
      if (conf < yoloScoreThresh) continue;
      let [x1, y1, x2, y2] = xywh2xyxy(cx, cy, w, h);
      x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain;
      x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
      x1 = Math.max(0, Math.min(origW-1, x1));
      y1 = Math.max(0, Math.min(origH-1, y1));
      x2 = Math.max(0, Math.min(origW-1, x2));
      y2 = Math.max(0, Math.min(origH-1, y2));
      boxes.push([x1,y1,x2,y2]);
      scores.push(conf);
      classes.push(bestIdx);
    }
  } else {
    console.warn('Unexpected YOLO output dims:', dims);
  }

  // NMS
  const keep = nms(boxes, scores, yoloNmsIouThresh, 20);
  const dets = keep.map(i => ({ box: boxes[i], score: scores[i], cls: classes[i] }));
  return dets;
}

async function runYoloOnCanvas(srcCanvas, waitForResult = false) {
  // Throttle YOLO to TARGET_FPS (works for worker and main-thread)
  const now = performance.now();
  const minInterval = 1000 / TARGET_FPS;
  if (now - lastYoloTimestamp < minInterval) {
    // Too soon to run a new YOLO; return last known detections
    return lastDetections || [];
  }
  // Reserve the timestamp immediately to prevent overlapping launches
  lastYoloTimestamp = now;

  if (yoloWorkerReady) {
    // If worker is ready, prefer it (non-blocking). Only send if no request in flight.
    if (!yoloRequestInFlight) {
      try {
        const bitmap = await createImageBitmap(srcCanvas);
        yoloRequestInFlight = true;
        yoloWorker.postMessage({ type: 'detect', bitmap, origW: srcCanvas.width, origH: srcCanvas.height }, [bitmap]);
      } catch (e) {
        // ignore send errors, will fallback to lastDetections
        yoloRequestInFlight = false;
      }
    }

    // If caller wants a fresh result, wait (with timeout) for the next worker result
    if (waitForResult) {
      return await new Promise((resolve) => {
        yoloPendingResolvers.push(resolve);
        // safety timeout (400ms) to avoid blocking UI too long
        setTimeout(() => {
          const idx = yoloPendingResolvers.indexOf(resolve);
          if (idx !== -1) yoloPendingResolvers.splice(idx, 1);
          resolve(lastDetections || []);
        }, 400);
      });
    }

    // Otherwise return last known detections immediately
    return lastDetections || [];
  }

  if (!ortSession) return [];
  const { input, info } = preprocessForYolo(srcCanvas);
  const feeds = {}; feeds[yoloInputName] = input;
  const output = await ortSession.run(feeds);
  // Use the first output tensor
  const first = output[Object.keys(output)[0]];
  const dets = postprocessYolo(first, info, srcCanvas.width, srcCanvas.height);
  // mark main-thread run time as result arrival
  lastYoloResultTimestamp = performance.now();
  return dets;
}

// ----------------- main processing loop -----------------
async function processVideo(){
  if (!isProcessing) return;
  
  // Skip frames to reduce CPU load
  frameSkipCounter++;
  if (frameSkipCounter % PROCESS_EVERY_N_FRAMES !== 0) {
    scheduleNextFrame(processVideo);
    return;
  }
  
  try {
    const t0 = performance.now();
    // Reuse canvas instead of creating new one
    if (!reusableTempCanvas) {
      reusableTempCanvas = document.createElement('canvas');
    }
    reusableTempCanvas.width = video.videoWidth;
    reusableTempCanvas.height = video.videoHeight;
  const tempCtx = reusableTempCanvas.getContext('2d', { willReadFrequently: true });
    tempCtx.drawImage(video, 0, 0, reusableTempCanvas.width, reusableTempCanvas.height);
    
    if (src) src.delete();
    src = cv.imread(reusableTempCanvas);
    
    let display = src.clone();
    
    const frameW = src.cols;
    const frameH = src.rows;
    const card_ratio = 86 / 54;
    
    let guideWidth = Math.floor(frameW * 0.7);
    let guideHeight = Math.floor(guideWidth / card_ratio);
    
    if (guideHeight > frameH * 0.8) {
      guideHeight = Math.floor(frameH * 0.8);
      guideWidth = Math.floor(guideHeight * card_ratio);
    }
    
    const guideX = Math.floor((frameW - guideWidth) / 2);
    const guideY = Math.floor((frameH - guideHeight) / 2);
    
    const guideBox = {
      x: guideX,
      y: guideY,
      width: guideWidth,
      height: guideHeight
    };
    
    // วาดกรอบอ้างอิง
    cv.rectangle(display, 
                new cv.Point(guideX, guideY),
                new cv.Point(guideX + guideWidth, guideY + guideHeight),
                new cv.Scalar(100, 100, 100, 255), 2);
    cv.putText(display, "Align card here", 
               new cv.Point(guideX + 10, guideY - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(100, 100, 100, 255), 2);

    // === YOLO detection instead of Canny/contours ===
    let detections = [];
    // Run YOLO (non-blocking via worker if available)
    if (yoloWorkerReady) {
      // If the last worker result is stale, wait briefly for a fresh result to avoid drawing old boxes.
      const now = performance.now();
      const stale = (now - (lastYoloResultTimestamp || 0)) > 500; // 500ms threshold
      if (!yoloRequestInFlight && stale) {
        // Wait for a fresh worker result (with fallback timeout inside runYoloOnCanvas)
        detections = await runYoloOnCanvas(reusableTempCanvas, true);
      } else {
        // Fire off detection if none in-flight, but don't await
        runYoloOnCanvas(reusableTempCanvas);
        detections = lastDetections || [];
      }
    } else {
      // Fallback: run on main thread and await
      detections = await runYoloOnCanvas(reusableTempCanvas);
    }

    // Pick best detection
    if (detections.length > 0) {
      // Highest confidence
      detections.sort((a,b)=>b.score-a.score);
      const det = detections[0];
      const [x1,y1,x2,y2] = det.box;

      // Build a 4-point rectangle contour
      const bestLocalContour = new cv.Mat(4,1,cv.CV_32SC2);
      // top-left
      bestLocalContour.intPtr(0,0)[0] = Math.round(x1);
      bestLocalContour.intPtr(0,0)[1] = Math.round(y1);
      // top-right
      bestLocalContour.intPtr(1,0)[0] = Math.round(x2);
      bestLocalContour.intPtr(1,0)[1] = Math.round(y1);
      // bottom-right
      bestLocalContour.intPtr(2,0)[0] = Math.round(x2);
      bestLocalContour.intPtr(2,0)[1] = Math.round(y2);
      // bottom-left
      bestLocalContour.intPtr(3,0)[0] = Math.round(x1);
      bestLocalContour.intPtr(3,0)[1] = Math.round(y2);

      const score = calculateCardScore(bestLocalContour, {width: src.cols, height: src.rows});
      
      // Only calculate sharpness if score is good enough
      let sharpness = 0;
      let reflectionData = { hasReflection: false, reflectionRatio: 0 };
      
      if (score > scoreThreshold - 0.1) { // Pre-filter by score
        sharpness = calculateSharpness(src, bestLocalContour);
        const isSharp = sharpness >= sharpnessThreshold;
        
        // Only check reflection if sharp enough
        if (isSharp) {
          reflectionData = detectReflection(src, bestLocalContour);
        }
      }
      
      const isSharp = sharpness >= sharpnessThreshold;
      const inFrame = isCardInFrame(bestLocalContour, guideBox);
      const hasReflection = reflectionData.hasReflection;
      const isGood = (score > scoreThreshold) && isSharp && inFrame && !hasReflection;

      // REMOVE OR COMMENT OUT THESE LINES (lines 405-410)
      // const contourVec = new cv.MatVector();
      // contourVec.push_back(bestLocalContour);
      // const color = isGood ? new cv.Scalar(0,255,0,255) : 
      //               inFrame ? new cv.Scalar(0,165,255,255) : 
      //               new cv.Scalar(255,0,0,255);
      // cv.drawContours(display, contourVec, 0, color, 2);
      // contourVec.delete();

      const color = isGood ? new cv.Scalar(0,255,0,255) : 
                    inFrame ? new cv.Scalar(0,165,255,255) : 
                    new cv.Scalar(255,0,0,255);

      cv.putText(display, `Score: ${score.toFixed(2)}`, new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, color,2);
      cv.putText(display, `Sharp: ${sharpness.toFixed(1)}`, new cv.Point(10,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(isSharp?0:255, isSharp?255:0, 0,255),2);
      cv.putText(display, `In Frame: ${inFrame?'YES':'NO'}`, new cv.Point(10,90), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(inFrame?0:255, inFrame?255:0, 0,255),2);
      cv.putText(display, `Reflection: ${(reflectionData.reflectionRatio*100).toFixed(1)}%`, new cv.Point(10,120), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(hasReflection?255:0, hasReflection?0:255, 0,255),2);

      if (isGood) {
        // Calculate quality score for comparison
        const qualityScore = calculateQualityScore(score, sharpness, reflectionData.reflectionRatio);
        
        // Store frame with quality score
        frameHistory.push({
          frame: src.clone(),
          contour: bestLocalContour.clone(),
          score: score,
          sharpness: sharpness,
          qualityScore: qualityScore
        });
        
        if (frameHistory.length > historySize) {
          const oldest = frameHistory.shift();
          oldest.frame.delete();
          oldest.contour.delete();
        }
        
        // Update best score based on quality
        if (qualityScore > bestScore) {
          bestScore = qualityScore;
          if (bestContour) bestContour.delete();
          bestContour = bestLocalContour.clone();
          stableCount = 0; // Reset when we find better quality
        } else {
          stableCount++;
        }
        
        cv.putText(display, `Quality: ${qualityScore.toFixed(2)}`, new cv.Point(10,150), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,255,0,255),2);
        cv.putText(display, `Stable: ${stableCount}/${requiredStableFrames}`, new cv.Point(10,180), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,255,255,255),2);

        if (stableCount >= requiredStableFrames) {
          // Find best frame based on quality score (not just sharpness)
          let bestFrameData = frameHistory[0];
          for (let i = 1; i < frameHistory.length; i++) {
            if (frameHistory[i].qualityScore > bestFrameData.qualityScore) {
              bestFrameData = frameHistory[i];
            }
          }
          
          console.log('Selected frame - Quality:', bestFrameData.qualityScore.toFixed(3), 
                     'Sharpness:', bestFrameData.sharpness.toFixed(1),
                     'Score:', bestFrameData.score.toFixed(3));
          
          if (capturedFrame) capturedFrame.delete();
          capturedFrame = bestFrameData.frame.clone();

          if (bestContour) bestContour.delete();
          bestContour = bestFrameData.contour.clone();
          
          if (bestCropped) bestCropped.delete();
          bestCropped = null;
          
          frameHistory.forEach(f => {
            f.frame.delete();
            f.contour.delete();
          });
          frameHistory = [];
          
          stableCount = 0;
          bestScore = 0.0;
          
          isProcessing = false;
          stopCamera();
          
          const displayFrame = capturedFrame.clone();
          // Draw nothing on final capture; keep clean
          
          cv.putText(displayFrame, "Card Detected! Click 'Crop Card' to extract", new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,255,0,255),2);
          cv.imshow(canvasOutput, displayFrame);
          displayFrame.delete();
          
          logStatus("Card detected! Click 'Crop Card' to extract or 'Start Camera' to retry.");
        }
      } else {
        stableCount = 0;
        frameHistory.forEach(f => {
          f.frame.delete();
          f.contour.delete();
        });
        frameHistory = [];
        
        if (toggleBlurWarn.checked) {
          if (!inFrame) cv.putText(display, "Move card INTO the frame", new cv.Point(10,180), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,0,0,255),2);
          if (!isSharp) cv.putText(display, "Image too blurry - hold steady", new cv.Point(10,210), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,0,255,255),2);
          if (hasReflection) cv.putText(display, "Light reflection detected - adjust angle", new cv.Point(10,240), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,165,0,255),2);
          if (score <= scoreThreshold) cv.putText(display, "Position card better in frame", new cv.Point(10,270), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,0,255,255),2);
        }
      }

      // Draw detection rectangle
      const rectColor = new cv.Scalar(0, 255, 0, 255);
      cv.rectangle(display, new cv.Point(Math.round(x1), Math.round(y1)), new cv.Point(Math.round(x2), Math.round(y2)), rectColor, 2);
      bestLocalContour.delete();
    } else {
      stableCount = 0;
      frameHistory.forEach(f => {
        f.frame.delete();
        f.contour.delete();
      });
      frameHistory = [];
      
      cv.putText(display, "No card detected", new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, new cv.Scalar(0,0,255,255),2);
      cv.putText(display, "Place card in the frame", new cv.Point(10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(0,0,255,255),2);
    }

    cv.imshow(canvasOutput, display);
    display.delete();

  } catch (err){
    console.error(err);
    logStatus("Processing error: " + err);
  }

  if (isProcessing) {
    // Adaptive frame skipping based on processing time
    const t1 = performance.now();
    const dt = t1 - (window.__lastFrameT || t1);
    window.__lastFrameT = t1;
    if (dt > 120 && PROCESS_EVERY_N_FRAMES < 4) {
      PROCESS_EVERY_N_FRAMES++;
    } else if (dt < 60 && PROCESS_EVERY_N_FRAMES > 1) {
      PROCESS_EVERY_N_FRAMES--;
    }
    // Lightweight FPS status
    const fps = (1000 / (dt || 16.7)).toFixed(1);
    if (statusEl) {
      const base = streaming ? 'Camera started' : statusEl.innerText.split(' | ')[0];
      statusEl.innerText = `${base} | FPS ~ ${fps}${lastYoloMs?` | YOLO ${lastYoloMs.toFixed(0)}ms`:''}`;
    }
    scheduleNextFrame(processVideo);
  }
}

// Toggle camera on/off
toggleCameraBtn.addEventListener('click', () => {
  if (streaming) {
    stopCamera();
    // ล้างประวัติเมื่อหยุด
    frameHistory.forEach(f => {
      f.frame.delete();
      f.contour.delete();
    });
    frameHistory = [];
  } else {
    // Reset captured frame when restarting
    if (capturedFrame) {
      capturedFrame.delete();
      capturedFrame = null;
    }
    if (bestContour) {
      bestContour.delete();
      bestContour = null;
    }
    startCamera();
  }
});

// Crop button - extract card from captured frame
saveBtn.addEventListener('click', ()=>{
  if (!capturedFrame || !bestContour) { 
    alert("No card detected yet. Please capture a card first."); 
    return; 
  }
  
  // Extract the card region จากภาพต้นฉบับที่ไม่มีกรอบ (capturedFrame)
  if (bestCropped) bestCropped.delete();
  bestCropped = extractCardRegion(capturedFrame, bestContour);
  
  // Show the cropped and sharpened card (ไม่มีกรอบเขียว)
  cv.imshow(canvasOutput, bestCropped);
  logStatus("Card cropped and sharpened! Downloading...");
  
  // Auto-download - Create a temporary canvas for clean output
  setTimeout(() => {
    // Create a temporary canvas to ensure clean output
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = bestCropped.cols;
    tempCanvas.height = bestCropped.rows;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the clean cropped image to temp canvas
    cv.imshow(tempCanvas, bestCropped);
    
    // Download from temp canvas
    const link = document.createElement('a');
    link.download = 'cropped_card.png';
    link.href = tempCanvas.toDataURL('image/png', 1.0); // คุณภาพสูงสุด
    link.click();
    
    logStatus("Card saved! Click 'Start Camera' to capture another card.");
  }, 100);
});                                                                              

// cleanup when page unloads
window.addEventListener('unload', ()=>{
  stopCamera();
  try {
    if (src) src.delete();
    if (gray) gray.delete();
    if (blurred) blurred.delete();
    if (contours) contours.delete();
    if (hierarchy) hierarchy.delete();
    if (bestContour) bestContour.delete();
    if (bestCropped) bestCropped.delete();
    if (cap) cap.delete();
    if (yoloWorker) { yoloWorker.terminate(); yoloWorker = null; }
  } catch(e){}
});

// Pause processing when tab is hidden to save battery
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    isProcessing = false;
  } else if (streaming && !isProcessing) {
    isProcessing = true;
    scheduleNextFrame(processVideo);
  }
});