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
// Treat yoloInputShape as [N, C, H, W] (NCHW). Fixed for 160x160 model.
let yoloInputShape = [1, 3, 160, 160];
let yoloInputName = null;
let yoloProviders = [];
const yoloScoreThresh = 0.25;   // fixed threshold
const yoloNmsIouThresh = 0.45; // NMS IoU threshold

const scoreThreshold = 0.60; // card positioning threshold (our combined score)
const sharpnessThreshold = 35; // Slightly lower for more flexibility
const requiredStableFrames = 15; // Increase to get more samples
const inFrameThreshold = 0.7;
// Add aspect ratio constants
const CARD_RATIO = 1.59; // Standard card ratio (86mm / 54mm)
const RATIO_TOLERANCE = 0.15; // Allow 15% deviation from standard ratio
let stableCount = 0, bestScore = 0.0, bestContour = null;
let frameHistory = [];
const historySize = 10; // Increase history size to capture more frames

// Add at top with other variables
let frameSkipCounter = 0;
let PROCESS_EVERY_N_FRAMES = 2;
let EXPENSIVE_CHECK_EVERY_N = 2;
let reusableTempCanvas = null; // Reuse canvas
let yoloCanvas = null; // For letterbox preprocessing

// Worker-based YOLO offload
let yoloWorker = null;
let yoloWorkerReady = false;
let yoloRequestInFlight = false;
let lastDetections = [];
let lastYoloMs = 0;
let modelPath = 'best160.onnx'; // auto-chosen model

// Lightweight profiling toggles/holders
let PROFILE = true; // set false to disable timing updates
let lastProfile = {
  grabMs: 0,
  yoloMs: 0,
  sharpMs: 0,
  reflMs: 0,
  drawMs: 0,
  totalMs: 0
};

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
    // Improved mobile-optimized constraints
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    const constraints = {
      video: {
        width: { 
          min: 640,
          ideal: isMobile ? 1920 : 1280,  // Higher resolution for mobile
          max: 4096 
        },
        height: { 
          min: 480,
          ideal: isMobile ? 1080 : 720,   // Higher resolution for mobile
          max: 2160 
        },
        facingMode: "environment",
        frameRate: { 
          ideal: isMobile ? 24 : 30,      // Slightly lower FPS for better quality
          max: 30 
        },
        // Add these for better quality on mobile
        aspectRatio: { ideal: 16/9 },
        focusMode: "continuous",
        whiteBalanceMode: "continuous"
      },
      audio: false
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = cameraStream;
    video.play();
    
    video.onloadedmetadata = () => {
        setTimeout(() => {
            const w = video.videoWidth;
            const h = video.videoHeight;
            console.log('Video dimensions:', w, 'x', h);
            console.log('Device type:', isMobile ? 'Mobile' : 'Desktop');
            
            if (!w || !h) {
                logStatus("Waiting for video size...");
                setTimeout(video.onloadedmetadata, 100);
                return;
            }
            initializeMats();
            isProcessing = true;
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
  
  // Maintain original resolution for canvas
  canvasOutput.width = w;
  canvasOutput.height = h;
  
  // Improve canvas rendering quality
  const ctx = canvasOutput.getContext('2d');
  ctx.imageSmoothingEnabled = false;  // Disable smoothing for sharper output
  // Alternative: use high-quality smoothing
  // ctx.imageSmoothingEnabled = true;
  // ctx.imageSmoothingQuality = 'high';
  
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
  // Fast path: compute Laplacian variance on bbox ROI instead of masked full frame
  // Convert contour (4-point rect) to bounding rect
  const rect = cv.boundingRect(contour);
  const x = Math.max(0, rect.x), y = Math.max(0, rect.y);
  const w = Math.max(1, Math.min(frameMatColor.cols - x, rect.width));
  const h = Math.max(1, Math.min(frameMatColor.rows - y, rect.height));
  const roi = frameMatColor.roi(new cv.Rect(x, y, w, h));
  const gray = new cv.Mat();
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
  const lap = new cv.Mat();
  cv.Laplacian(gray, lap, cv.CV_64F);
  const mean = new cv.Mat();
  const stddev = new cv.Mat();
  // No mask needed when using ROI
  cv.meanStdDev(lap, mean, stddev);
  const variance = Math.pow(stddev.doubleAt(0,0), 2);
  // cleanup
  roi.delete(); gray.delete(); lap.delete(); mean.delete(); stddev.delete();
  // scale to 0-100 similarly
  return Math.min(variance / 5.0, 100.0);
}

function detectReflection(frameMatColor, contour) {
  // Fast reflection check on bbox ROI
  const rect = cv.boundingRect(contour);
  const rx = Math.max(0, rect.x), ry = Math.max(0, rect.y);
  const rw = Math.max(1, Math.min(frameMatColor.cols - rx, rect.width));
  const rh = Math.max(1, Math.min(frameMatColor.rows - ry, rect.height));
  const roi = frameMatColor.roi(new cv.Rect(rx, ry, rw, rh));
  const gray = new cv.Mat();
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);

  const meanMat = new cv.Mat();
  const stddevMat = new cv.Mat();
  cv.meanStdDev(gray, meanMat, stddevMat);
  const meanBrightness = meanMat.doubleAt(0, 0);
  const stddevBrightness = stddevMat.doubleAt(0, 0);

  const spikeThreshold = Math.min(meanBrightness + 2 * stddevBrightness, 245);
  const bright = new cv.Mat();
  cv.threshold(gray, bright, spikeThreshold, 255, cv.THRESH_BINARY);
  const brightPixels = cv.countNonZero(bright);
  const totalPixels = rw * rh;
  const reflectionRatio = brightPixels / totalPixels;

  // Optionally run expensive spike analysis only on desktop
  let hasSpikeReflection = false;
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const cents = new cv.Mat();
  cv.connectedComponentsWithStats(bright, labels, stats, cents);
  // Heuristic: any compact component within 0.1% - 5% area
  const minSpike = totalPixels * 0.001;
  const maxSpike = totalPixels * 0.05;
  const rows = stats.rows;
  for (let i = 1; i < rows; i++) {
    const area = stats.intAt(i, cv.CC_STAT_AREA);
    const bw = stats.intAt(i, cv.CC_STAT_WIDTH);
    const bh = stats.intAt(i, cv.CC_STAT_HEIGHT);
    const density = area / (bw * bh);
    if (area > minSpike && area < maxSpike && density > 0.3) { hasSpikeReflection = true; break; }
  }
  labels.delete(); stats.delete(); cents.delete();

  // cleanup
  roi.delete(); gray.delete(); meanMat.delete(); stddevMat.delete(); bright.delete();

  return {
    hasReflection: reflectionRatio > 0.15 || hasSpikeReflection,
    reflectionRatio,
    hasSpikeReflection
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
  // Try to init Web Worker first for multi-thread speedup
  try {
    yoloWorker = new Worker('yoloWorker.js');
  const allowWebGPU = !!navigator.gpu; // allow WebGPU
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
      } else if (m.type === 'error') {
        console.warn('YOLO worker error:', m.error);
        yoloRequestInFlight = false;
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
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
      ort.env.wasm.simd = true;
      const coi = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
      ort.env.wasm.numThreads = coi ? Math.min(4, (navigator.hardwareConcurrency || 4)) : 1;
    }
  } catch (e) { /* ignore */ }

  const ep = [];
  const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
  if (navigator.gpu) ep.push('webgpu');
  if (gl) ep.push('webgl');
  ep.push('wasm');
  yoloProviders = ep;

  const so = { executionProviders: ep, graphOptimizationLevel: 'all' };
  ortSession = await ort.InferenceSession.create(modelPath, so);
  yoloInputName = ortSession.inputNames[0];

  if (!yoloCanvas) {
    if (typeof OffscreenCanvas !== 'undefined') yoloCanvas = new OffscreenCanvas(yoloInputShape[3], yoloInputShape[2]);
    else yoloCanvas = document.createElement('canvas');
  }
  yoloCanvas.width = yoloInputShape[3];
  yoloCanvas.height = yoloInputShape[2];

  console.log('YOLO main-thread initialized (fixed 160x160):', { providers: yoloProviders, inputName: yoloInputName, inputShape: yoloInputShape, modelPath, yoloScoreThresh });
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
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'low';
  ctx.drawImage(srcCanvas, 0, 0, srcW, srcH, dw, dh, newW, newH);
  return { scale: r, padX: dw, padY: dh, newW, newH };
}

function preprocessForYolo(srcCanvas) {
  // yoloInputShape is [N, C, H, W]
  const [n, c, H, W] = yoloInputShape;
  const info = letterboxToCanvas(srcCanvas, yoloCanvas, W, H);
  const ctx = yoloCanvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, W, H);
  const data = imageData.data; // RGBA
  const chw = new Float32Array(n * c * H * W);
  // Convert to RGB, normalize 0..1, NCHW
  const wh = W * H;
  for (let i = 0; i < wh; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = r;            // R channel
    chw[i + wh] = g;       // G channel
    chw[i + 2 * wh] = b;   // B channel
  }
  const input = new ort.Tensor('float32', chw, [n, c, H, W]);
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
  // yoloInputShape=[N,C,H,W]
  const inH = yoloInputShape[2];
  const inW = yoloInputShape[3];

  // Heuristic to detect layout without assuming large N. Decide by which axis looks like 'no' (<=256).
  if (dims.length === 3) {
    const a = dims[1];
    const b = dims[2];
    if (b >= 6 && b <= 256) {
      // [1, N=a, no=b] (YOLOv5 style with obj then classes)
      const N = a; const no = b; const numClasses = no - 5;
      let needSigmoid = false;
      const probe = Math.min(10, N);
      for (let k = 0; k < probe; k++) { if (data[k*no + 4] > 1) { needSigmoid = true; break; } }
      for (let i = 0; i < N; i++) {
        const off = i * no;
        let cx = data[off + 0]; let cy = data[off + 1]; let w = data[off + 2]; let h = data[off + 3];
        let obj = data[off + 4]; if (needSigmoid) obj = sigmoid(obj);
        let best = 0, bestIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          let v = data[off + 5 + c]; if (needSigmoid) v = sigmoid(v);
          if (v > best) { best = v; bestIdx = c; }
        }
        const conf = obj * (numClasses > 0 ? best : 1);
        if (conf < yoloScoreThresh) continue;
        let [x1, y1, x2, y2] = xywh2xyxy(cx, cy, w, h);
        x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain; x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
        x1 = Math.max(0, Math.min(origW-1, x1)); y1 = Math.max(0, Math.min(origH-1, y1));
        x2 = Math.max(0, Math.min(origW-1, x2)); y2 = Math.max(0, Math.min(origH-1, y2));
        boxes.push([x1,y1,x2,y2]); scores.push(conf); classes.push(bestIdx);
      }
    } else if (a >= 6 && a <= 256) {
      // [1, no=a, N=b] (YOLOv8 style: first 4 rows boxes, others class probs)
      const no = a; const N = b; const numClasses = no - 4;
      let needSigmoid = false;
      const probe = Math.min(10, N);
      for (let k = 0; k < probe; k++) { if (data[4*N + k] > 1) { needSigmoid = true; break; } }
      for (let i = 0; i < N; i++) {
        const cx = data[0 * N + i]; const cy = data[1 * N + i]; const w  = data[2 * N + i]; const h  = data[3 * N + i];
        let best = 0, bestIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          let v = data[(4 + c) * N + i]; if (needSigmoid) v = sigmoid(v);
          if (v > best) { best = v; bestIdx = c; }
        }
        const conf = best; if (conf < yoloScoreThresh) continue;
        let [x1, y1, x2, y2] = xywh2xyxy(cx, cy, w, h);
        x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain; x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
        x1 = Math.max(0, Math.min(origW-1, x1)); y1 = Math.max(0, Math.min(origH-1, y1));
        x2 = Math.max(0, Math.min(origW-1, x2)); y2 = Math.max(0, Math.min(origH-1, y2));
        boxes.push([x1,y1,x2,y2]); scores.push(conf); classes.push(bestIdx);
      }
    } else {
      console.warn('Unexpected YOLO output dims (small N heuristic failed):', dims);
    }
  } else {
    console.warn('Unexpected YOLO output rank:', dims);
  }

  // NMS
  const keep = nms(boxes, scores, yoloNmsIouThresh, 20);
  const dets = keep.map(i => ({ box: boxes[i], score: scores[i], cls: classes[i] }));
  return dets;
}

async function runYoloOnCanvas(srcCanvas) {
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
    // Return last known detections immediately
    return lastDetections || [];
  }

  if (!ortSession) return [];
  const y0 = performance.now();
  const { input, info } = preprocessForYolo(srcCanvas);
  const feeds = {}; feeds[yoloInputName] = input;
  const output = await ortSession.run(feeds);
  const y1 = performance.now();
  lastYoloMs = y1 - y0;
  // Use the first output tensor
  const first = output[Object.keys(output)[0]];
  // Debug dims and sample values to help diagnose tiny-model issues
  try {
    console.debug('YOLO output dims:', first && first.dims, 'providers:', yoloProviders, 'inputShape:', yoloInputShape);
  } catch(e) {}
  const dets = postprocessYolo(first, info, srcCanvas.width, srcCanvas.height);
  if (!dets || dets.length === 0) {
    console.warn('YOLO produced no detections at threshold', yoloScoreThresh);
  }
  return dets;
}

// ----------------- main processing loop -----------------
async function processVideo(){
  if (!isProcessing) return;
  
  // Reduce frame skipping on mobile for better quality
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const skipFrames = isMobile ? 1 : 2;  // Process more frames on mobile
  
  frameSkipCounter++;
  if (frameSkipCounter % skipFrames !== 0) {
    scheduleNextFrame(processVideo);
    return;
  }
  
  try {
    const tFrame0 = performance.now();
    
    // Create high-quality canvas for mobile
    if (!reusableTempCanvas) {
      reusableTempCanvas = document.createElement('canvas');
    }
    reusableTempCanvas.width = video.videoWidth;
    reusableTempCanvas.height = video.videoHeight;
    
    const tempCtx = reusableTempCanvas.getContext('2d', { 
      willReadFrequently: true,
      alpha: false  // Better performance
    });
    
    // Improve canvas rendering quality
    tempCtx.imageSmoothingEnabled = false;  // Sharper on mobile
    tempCtx.drawImage(video, 0, 0, reusableTempCanvas.width, reusableTempCanvas.height);

    // Draw to output with high quality settings
    const outCtx = canvasOutput.getContext('2d');
    outCtx.imageSmoothingEnabled = false;  // Maintain sharpness
    outCtx.drawImage(reusableTempCanvas, 0, 0);

    const frameW = reusableTempCanvas.width;
    const frameH = reusableTempCanvas.height;
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
    
  // Draw guide box and hint text with Canvas 2D
  outCtx.strokeStyle = 'rgb(100,100,100)';
  outCtx.lineWidth = 2;
  outCtx.strokeRect(guideX, guideY, guideWidth, guideHeight);
  outCtx.fillStyle = 'rgb(100,100,100)';
  outCtx.font = '16px sans-serif';
  outCtx.fillText('Align card here', guideX + 10, Math.max(20, guideY - 10));

    // === YOLO detection instead of Canny/contours ===
    let detections = [];
    // Run YOLO (non-blocking via worker if available)
    if (yoloWorkerReady) {
      // Fire off detection if none in-flight, but don't await
      runYoloOnCanvas(reusableTempCanvas);
      detections = lastDetections || [];
    } else {
      // Fallback: run on main thread and await
      detections = await runYoloOnCanvas(reusableTempCanvas);
    }
    const tAfterYolo = performance.now();

  // Pick best detection
    if (detections.length > 0) {
      // Highest confidence
      detections.sort((a,b)=>b.score-a.score);
      const det = detections[0];
      const [x1,y1,x2,y2] = det.box;

      // Check aspect ratio first
      const ratioCheck = isValidCardRatio(det.box);
      const hasValidRatio = ratioCheck.isValid;

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

      const score = calculateCardScore(bestLocalContour, {width: frameW, height: frameH});
      
      // Only calculate sharpness/reflection if score AND ratio are good enough
      let sharpness = 0;
      let reflectionData = { hasReflection: false, reflectionRatio: 0 };
      let srcMatForThisFrame = null;
      let doExpensive = true;
      if (score > scoreThreshold - 0.1 && hasValidRatio && doExpensive) { // Add ratio check
        // Only create OpenCV Mat when needed (lazy)
        srcMatForThisFrame = cv.imread(reusableTempCanvas);
        const tS0 = performance.now();
        sharpness = calculateSharpness(srcMatForThisFrame, bestLocalContour);
        const tS1 = performance.now();
        if (PROFILE) lastProfile.sharpMs = tS1 - tS0;
        const isSharpNow = sharpness >= sharpnessThreshold;
        // Only check reflection if sharp enough
        if (isSharpNow) {
          const tR0 = performance.now();
          reflectionData = detectReflection(srcMatForThisFrame, bestLocalContour);
          const tR1 = performance.now();
          if (PROFILE) lastProfile.reflMs = tR1 - tR0;
        }
      }
      
      const isSharp = sharpness >= sharpnessThreshold;
      const inFrame = isCardInFrame(bestLocalContour, guideBox);
      const hasReflection = reflectionData.hasReflection;
      const isGood = (score > scoreThreshold) && isSharp && inFrame && !hasReflection && hasValidRatio; // Add ratio check

      // Colors - show red if ratio is invalid
      const colorStr = isGood ? 'rgb(0,255,0)' : 
                      (!hasValidRatio ? 'rgb(255,0,255)' : // Magenta for bad ratio
                      (inFrame ? 'rgb(255,165,0)' : 'rgb(255,0,0)'));
      outCtx.fillStyle = colorStr;
      outCtx.font = '18px sans-serif';
      outCtx.fillText(`Score: ${score.toFixed(2)}`, 10, 30);
      outCtx.fillStyle = (isSharp ? 'rgb(0,255,0)' : 'rgb(255,0,0)');
      outCtx.font = '16px sans-serif';
      outCtx.fillText(`Sharp: ${sharpness.toFixed(1)}`, 10, 60);
      outCtx.fillStyle = (inFrame ? 'rgb(0,255,0)' : 'rgb(255,0,0)');
      outCtx.fillText(`In Frame: ${inFrame?'YES':'NO'}`, 10, 90);
      outCtx.fillStyle = (hasReflection ? 'rgb(255,0,0)' : 'rgb(0,255,0)');
      outCtx.fillText(`Reflection: ${(reflectionData.reflectionRatio*100).toFixed(1)}%`, 10, 120);
      
      // Add ratio display
      outCtx.fillStyle = (hasValidRatio ? 'rgb(0,255,0)' : 'rgb(255,0,255)');
      outCtx.fillText(`Ratio: ${ratioCheck.ratio.toFixed(2)} (${ratioCheck.orientation})`, 10, 150);
      outCtx.fillStyle = (hasValidRatio ? 'rgb(0,255,0)' : 'rgb(255,0,255)');
      outCtx.fillText(`Expected: ${CARD_RATIO.toFixed(2)} ±${RATIO_TOLERANCE.toFixed(2)}`, 10, 180);

      if (isGood) {
        // Calculate quality score for comparison
        const qualityScore = calculateQualityScore(score, sharpness, reflectionData.reflectionRatio);
        
        // Store frame with quality score (ensure we have a Mat)
        if (!srcMatForThisFrame) srcMatForThisFrame = cv.imread(reusableTempCanvas);
        frameHistory.push({
          frame: srcMatForThisFrame.clone(),
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
        
        outCtx.fillStyle = 'rgb(255,255,0)';
        outCtx.fillText(`Quality: ${qualityScore.toFixed(2)}`, 10, 210);
        outCtx.fillStyle = 'rgb(0,255,255)';
        outCtx.fillText(`Stable: ${stableCount}/${requiredStableFrames}`, 10, 240);

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
          // Draw notice on final capture and show once via OpenCV (OK to use cv here)
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
          outCtx.fillStyle = 'rgb(255,0,0)';
          if (!inFrame) outCtx.fillText('Move card INTO the frame', 10, 270);
          if (!isSharp) outCtx.fillText('Image too blurry - hold steady', 10, 300);
          if (hasReflection) { outCtx.fillStyle = 'rgb(255,165,0)'; outCtx.fillText('Light reflection detected - adjust angle', 10, 330); }
          if (score <= scoreThreshold) { outCtx.fillStyle = 'rgb(0,0,255)'; outCtx.fillText('Position card better in frame', 10, 360); }
          if (!hasValidRatio) { outCtx.fillStyle = 'rgb(255,0,255)'; outCtx.fillText('Card shape not recognized - adjust angle', 10, 390); }
        }
      }

      // Draw detection rectangle (Canvas 2D) - use magenta for invalid ratio
      outCtx.strokeStyle = hasValidRatio ? 'rgb(0,255,0)' : 'rgb(255,0,255)';
      outCtx.lineWidth = 2;
      outCtx.strokeRect(Math.round(x1), Math.round(y1), Math.round(x2-x1), Math.round(y2-y1));
      bestLocalContour.delete();
      if (srcMatForThisFrame) { srcMatForThisFrame.delete(); srcMatForThisFrame = null; }
    } else {
      stableCount = 0;
      frameHistory.forEach(f => {
        f.frame.delete();
        f.contour.delete();
      });
      frameHistory = [];
      
      outCtx.fillStyle = 'rgb(0,0,255)';
      outCtx.font = '18px sans-serif';
      outCtx.fillText('No card detected', 10, 30);
      outCtx.font = '16px sans-serif';
      outCtx.fillText('Place card in the frame', 10, 60);
    }
    const tD0 = performance.now();
    // We already drew to outCtx; treat the overlay duration as draw time
    const tD1 = performance.now();
    if (PROFILE) lastProfile.drawMs = tD1 - tD0;
    if (PROFILE) {
      lastProfile.grabMs = tAfterGrab - tFrame0;
      lastProfile.yoloMs = lastYoloMs || (tAfterYolo - tAfterGrab);
      lastProfile.totalMs = (performance.now() - tFrame0);
      // Log occasionally to avoid spamming
      if ((window.__profTick = (window.__profTick||0)+1) % 30 === 0) {
        console.log(`Perf: total=${lastProfile.totalMs.toFixed(1)}ms, grab=${lastProfile.grabMs.toFixed(1)}ms, yolo=${lastProfile.yoloMs.toFixed(1)}ms, sharp=${lastProfile.sharpMs.toFixed(1)}ms, refl=${lastProfile.reflMs.toFixed(1)}ms, draw=${lastProfile.drawMs.toFixed(1)}ms`);
      }
    }

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
      let perfStr = '';
      if (PROFILE) {
        perfStr = ` | Proc ${lastProfile.totalMs.toFixed(0)}ms (grab ${lastProfile.grabMs.toFixed(0)}, yolo ${lastProfile.yoloMs.toFixed(0)}, sharp ${lastProfile.sharpMs.toFixed(0)}, refl ${lastProfile.reflMs.toFixed(0)}, draw ${lastProfile.drawMs.toFixed(0)})`;
      } else if (lastYoloMs) {
        perfStr = ` | YOLO ${lastYoloMs.toFixed(0)}ms`;
      }
      statusEl.innerText = `${base} | FPS ~ ${fps}${perfStr}`;
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

// New function to check aspect ratio of detected card
function isValidCardRatio(box) {
  const [x1, y1, x2, y2] = box;
  const width = Math.abs(x2 - x1);
  const height = Math.abs(y2 - y1);
  
  if (width === 0 || height === 0) return { isValid: false, ratio: 0, orientation: 'unknown' };
  
  // Calculate both possible ratios (landscape and portrait)
  const ratio1 = width / height;  // landscape
  const ratio2 = height / width;  // portrait
  
  // Check if either orientation matches card ratio within tolerance
  const landscapeMatch = Math.abs(ratio1 - CARD_RATIO) <= RATIO_TOLERANCE;
  const portraitMatch = Math.abs(ratio2 - CARD_RATIO) <= RATIO_TOLERANCE;
  
  const isValid = landscapeMatch || portraitMatch;
  const actualRatio = ratio1 > ratio2 ? ratio1 : ratio2; // Use the larger ratio
  const orientation = ratio1 > 1 ? 'landscape' : 'portrait';
  
  return { isValid, ratio: actualRatio, orientation };
}