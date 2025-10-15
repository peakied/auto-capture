// script.js
let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let statusEl = document.getElementById('status');
let saveBtn = document.getElementById('saveBtn');
let toggleBlurWarn = document.getElementById('toggleBlurWarn');
let toggleCameraBtn = document.getElementById('toggleCameraBtn');

let streaming = false;
let src = null, gray = null, blurred = null, edges = null;
let contours = null, hierarchy = null;
let cap = null;
let bestCropped = null;
let cameraStream = null;
let isProcessing = false;
let capturedFrame = null; // Store the captured frame

const scoreThreshold = 0.60; // Slightly lower threshold
const sharpnessThreshold = 35; // Slightly lower for more flexibility
const requiredStableFrames = 15; // Increase to get more samples
const inFrameThreshold = 0.7;
let stableCount = 0, bestScore = 0.0, bestContour = null;
let frameHistory = [];
const historySize = 10; // Increase history size to capture more frames

// Add at top with other variables
let frameSkipCounter = 0;
const PROCESS_EVERY_N_FRAMES = 3; // Process every 3rd frame
let reusableTempCanvas = null; // Reuse canvas

function logStatus(s){ statusEl.innerText = s; }

// wait for OpenCV to be ready
function onOpenCvReady() {
  if (typeof cv === 'undefined') {
    logStatus("Waiting for OpenCV...");
    setTimeout(onOpenCvReady, 100);
    return;
  }
  if (cv.getBuildInformation) {
    logStatus("OpenCV.js loaded - Click 'Start Camera' to begin");
    toggleCameraBtn.disabled = false;
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
    // Request maximum quality settings for best camera performance
    const constraints = {
      video: {
        // Resolution: Try for 4K, fallback to 1080p, then 720p
        width: { ideal: 3840, min: 1280 },
        height: { ideal: 2160, min: 720 },
        
        // Use rear camera (usually higher quality)
        facingMode: { ideal: "environment" },
        
        // Frame rate: Higher = better motion but may reduce quality per frame
        // 30fps is a good balance for document capture
        frameRate: { ideal: 30, max: 60 },
        
        // Request best possible quality settings
        aspectRatio: { ideal: 16/9 },
        
        // Advanced constraints (browser support varies)
        focusMode: { ideal: "continuous" },  // Continuous autofocus
        exposureMode: { ideal: "continuous" }, // Auto exposure
        whiteBalanceMode: { ideal: "continuous" }, // Auto white balance
        
        // Prefer hardware acceleration
        resizeMode: { ideal: "none" } // Don't downsample at source
      },
      audio: false
    };
    
    try {
      // Try with all advanced features first
      cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.log("Advanced constraints failed, trying basic high-res:", err);
      // Fallback: Try without advanced constraints
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 3840, min: 1280 },
          height: { ideal: 2160, min: 720 },
          facingMode: "environment",
          frameRate: { ideal: 30 }
        },
        audio: false
      });
    }
    
    // Log actual capabilities achieved
    const videoTrack = cameraStream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();
    console.log("Camera settings achieved:", {
      resolution: `${settings.width}x${settings.height}`,
      frameRate: settings.frameRate,
      facingMode: settings.facingMode
    });
    
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
            // Give one more frame delay before processing
            setTimeout(() => requestAnimationFrame(processVideo), 100);
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
  edges = new cv.Mat();
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
  
  // ตรวจจับ spike reflections โดยหา connected components
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const numLabels = cv.connectedComponentsWithStats(cardRegion, labels, stats, centroids);
  
  let hasSpikeReflection = false;
  let maxSpikeIntensity = 0;
  
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
      cv.compare(spikeMask, new cv.Mat(height, width, spikeMask.type(), new cv.Scalar(i)), spikeMaskBinary, cv.CMP_EQ);
      
      const spikeMean = new cv.Mat();
      cv.meanStdDev(spikeROI, spikeMean, new cv.Mat(), spikeMaskBinary);
      const spikeIntensity = spikeMean.doubleAt(0, 0);
      
      // Spike ต้องสว่างกว่าค่าเฉลี่ยอย่างน้อย 40 units
      if (spikeIntensity - meanBrightness > 40) {
        hasSpikeReflection = true;
        maxSpikeIntensity = Math.max(maxSpikeIntensity, spikeIntensity - meanBrightness);
      }
      
      spikeMaskBinary.delete();
      spikeMean.delete();
    }
  }
  
  // cleanup
  mask.delete();
  contourVec.delete();
  grayLocal.delete();
  cardRegionGray.delete();
  meanMat.delete();
  stddevMat.delete();
  brightThreshold.delete();
  cardRegion.delete();
  labels.delete();
  stats.delete();
  centroids.delete();
  
  // ตรวจพบแสงสะท้อนถ้า:
  // 1. มีพื้นที่สว่างเกินไปมากกว่า 15%
  // 2. หรือมี spike reflection ที่สว่างกว่าค่าเฉลี่ยมาก
  return {
    hasReflection: reflectionRatio > 0.3 || hasSpikeReflection,
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
  // Apply moderate sharpening filter
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

function sharpenImageStrong(mat) {
  // Apply STRONG sharpening - two-pass approach for best quality
  
  // Pass 1: Unsharp mask (professional sharpening technique)
  const blurred = new cv.Mat();
  cv.GaussianBlur(mat, blurred, new cv.Size(0, 0), 1.5);
  
  const unsharpMask = new cv.Mat();
  cv.addWeighted(mat, 1.5, blurred, -0.5, 0, unsharpMask);
  blurred.delete();
  
  // Pass 2: High-pass sharpening kernel for edges
  const kernel = cv.matFromArray(3, 3, cv.CV_32FC1, [
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
  ]);
  
  const finalSharpened = new cv.Mat();
  cv.filter2D(unsharpMask, finalSharpened, cv.CV_8U, kernel);
  
  kernel.delete();
  unsharpMask.delete();
  
  return finalSharpened;
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
  
  // CRITICAL: Set MINIMUM output resolution for sharp results
  // Standard credit card is 85.6 × 53.98 mm
  // Target: At least 1000px on long side (higher quality)
  const MIN_LONG_SIDE = 1000;
  const MIN_SHORT_SIDE = Math.floor(MIN_LONG_SIDE / card_ratio);
  
  // คำนวณขนาดสุดท้าย - use ACTUAL size or minimum, whichever is larger
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
    
    // Upscale if too small
    if (finalWidth < MIN_LONG_SIDE) {
      const scale = MIN_LONG_SIDE / finalWidth;
      finalWidth = MIN_LONG_SIDE;
      finalHeight = Math.floor(finalHeight * scale);
    }
  } else {
    // portrait orientation
    const portrait_ratio = 54 / 86;
    if (current_ratio > portrait_ratio) {
      finalWidth = Math.floor(finalHeight * portrait_ratio);
    } else {
      finalHeight = Math.floor(finalWidth / portrait_ratio);
    }
    
    // Upscale if too small
    if (finalHeight < MIN_LONG_SIDE) {
      const scale = MIN_LONG_SIDE / finalHeight;
      finalHeight = MIN_LONG_SIDE;
      finalWidth = Math.floor(finalWidth * scale);
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
  
  // Apply perspective transformation with HIGH-QUALITY interpolation
  const warped = new cv.Mat();
  // CRITICAL: Use INTER_CUBIC for much better quality (vs default INTER_LINEAR)
  // INTER_CUBIC is slower but produces much sharper results
  cv.warpPerspective(
    frameMat, 
    warped, 
    M, 
    new cv.Size(finalWidth, finalHeight),
    cv.INTER_CUBIC,  // High-quality bicubic interpolation
    cv.BORDER_CONSTANT,
    new cv.Scalar()
  );

  // cleanup
  srcPts.delete();
  dst.delete();
  M.delete();
  
  // Apply STRONGER sharpening to improve clarity
  const sharpened = sharpenImageStrong(warped);
  warped.delete();
  
  return sharpened;
}

// ----------------- main processing loop -----------------
function processVideo(){
  if (!isProcessing) return;
  
  // Skip frames to reduce CPU load
  frameSkipCounter++;
  if (frameSkipCounter % PROCESS_EVERY_N_FRAMES !== 0) {
    requestAnimationFrame(processVideo);
    return;
  }
  
  try {
    // Reuse canvas instead of creating new one
    if (!reusableTempCanvas) {
      reusableTempCanvas = document.createElement('canvas');
      // Configure canvas context for high quality
      const ctx = reusableTempCanvas.getContext('2d', {
        alpha: false,
        desynchronized: false,
        willReadFrequently: true // We read frequently for processing
      });
      // Store context reference
      reusableTempCanvas._ctx = ctx;
    }
    
    reusableTempCanvas.width = video.videoWidth;
    reusableTempCanvas.height = video.videoHeight;
    const tempCtx = reusableTempCanvas._ctx || reusableTempCanvas.getContext('2d');
    
    // Use high-quality image smoothing for better capture
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    
    // Capture frame with full quality
    tempCtx.drawImage(video, 0, 0, reusableTempCanvas.width, reusableTempCanvas.height);
    
    if (src) src.delete();
    src = cv.imread(reusableTempCanvas);
    
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5,5), 0);
    
    const thresh = new cv.Mat();
    cv.adaptiveThreshold(blurred, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);
    
    cv.Canny(thresh, edges, 30, 100);
    thresh.delete();

    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3,3));
    cv.dilate(edges, edges, kernel, new cv.Point(-1,-1), 1);
    cv.erode(edges, edges, kernel, new cv.Point(-1,-1), 1);

    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

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

    let bestLocalScore = 0, bestLocalContour = null;
    for (let i=0;i<contours.size();i++){
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < (src.rows * src.cols) * 0.02 || area > (src.rows*src.cols)*0.85) { 
        cnt.delete(); 
        continue; 
      }
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.015 * peri, true);
      if (approx.rows === 4) {
        const rect = cv.boundingRect(approx);
        const aspect = rect.width / rect.height;
        if (0.3 < aspect && aspect < 3.0){
          const score = calculateCardScore(approx, {width: src.cols, height: src.rows});
          if (score > bestLocalScore) { 
            bestLocalScore = score;
            if (bestLocalContour) bestLocalContour.delete();
            bestLocalContour = approx.clone(); 
          }
        }
      }
      approx.delete(); 
      cnt.delete();
    }

    if (bestLocalContour) {
      const score = bestLocalScore;
      
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
          let bestFrameIndex = 0;
          for (let i = 1; i < frameHistory.length; i++) {
            if (frameHistory[i].qualityScore > bestFrameData.qualityScore) {
              bestFrameData = frameHistory[i];
              bestFrameIndex = i;
            }
          }
          
          console.log('=== FRAME SELECTION ===');
          console.log(`Selected frame ${bestFrameIndex + 1} of ${frameHistory.length} frames`);
          console.log('Best frame stats:');
          console.log('  Quality Score:', bestFrameData.qualityScore.toFixed(3));
          console.log('  Sharpness:', bestFrameData.sharpness.toFixed(1));
          console.log('  Position Score:', bestFrameData.score.toFixed(3));
          console.log('All frames comparison:');
          frameHistory.forEach((f, idx) => {
            console.log(`  Frame ${idx + 1}: Quality=${f.qualityScore.toFixed(3)}, Sharp=${f.sharpness.toFixed(1)}, Score=${f.score.toFixed(3)}${idx === bestFrameIndex ? ' <- SELECTED' : ''}`);
          });
          
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
          // REMOVE OR COMMENT OUT THESE LINES (lines 461-465)
          // const contourVec2 = new cv.MatVector();
          // contourVec2.push_back(bestContour);
          // cv.drawContours(displayFrame, contourVec2, 0, new cv.Scalar(0,255,0,255), 2);
          // contourVec2.delete();
          
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
    kernel.delete();

  } catch (err){
    console.error(err);
    logStatus("Processing error: " + err);
  }

  if (isProcessing) {
    requestAnimationFrame(processVideo); // Remove setTimeout, use requestAnimationFrame directly
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
  
  console.log('=== CROPPING CARD ===');
  console.log('Input frame size:', capturedFrame.cols, 'x', capturedFrame.rows);
  console.log('Contour points:', bestContour.rows);
  
  // Extract the card region จากภาพต้นฉบับที่ไม่มีกรอบ (capturedFrame)
  if (bestCropped) bestCropped.delete();
  bestCropped = extractCardRegion(capturedFrame, bestContour);
  
  console.log('Output cropped size:', bestCropped.cols, 'x', bestCropped.rows);
  console.log('Aspect ratio:', (bestCropped.cols / bestCropped.rows).toFixed(2));
  
  // Show the cropped and sharpened card (ไม่มีกรอบเขียว)
  cv.imshow(canvasOutput, bestCropped);
  logStatus("Card cropped and sharpened! Downloading...");
  
  // Auto-download - Create a temporary canvas for clean, high-quality output
  setTimeout(() => {
    // Create a temporary canvas to ensure clean output
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = bestCropped.cols;
    tempCanvas.height = bestCropped.rows;
    
    // Use 2d context with high-quality settings
    const tempCtx = tempCanvas.getContext('2d', {
      alpha: false, // No transparency = better compression
      desynchronized: false, // Prioritize quality over speed
      willReadFrequently: false
    });
    
    // CRITICAL: DISABLE image smoothing to preserve sharpness
    // Image smoothing blurs the image - we want pixel-perfect rendering
    tempCtx.imageSmoothingEnabled = false;
    
    // Draw the clean cropped image to temp canvas
    cv.imshow(tempCanvas, bestCropped);
    
    // Download with maximum quality PNG
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `card_${timestamp}.png`;
    
    // PNG with maximum quality (lossless)
    // Note: PNG quality parameter is ignored (always lossless), but 1.0 is convention
    const dataUrl = tempCanvas.toDataURL('image/png', 1.0);
    link.href = dataUrl;
    
    console.log('=== SAVING CARD ===');
    console.log('Filename:', link.download);
    console.log('Resolution:', tempCanvas.width, 'x', tempCanvas.height);
    console.log('Total pixels:', (tempCanvas.width * tempCanvas.height).toLocaleString());
    console.log('Format: PNG (lossless)');
    console.log('Estimated file size:', (dataUrl.length * 0.75 / 1024).toFixed(1), 'KB');
    console.log('Sharpening: Strong (2-pass unsharp mask + edge enhancement)');
    console.log('Interpolation: Cubic (high-quality perspective transform)');
    
    link.click();
    
    logStatus("Card saved in high quality! Click 'Start Camera' to capture another card.");
  }, 100);
});                                                                              

// cleanup when page unloads
window.addEventListener('unload', ()=>{
  stopCamera();
  try {
    if (src) src.delete();
    if (gray) gray.delete();
    if (blurred) blurred.delete();
    if (edges) edges.delete();
    if (contours) contours.delete();
    if (hierarchy) hierarchy.delete();
    if (bestContour) bestContour.delete();
    if (bestCropped) bestCropped.delete();
    if (cap) cap.delete();
  } catch(e){}
});
