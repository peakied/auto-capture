// script.js
let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let statusEl = document.getElementById('status');
let toggleCameraBtn = document.getElementById('toggleCameraBtn');
let downloadImageBtn = document.getElementById('downloadImageBtn'); // Add reference to the button

let streaming = false;
let src = null, gray = null, blurred = null, edges = null;
let contours = null, hierarchy = null;
let cap = null;
let cameraStream = null;
let isProcessing = false;
let capturedFrame = null; // Store the captured frame
let croppedCardImage = null; // Store the cropped card image

const scoreThreshold = 0.50; // Lower threshold for easier detection
const sharpnessThreshold = 25; // Lower for more flexibility
const requiredStableFrames = 10; // Reduce for faster detection
const inFrameThreshold = 0.7;
let stableCount = 0, bestScore = 0.0, bestContour = null;
let frameHistory = [];
const historySize = 8; // Reduce history size

// Add at top with other variables
let frameSkipCounter = 0;
const PROCESS_EVERY_N_FRAMES = 2; // Process every 2nd frame for better performance
let reusableTempCanvas = null;
let showCannyMode = false; // Add toggle for Canny visualization

let toggleBlurWarn = document.getElementById('toggleBlurWarn'); // Add missing reference

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
    cameraStream = await navigator.mediaDevices.getUserMedia({ 
      video: {
        width: { ideal: 1280 }, // Request higher resolution for better detection
        height: { ideal: 720 },
        facingMode: "environment"
      }, 
      audio: false 
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
  
  // Set canvas to 50% of video size for display
  canvasOutput.width = w * 0.5;
  canvasOutput.height = h * 0.5;
  
  // Don't use VideoCapture, we'll capture from canvas instead
  cap = null;
  src = null;
  
  gray = new cv.Mat();
  blurred = new cv.Mat();
  edges = new cv.Mat();
  contours = new cv.MatVector();
  hierarchy = new cv.Mat();
  
  console.log('Mats initialized - Canvas size:', canvasOutput.width, 'x', canvasOutput.height);
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
  if (!cardContour || cardContour.isDeleted() || cardContour.rows !== 4) return false;
  
  try {
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
  } catch (error) {
    console.warn('Error in isCardInFrame:', error);
    return false;
  }
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

// Add after helpers: perspective crop from 4-point contour
function crop(imageMatColor, quadContour, options = {}) {
  // Options:
  // - outputWidth: target width of cropped card (default 860)
  // - enforceRatio: keep standard card ratio 86:54 (default true)
  const outputWidth = options.outputWidth ?? 860;
  const enforceRatio = options.enforceRatio ?? true;
  const CARD_RATIO = 86 / 54; // width / height

  if (!imageMatColor || !quadContour || quadContour.rows !== 4) {
    console.warn('crop(): invalid input');
    return null;
  }

  // Extract 4 points from contour mat
  function mat4ToPointsArray(mat) {
    const pts = [];
    for (let i = 0; i < 4; i++) {
      let x, y;
      // Support both CV_32SC2 and CV_32FC2
      if (mat.type() === cv.CV_32SC2) {
        const p = mat.intPtr(i, 0);
        x = p[0]; y = p[1];
      } else {
        const p = mat.floatPtr(i, 0);
        x = p[0]; y = p[1];
      }
      pts.push({ x, y });
    }
    return pts;
  }

  // Order points: [tl, tr, br, bl]
  function orderQuadPoints(pts) {
    // Sum and diff method
    let tl = pts[0], tr = pts[0], br = pts[0], bl = pts[0];
    let minSum = Infinity, maxSum = -Infinity, minDiff = Infinity, maxDiff = -Infinity;
    for (const p of pts) {
      const s = p.x + p.y;
      const d = p.x - p.y;
      if (s < minSum) { minSum = s; tl = p; }
      if (s > maxSum) { maxSum = s; br = p; }
      if (d < minDiff) { minDiff = d; bl = p; }
      if (d > maxDiff) { maxDiff = d; tr = p; }
    }
    return [tl, tr, br, bl];
  }

  const srcPts = orderQuadPoints(mat4ToPointsArray(quadContour));

  // Measure current dimensions
  const widthA  = Math.hypot(srcPts[2].x - srcPts[3].x, srcPts[2].y - srcPts[3].y); // br-bl
  const widthB  = Math.hypot(srcPts[1].x - srcPts[0].x, srcPts[1].y - srcPts[0].y); // tr-tl
  const heightA = Math.hypot(srcPts[1].x - srcPts[2].x, srcPts[1].y - srcPts[2].y); // tr-br
  const heightB = Math.hypot(srcPts[0].x - srcPts[3].x, srcPts[0].y - srcPts[3].y); // tl-bl

  // Decide output size
  let dstW = Math.max(Math.round(Math.max(widthA, widthB)), 100);
  let dstH = Math.max(Math.round(Math.max(heightA, heightB)), 100);

  // If enforceRatio, override using target width and compute height by ratio
  if (enforceRatio) {
    dstW = outputWidth;
    dstH = Math.max(1, Math.round(dstW / CARD_RATIO));
  }

  // Build transform
  const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
    srcPts[0].x, srcPts[0].y, // tl
    srcPts[1].x, srcPts[1].y, // tr
    srcPts[2].x, srcPts[2].y, // br
    srcPts[3].x, srcPts[3].y  // bl
  ]);
  const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    dstW - 1, 0,
    dstW - 1, dstH - 1,
    0, dstH - 1
  ]);

  const M = cv.getPerspectiveTransform(srcQuad, dstQuad);
  const warped = new cv.Mat();
  cv.warpPerspective(imageMatColor, warped, M, new cv.Size(dstW, dstH), cv.INTER_LINEAR, cv.BORDER_REPLICATE);

  // Cleanup
  srcQuad.delete();
  dstQuad.delete();
  M.delete();

  return warped; // Caller must delete()
}

// Enhanced crop function with better error handling and options
function cropCardToStandardSize(imageMatColor, quadContour, options = {}) {
  const outputWidth = options.outputWidth ?? 860;
  const outputHeight = options.outputHeight ?? 540;
  const enforceRatio = options.enforceRatio ?? true;
  const addPadding = options.addPadding ?? 10;

  if (!imageMatColor || imageMatColor.isDeleted() || 
      !quadContour || quadContour.isDeleted() || quadContour.rows !== 4) {
    console.warn('cropCardToStandardSize(): invalid input');
    return null;
  }

  try {
    // Get the basic crop first
    const basicCrop = crop(imageMatColor, quadContour, { outputWidth, enforceRatio });
    if (!basicCrop || basicCrop.isDeleted()) return null;

    // Apply sharpening to enhance the cropped image
    const sharpened = sharpenImage(basicCrop);
    if (basicCrop && !basicCrop.isDeleted()) basicCrop.delete();

    if (!sharpened || sharpened.isDeleted()) return null;

    // Add padding if requested
    if (addPadding > 0) {
      const paddedWidth = sharpened.cols + (addPadding * 2);
      const paddedHeight = sharpened.rows + (addPadding * 2);
      const padded = new cv.Mat(paddedHeight, paddedWidth, sharpened.type(), new cv.Scalar(255, 255, 255, 255));
      
      const roi = padded.roi(new cv.Rect(addPadding, addPadding, sharpened.cols, sharpened.rows));
      sharpened.copyTo(roi);
      if (roi && !roi.isDeleted()) roi.delete();
      if (sharpened && !sharpened.isDeleted()) sharpened.delete();
      
      return padded;
    }

    return sharpened;
  } catch (error) {
    console.error('Error in cropCardToStandardSize:', error);
    return null;
  }
}

// Function to display cropped card
function displayCroppedCard(imageMat, contour) {
  if (!imageMat || imageMat.isDeleted() || !contour || contour.isDeleted()) {
    console.warn('displayCroppedCard: invalid input - imageMat:', !!imageMat, 'contour:', !!contour);
    return false;
  }

  try {
    // Clean up previous cropped image
    if (croppedCardImage && !croppedCardImage.isDeleted()) {
      croppedCardImage.delete();
      croppedCardImage = null;
    }

    // Crop the card to standard size
    croppedCardImage = cropCardToStandardSize(imageMat, contour, {
      outputWidth: 860,
      enforceRatio: true,
      addPadding: 20
    });

    console.log('Cropped image result:', !!croppedCardImage, croppedCardImage ? 'size: ' + croppedCardImage.cols + 'x' + croppedCardImage.rows : 'null');

    if (croppedCardImage && !croppedCardImage.isDeleted()) {
      // Resize canvas to fit cropped card at 50% size for display
      canvasOutput.width = croppedCardImage.cols * 0.5;
      canvasOutput.height = croppedCardImage.rows * 0.5;
      
      // Create a display version at 50% size
      const displayMat = new cv.Mat();
      cv.resize(croppedCardImage, displayMat, new cv.Size(canvasOutput.width, canvasOutput.height));
      
      // Display the resized cropped card
      cv.imshow(canvasOutput, displayMat);
      displayMat.delete();
      
      // Create overlay with text on the canvas context directly
      const ctx = canvasOutput.getContext('2d');
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(5, 5, 200, 50);
      
      ctx.fillStyle = 'white';
      ctx.font = 'bold 12px Arial';
      ctx.fillText('Cropped Card - Standard Size', 10, 20);
      
      ctx.font = '10px Arial';
      ctx.fillText(`Size: ${croppedCardImage.cols} x ${croppedCardImage.rows}`, 10, 32);
      ctx.fillText('Click "Download" to save', 10, 42);
      
      enableDownloadButton();
      console.log('Download button should now be enabled');
      return true;
    } else {
      console.warn('Failed to create cropped image');
    }
  } catch (error) {
    console.error('Error in displayCroppedCard:', error);
  }
  return false;
}

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
    }
    reusableTempCanvas.width = video.videoWidth;
    reusableTempCanvas.height = video.videoHeight;
    const tempCtx = reusableTempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, reusableTempCanvas.width, reusableTempCanvas.height);
    
    if (src && !src.isDeleted()) src.delete();
    src = cv.imread(reusableTempCanvas);
    
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    
    // Improved preprocessing with bilateral filter for noise reduction while preserving edges
    const bilateral = new cv.Mat();
    cv.bilateralFilter(gray, bilateral, 9, 75, 75);
    
    // Use adaptive threshold for better edge detection in varying lighting
    const thresh = new cv.Mat();
    cv.adaptiveThreshold(bilateral, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);
    
    // Enhanced Canny edge detection with adjusted thresholds
    cv.Canny(bilateral, edges, 15, 45, 3, false);
    
    // Enhanced morphological operations for better edge connectivity
    const kernel1 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
    const kernel2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));

    cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel1, new cv.Point(-1, -1), 2);
    cv.dilate(edges, edges, kernel2, new cv.Point(-1, -1), 1);
    cv.erode(edges, edges, kernel2, new cv.Point(-1, -1), 1);

    // Clear previous contours
    if (contours && !contours.isDeleted()) {
      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        if (cnt && !cnt.isDeleted()) cnt.delete();
      }
      contours.delete();
    }
    contours = new cv.MatVector();
    
    if (hierarchy && !hierarchy.isDeleted()) hierarchy.delete();
    hierarchy = new cv.Mat();

    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let display = src.clone();
    
    // Show Canny edges if enabled (press 'C' key to toggle)
    if (showCannyMode) {
      const cannyDisplay = new cv.Mat();
      cv.cvtColor(edges, cannyDisplay, cv.COLOR_GRAY2RGBA);
      cv.addWeighted(display, 0.7, cannyDisplay, 0.3, 0, display);
      cannyDisplay.delete();
      
      cv.putText(display, "Canny Edges (Press C to toggle)", 
                new cv.Point(10, display.rows - 20), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(255, 255, 0, 255), 2);
    }
    
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
    let totalContours = contours.size();
    let validContours = 0;
    let fourPointContours = 0;
    
    // Improved contour filtering with proper memory management
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      if (!cnt || cnt.isDeleted()) continue;
      
      const area = cv.contourArea(cnt);
      const frameArea = src.rows * src.cols;
      
      if (area < frameArea * 0.01 || area > frameArea * 0.9) {
        continue;
      }
      
      validContours++;
      
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      
      try {
        cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
        
        if (approx.rows === 4) {
          fourPointContours++;
          
          const rect = cv.boundingRect(approx);
          const aspect = rect.width / rect.height;
          
          if (0.2 < aspect && aspect < 5.0) {
            const score = calculateCardScore(approx, {width: src.cols, height: src.rows});
            if (score > bestLocalScore) {
              bestLocalScore = score;
              if (bestLocalContour && !bestLocalContour.isDeleted()) {
                bestLocalContour.delete();
              }
              bestLocalContour = approx.clone();
            }
            
            if (showCannyMode) {
              const debugContourVec = new cv.MatVector();
              debugContourVec.push_back(approx);
              cv.drawContours(display, debugContourVec, 0, new cv.Scalar(255, 255, 0, 255), 1);
              debugContourVec.delete();
            }
          }
        }
      } catch (error) {
        console.warn('Error processing contour:', error);
      }
      
      if (approx && !approx.isDeleted()) approx.delete();
    }

    // Add debug information
    cv.putText(display, `Contours: ${totalContours} | Valid: ${validContours} | 4-Point: ${fourPointContours}`, 
               new cv.Point(10, display.rows - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(255, 255, 255, 255), 1);

    if (bestLocalContour && !bestLocalContour.isDeleted()) {
      const score = bestLocalScore;
      
      // Only draw contours when Canny mode is enabled
      if (showCannyMode) {
        const bestContourVec = new cv.MatVector();
        bestContourVec.push_back(bestLocalContour);
        cv.drawContours(display, bestContourVec, 0, new cv.Scalar(0, 255, 255, 255), 3);
        bestContourVec.delete();
      }
      
      // Only calculate sharpness if score is good enough
      let sharpness = 0;
      let reflectionData = { hasReflection: false, reflectionRatio: 0 };
      
      if (score > scoreThreshold - 0.2) {
        sharpness = calculateSharpness(src, bestLocalContour);
        const isSharp = sharpness >= sharpnessThreshold;
        
        if (isSharp) {
          reflectionData = detectReflection(src, bestLocalContour);
        }
      }
      
      const isSharp = sharpness >= sharpnessThreshold;
      const inFrame = isCardInFrame(bestLocalContour, guideBox);
      const hasReflection = reflectionData.hasReflection;
      const isGood = (score > scoreThreshold) && isSharp && inFrame && !hasReflection;

      const color = isGood ? new cv.Scalar(0,255,0,255) : 
                    inFrame ? new cv.Scalar(0,165,255,255) : 
                    new cv.Scalar(255,0,0,255);

      cv.putText(display, `Score: ${score.toFixed(2)} (need: ${scoreThreshold})`, new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color,2);
      cv.putText(display, `Sharp: ${sharpness.toFixed(1)} (need: ${sharpnessThreshold})`, new cv.Point(10,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(isSharp?0:255, isSharp?255:0, 0,255),2);
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
          if (oldest.frame && !oldest.frame.isDeleted()) oldest.frame.delete();
          if (oldest.contour && !oldest.contour.isDeleted()) oldest.contour.delete();
        }
        
        // Update best score based on quality
        if (qualityScore > bestScore) {
          bestScore = qualityScore;
          if (bestContour && !bestContour.isDeleted()) bestContour.delete();
          bestContour = bestLocalContour.clone();
          stableCount = 0;
        } else {
          stableCount++;
        }
        
        cv.putText(display, `Quality: ${qualityScore.toFixed(2)}`, new cv.Point(10,150), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,255,0,255),2);
        cv.putText(display, `Stable: ${stableCount}/${requiredStableFrames}`, new cv.Point(10,180), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,255,255,255),2);

        if (stableCount >= requiredStableFrames) {
          // Find best frame based on quality score
          let bestFrameData = frameHistory[0];
          for (let i = 1; i < frameHistory.length; i++) {
            if (frameHistory[i].qualityScore > bestFrameData.qualityScore) {
              bestFrameData = frameHistory[i];
            }
          }
          
          console.log('Selected frame - Quality:', bestFrameData.qualityScore.toFixed(3), 
                     'Sharpness:', bestFrameData.sharpness.toFixed(1),
                     'Score:', bestFrameData.score.toFixed(3));
          
          if (capturedFrame && !capturedFrame.isDeleted()) capturedFrame.delete();
          capturedFrame = bestFrameData.frame.clone();

          if (bestContour && !bestContour.isDeleted()) bestContour.delete();
          bestContour = bestFrameData.contour.clone();
          
          // Clean up frame history
          frameHistory.forEach(f => {
            if (f.frame && !f.frame.isDeleted()) f.frame.delete();
            if (f.contour && !f.contour.isDeleted()) f.contour.delete();
          });
          frameHistory = [];
          
          stableCount = 0;
          bestScore = 0.0;
          
          isProcessing = false;
          stopCamera();
          
          console.log('Attempting to display cropped card...');
          
          // Display cropped card instead of full frame
          const cropSuccess = displayCroppedCard(capturedFrame, bestContour);
          
          if (!cropSuccess) {
            console.warn('Cropping failed, showing original frame and enabling download anyway');
            // Fallback to original display if cropping fails
            const displayFrame = capturedFrame.clone();
            cv.putText(displayFrame, "Card Detected! (Crop failed)", new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,255,0,255),2);
            cv.imshow(canvasOutput, displayFrame);
            if (displayFrame && !displayFrame.isDeleted()) displayFrame.delete();
            
            // Enable download button even if cropping failed
            enableDownloadButton();
          }

          logStatus("Card detected and cropped! Click 'Download Image' to save.");
        }
      } else {
        stableCount = 0;
        frameHistory.forEach(f => {
          if (f.frame && !f.frame.isDeleted()) f.frame.delete();
          if (f.contour && !f.contour.isDeleted()) f.contour.delete();
        });
        frameHistory = [];
        
        if (toggleBlurWarn.checked) {
          if (!inFrame) cv.putText(display, "Move card INTO the frame", new cv.Point(10,180), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,0,0,255),2);
          if (!isSharp) cv.putText(display, "Image too blurry - hold steady", new cv.Point(10,210), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,0,255,255),2);
          if (hasReflection) cv.putText(display, "Light reflection detected - adjust angle", new cv.Point(10,240), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255,165,0,255),2);
          if (score <= scoreThreshold) cv.putText(display, "Position card better in frame", new cv.Point(10,270), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,0,255,255),2);
        }
      }

      if (bestLocalContour && !bestLocalContour.isDeleted()) {
        bestLocalContour.delete();
      }
    } else {
      stableCount = 0;
      frameHistory.forEach(f => {
        if (f.frame && !f.frame.isDeleted()) f.frame.delete();
        if (f.contour && !f.contour.isDeleted()) f.contour.delete();
      });
      frameHistory = [];
      
      cv.putText(display, "No rectangular card detected", new cv.Point(10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, new cv.Scalar(0,0,255,255),2);
      cv.putText(display, "Try better lighting or card positioning", new cv.Point(10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(0,0,255,255),2);
      cv.putText(display, "Press 'C' to toggle Canny edge view", new cv.Point(10,90), cv.FONT_HERSHEY_SIMPLEX, 0.6, new cv.Scalar(255,255,0,255),2);
    }

    cv.imshow(canvasOutput, display);
    
    // Cleanup temporary matrices
    if (display && !display.isDeleted()) display.delete();
    if (bilateral && !bilateral.isDeleted()) bilateral.delete();
    if (thresh && !thresh.isDeleted()) thresh.delete();
    if (kernel1 && !kernel1.isDeleted()) kernel1.delete();
    if (kernel2 && !kernel2.isDeleted()) kernel2.delete();

  } catch (err){
    console.error('Processing error:', err);
    logStatus("Processing error: " + err);
  }

  if (isProcessing) {
    requestAnimationFrame(processVideo);
  }
}

// Enhanced download function with better fallback
function downloadImage() {
  let imageToDownload = null;
  let needsCleanup = false;
  let filename = 'card';

  console.log('Download attempt - croppedCardImage exists:', !!croppedCardImage, 
              'capturedFrame exists:', !!capturedFrame, 
              'bestContour exists:', !!bestContour);

  if (croppedCardImage && !croppedCardImage.isDeleted()) {
    console.log('Using existing cropped image');
    imageToDownload = croppedCardImage;
    filename = 'cropped-card';
  } else if (capturedFrame && !capturedFrame.isDeleted() && 
             bestContour && !bestContour.isDeleted()) {
    console.log('Creating cropped image on demand');
    imageToDownload = cropCardToStandardSize(capturedFrame, bestContour, {
      outputWidth: 860,
      enforceRatio: true,
      addPadding: 20
    });
    needsCleanup = true;
    filename = 'cropped-card';
  } else if (capturedFrame && !capturedFrame.isDeleted()) {
    console.log('Using original captured frame as fallback');
    imageToDownload = capturedFrame;
    filename = 'original-card';
  }

  if (!imageToDownload || imageToDownload.isDeleted()) {
    console.warn("No image available to download.");
    logStatus("No image available to download.");
    return;
  }

  try {
    // Create a clean temporary canvas for download (without overlay text)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageToDownload.cols;
    tempCanvas.height = imageToDownload.rows;
    
    // Show the clean image on temporary canvas
    cv.imshow(tempCanvas, imageToDownload);

    // Create download link
    const link = document.createElement('a');
    link.href = tempCanvas.toDataURL('image/png');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    link.download = `${filename}-${timestamp}.png`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Cleanup
    tempCanvas.remove();
    
    if (needsCleanup && imageToDownload && !imageToDownload.isDeleted()) {
      imageToDownload.delete();
    }
    
    logStatus("Image downloaded successfully!");
    console.log('Image downloaded successfully as:', link.download);
    
    // Visual feedback
    downloadImageBtn.textContent = 'Downloaded!';
    downloadImageBtn.style.backgroundColor = '#28a745';
    setTimeout(() => {
      downloadImageBtn.textContent = 'Download Cropped Card';
      downloadImageBtn.style.backgroundColor = '#4CAF50';
    }, 2000);
    
  } catch (error) {
    console.error('Download error:', error);
    logStatus("Download failed: " + error.message);
    
    // Error feedback
    downloadImageBtn.textContent = 'Download Failed';
    downloadImageBtn.style.backgroundColor = '#dc3545';
    setTimeout(() => {
      downloadImageBtn.textContent = 'Download Cropped Card';
      downloadImageBtn.style.backgroundColor = '#4CAF50';
    }, 2000);
  }
}

// Enable download button after capturing the frame
function enableDownloadButton() {
  console.log('Enabling download button');
  downloadImageBtn.disabled = false;
  downloadImageBtn.style.backgroundColor = '#4CAF50';
  downloadImageBtn.style.color = 'white';
  downloadImageBtn.textContent = 'Download Cropped Card';
  
  // Remove any existing event listeners and add new one
  downloadImageBtn.onclick = null;
  downloadImageBtn.addEventListener('click', downloadImage, { once: false });
  
  console.log('Download button enabled and event listener attached');
}

// Toggle camera on/off
toggleCameraBtn.addEventListener('click', () => {
  if (streaming) {
    stopCamera();
    // ล้างประวัติเมื่อหยุด
    frameHistory.forEach(f => {
      if (f.frame && !f.frame.isDeleted()) f.frame.delete();
      if (f.contour && !f.contour.isDeleted()) f.contour.delete();
    });
    frameHistory = [];
  } else {
    // Reset captured frame when restarting
    if (capturedFrame && !capturedFrame.isDeleted()) {
      capturedFrame.delete();
      capturedFrame = null;
    }
    if (bestContour && !bestContour.isDeleted()) {
      bestContour.delete();
      bestContour = null;
    }
    // Disable download button when restarting
    downloadImageBtn.disabled = true;
    downloadImageBtn.style.backgroundColor = '';
    downloadImageBtn.style.color = '';
    startCamera();
  }
});

// cleanup when page unloads
window.addEventListener('unload', ()=>{
  stopCamera();
  try {
    if (src && !src.isDeleted()) src.delete();
    if (gray && !gray.isDeleted()) gray.delete();
    if (blurred && !blurred.isDeleted()) blurred.delete();
    if (edges && !edges.isDeleted()) edges.delete();
    if (contours && !contours.isDeleted()) {
      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        if (cnt && !cnt.isDeleted()) cnt.delete();
      }
      contours.delete();
    }
    if (hierarchy && !hierarchy.isDeleted()) hierarchy.delete();
    if (bestContour && !bestContour.isDeleted()) bestContour.delete();
    if (cap && !cap.isDeleted()) cap.delete();
    if (croppedCardImage && !croppedCardImage.isDeleted()) croppedCardImage.delete();
    if (capturedFrame && !capturedFrame.isDeleted()) capturedFrame.delete();
  } catch(e){
    console.warn('Cleanup error:', e);
  }
});