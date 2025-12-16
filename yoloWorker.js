// yoloWorker.js
// Runs YOLO (onnxruntime-web) in a Web Worker to keep the main thread responsive.
// Loads ort from CDN via importScripts, preprocesses frames on OffscreenCanvas,
// performs inference, postprocesses, and returns detections.

/* global importScripts, OffscreenCanvas */

let ortLoaded = false;
let ortSession = null;
let yoloInputShape = [1, 3, 320, 320]; // NCHW (fixed for 320x320)
let yoloInputName = null;
let providers = [];
let yoloScoreThresh = 0.4;
let yoloNmsIouThresh = 0.45;
const modelPath = 'best320.onnx';

// Utility
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function xywh2xyxy(x, y, w, h) {
  const x1 = x - w / 2;
  const y1 = y - h / 2;
  const x2 = x + w / 2;
  const y2 = y + h / 2;
  return [x1, y1, x2, y2];
}
function bboxIoU(a, b) {
  const [ax1,ay1,ax2,ay2] = a; const [bx1,by1,bx2,by2] = b;
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const ia = iw * ih;
  const aa = Math.max(0, ax2-ax1) * Math.max(0, ay2-ay1);
  const ba = Math.max(0, bx2-bx1) * Math.max(0, by2-by1);
  const ua = aa + ba - ia;
  return ua <= 0 ? 0 : ia / ua;
}
function nms(boxes, scores, iouThresh, topK=100) {
  const idxs = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(v=>v[1]);
  const picked = []; let arr = idxs;
  while (arr.length) {
    const i = arr.shift();
    picked.push(i);
    if (picked.length >= topK) break;
    const rest = [];
    for (const j of arr) {
      if (bboxIoU(boxes[i], boxes[j]) < iouThresh) rest.push(j);
    }
    arr = rest;
  }
  return picked;
}

function letterboxToCanvas(srcBitmap, dstCanvas, dstW, dstH, fill=[114,114,114,255]) {
  const srcW = srcBitmap.width, srcH = srcBitmap.height;
  const r = Math.min(dstW / srcW, dstH / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const dw = Math.floor((dstW - newW) / 2);
  const dh = Math.floor((dstH - newH) / 2);
  const ctx = dstCanvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = `rgba(${fill[0]},${fill[1]},${fill[2]},${fill[3]/255})`;
  ctx.fillRect(0, 0, dstW, dstH);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'low';
  ctx.drawImage(srcBitmap, 0, 0, srcW, srcH, dw, dh, newW, newH);
  return { scale: r, padX: dw, padY: dh, newW, newH };
}

function preprocessToCHW(dstCanvas) {
  const w = dstCanvas.width, h = dstCanvas.height;
  const ctx = dstCanvas.getContext('2d', { willReadFrequently: true });
  const img = ctx.getImageData(0, 0, w, h);
  const data = img.data; // RGBA
  const chw = new Float32Array(3 * w * h);
  const wh = w * h;
  for (let i = 0; i < wh; i++) {
    const r = data[i*4] / 255;
    const g = data[i*4 + 1] / 255;
    const b = data[i*4 + 2] / 255;
    chw[i] = r;
    chw[i + wh] = g;
    chw[i + 2*wh] = b;
  }
  return chw;
}

function postprocessYolo(outTensor, info, origW, origH) {
  const data = outTensor.data;
  const dims = outTensor.dims;
  const boxes = [], scores = [], classes = [];
  const gain = info.scale, padX = info.padX, padY = info.padY;

  if (dims.length === 3) {
    const a = dims[1], b = dims[2];
    if (b >= 6 && b <= 256) {
      // [1, N=a, no=b]
      const N = a; const no = b; const numClasses = no - 5;
      let needSig = false;
      for (let k = 0; k < Math.min(10, N); k++) { if (data[k*no+4] > 1) { needSig = true; break; } }
      for (let i = 0; i < N; i++) {
        const off = i * no;
        let cx = data[off+0], cy = data[off+1], w = data[off+2], h = data[off+3];
        let obj = data[off+4]; if (needSig) obj = sigmoid(obj);
        let best = 0, bestIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          let v = data[off+5+c]; if (needSig) v = sigmoid(v);
          if (v > best) { best = v; bestIdx = c; }
        }
        const conf = obj * (numClasses > 0 ? best : 1);
        if (conf < yoloScoreThresh) continue;
        let [x1,y1,x2,y2] = xywh2xyxy(cx,cy,w,h);
        x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain; x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
        x1 = Math.max(0, Math.min(origW-1, x1)); y1 = Math.max(0, Math.min(origH-1, y1));
        x2 = Math.max(0, Math.min(origW-1, x2)); y2 = Math.max(0, Math.min(origH-1, y2));
        boxes.push([x1,y1,x2,y2]); scores.push(conf); classes.push(bestIdx);
      }
    } else if (a >= 6 && a <= 256) {
      // [1, no=a, N=b]
      const no = a; const N = b; const numClasses = no - 4;
      let needSig = false;
      for (let k = 0; k < Math.min(10, N); k++) { if (data[4*N + k] > 1) { needSig = true; break; } }
      for (let i = 0; i < N; i++) {
        const cx = data[0*N + i], cy = data[1*N + i], w = data[2*N + i], h = data[3*N + i];
        let best = 0, bestIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          let v = data[(4+c)*N + i]; if (needSig) v = sigmoid(v);
          if (v > best) { best = v; bestIdx = c; }
        }
        const conf = best; if (conf < yoloScoreThresh) continue;
        let [x1,y1,x2,y2] = xywh2xyxy(cx,cy,w,h);
        x1 = (x1 - padX) / gain; y1 = (y1 - padY) / gain; x2 = (x2 - padX) / gain; y2 = (y2 - padY) / gain;
        x1 = Math.max(0, Math.min(origW-1, x1)); y1 = Math.max(0, Math.min(origH-1, y1));
        x2 = Math.max(0, Math.min(origW-1, x2)); y2 = Math.max(0, Math.min(origH-1, y2));
        boxes.push([x1,y1,x2,y2]); scores.push(conf); classes.push(bestIdx);
      }
    } else {
      console.warn('Unexpected YOLO output dims (worker):', dims);
    }
  }
  const keep = nms(boxes, scores, yoloNmsIouThresh, 20);
  return keep.map(i => ({ box: boxes[i], score: scores[i], cls: classes[i] }));
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (!msg || !msg.type) return;

  try {
    if (msg.type === 'init') {
      // Load ORT from CDN in worker
      if (!ortLoaded) {
        importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.min.js');
        ortLoaded = true;
      }
      // Configure WASM env if available
      try {
        if (self.ort && self.ort.env && self.ort.env.wasm) {
          // Ensure assets load from CDN instead of page origin
          self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
          self.ort.env.wasm.simd = true;
          // Use 1 thread unless cross-origin isolation is enabled
          const coi = (typeof self.crossOriginIsolated !== 'undefined' && self.crossOriginIsolated);
          self.ort.env.wasm.numThreads = coi ? (msg.numThreads || 1) : 1;
        }
      } catch (e) {}

      yoloScoreThresh = msg.scoreThresh ?? yoloScoreThresh;
      yoloNmsIouThresh = msg.nmsIouThresh ?? yoloNmsIouThresh;

      // Providers preference: webgpu -> webgl -> wasm
      providers = [];
      if (msg.allowWebGPU && 'gpu' in self.navigator) providers.push('webgpu');
      // In worker, create a scratch canvas to probe webgl
      let glOk = false;
      try {
        const probe = new OffscreenCanvas(1,1);
        const gl = probe.getContext('webgl2') || probe.getContext('webgl');
        glOk = !!gl;
      } catch (e) { glOk = false; }
      if (glOk) providers.push('webgl');
      providers.push('wasm');

      const so = { executionProviders: providers, graphOptimizationLevel: 'all' };

      ortSession = await self.ort.InferenceSession.create(modelPath, so);
      const firstInput = ortSession.inputNames[0];
      yoloInputName = firstInput;

      self.postMessage({ type: 'ready', providers, inputShape: yoloInputShape });
      return;
    }

    if (msg.type === 'detect') {
      if (!ortSession) { self.postMessage({ type: 'error', error: 'Session not ready' }); return; }
    const bitmap = msg.bitmap; // ImageBitmap
    const origW = msg.origW, origH = msg.origH;
    // Use [N,C,H,W] to derive canvas width/height
  const inH = yoloInputShape[2], inW = yoloInputShape[3];
  const canvas = new OffscreenCanvas(inW, inH);
  const info = letterboxToCanvas(bitmap, canvas, inW, inH);
      const chw = preprocessToCHW(canvas);
      const input = new self.ort.Tensor('float32', chw, [1, 3, inH, inW]);
      const feeds = {}; feeds[yoloInputName] = input;
      const t0 = performance.now();
      const output = await ortSession.run(feeds);
      const t1 = performance.now();
      const first = output[Object.keys(output)[0]];
      const dets = postprocessYolo(first, info, origW, origH);
      self.postMessage({ type: 'result', dets, timeMs: t1 - t0 });
      try { bitmap.close && bitmap.close(); } catch (e) {}
      return;
    }
  } catch (e) {
    self.postMessage({ type: 'error', error: String(e && e.message ? e.message : e) });
  }
};
