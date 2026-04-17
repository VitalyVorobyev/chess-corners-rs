import init, { ChessDetector } from '../pkg/chess_corners_wasm.js';

// Layout of each corner in the stride-9 Float32Array returned by `detect()`.
const STRIDE = 9;
const IDX_X = 0,
      IDX_Y = 1,
      IDX_RESP = 2,
      IDX_CONTRAST = 3,
      IDX_RMS = 4,
      IDX_A0_ANGLE = 5,
      IDX_A0_SIGMA = 6,
      IDX_A1_ANGLE = 7,
      IDX_A1_SIGMA = 8;

const AXIS0_COLOR = '#17a2b8';
const AXIS1_COLOR = '#ff8c42';
const DOT_COLOR = '#ffffffcc';

const $ = (id) => document.getElementById(id);

const runBtn = $('runBtn');
const camBtn = $('camBtn');
const statusEl = $('status');
const statsEl = $('stats');
const fileInput = $('fileInput');
const sampleSelect = $('sampleSelect');
const dropHint = $('dropHint');
const canvas = $('cv');
const hoverInfo = $('hoverInfo');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// Config controls.
const threshold = $('threshold');
const thresholdVal = $('thresholdVal');
const nmsRadius = $('nmsRadius');
const nmsRadiusVal = $('nmsRadiusVal');
const minCluster = $('minCluster');
const minClusterVal = $('minClusterVal');
const broadMode = $('broadMode');
const pyrLevels = $('pyrLevels');
const pyrLevelsVal = $('pyrLevelsVal');
const pyrMinSize = $('pyrMinSize');
const pyrMinSizeVal = $('pyrMinSizeVal');
const upscaleFactor = $('upscaleFactor');
const refiner = $('refiner');
const autoRun = $('autoRun');
const showAxes = $('showAxes');
const arrowLen = $('arrowLen');
const arrowLenVal = $('arrowLenVal');
const dotRadius = $('dotRadius');
const dotRadiusVal = $('dotRadiusVal');

let detector = null;
let currentImageBitmap = null; // ImageBitmap or HTMLImageElement
let currentRGBA = null;        // Uint8ClampedArray
let currentW = 0;
let currentH = 0;
let lastCorners = null;        // Float32Array with stride-9 layout
let cameraStream = null;
let cameraVideo = null;
let cameraRAF = 0;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function linkRange(input, label, fmt = (v) => v) {
  input.addEventListener('input', () => {
    label.textContent = fmt(input.value);
  });
  label.textContent = fmt(input.value);
}

linkRange(threshold, thresholdVal, (v) => Number(v).toFixed(2));
linkRange(nmsRadius, nmsRadiusVal);
linkRange(minCluster, minClusterVal);
linkRange(pyrLevels, pyrLevelsVal);
linkRange(pyrMinSize, pyrMinSizeVal);
linkRange(arrowLen, arrowLenVal);
linkRange(dotRadius, dotRadiusVal);

function configureDetector(det) {
  det.set_threshold(parseFloat(threshold.value));
  det.set_nms_radius(parseInt(nmsRadius.value, 10));
  det.set_min_cluster_size(parseInt(minCluster.value, 10));
  det.set_broad_mode(broadMode.checked);
  det.set_pyramid_levels(parseInt(pyrLevels.value, 10));
  det.set_pyramid_min_size(parseInt(pyrMinSize.value, 10));
  det.set_upscale_factor(parseInt(upscaleFactor.value, 10));
  det.set_refiner(refiner.value);
}

function drawImage(bitmap) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  currentRGBA = data.data;
  currentW = canvas.width;
  currentH = canvas.height;
}

function drawCorners(corners) {
  if (!corners) return;
  const r = parseInt(dotRadius.value, 10);
  const showArr = showAxes.checked;
  const aLen = parseInt(arrowLen.value, 10);

  // Scale stroke widths proportional to image size so overlays stay visible
  // on tiny/huge images alike.
  const lineW = Math.max(1, Math.round(Math.min(currentW, currentH) / 700));
  ctx.lineWidth = lineW;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Corners first.
  ctx.fillStyle = DOT_COLOR;
  for (let i = 0; i < corners.length; i += STRIDE) {
    const x = corners[i + IDX_X];
    const y = corners[i + IDX_Y];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  if (!showArr) return;

  // Two arrows per corner.
  for (let i = 0; i < corners.length; i += STRIDE) {
    const x = corners[i + IDX_X];
    const y = corners[i + IDX_Y];
    drawArrow(x, y, corners[i + IDX_A0_ANGLE], aLen, AXIS0_COLOR);
    drawArrow(x, y, corners[i + IDX_A1_ANGLE], aLen, AXIS1_COLOR);
  }
}

function drawArrow(x, y, angle, len, color) {
  const dx = Math.cos(angle) * len;
  const dy = Math.sin(angle) * len;
  const tx = x + dx;
  const ty = y + dy;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(tx, ty);
  ctx.stroke();

  // Arrowhead.
  const head = Math.max(3, len * 0.25);
  const a = Math.atan2(dy, dx);
  const ax1 = tx - Math.cos(a - 0.45) * head;
  const ay1 = ty - Math.sin(a - 0.45) * head;
  const ax2 = tx - Math.cos(a + 0.45) * head;
  const ay2 = ty - Math.sin(a + 0.45) * head;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(ax1, ay1);
  ctx.lineTo(ax2, ay2);
  ctx.closePath();
  ctx.fill();
}

function run() {
  if (!detector || !currentRGBA) return;
  try {
    configureDetector(detector);
  } catch (err) {
    setStatus(`config error: ${err}`);
    return;
  }
  const t0 = performance.now();
  const result = detector.detect_rgba(currentRGBA, currentW, currentH);
  const t1 = performance.now();
  lastCorners = result;
  const n = result.length / STRIDE;

  // Redraw image + overlay.
  ctx.putImageData(new ImageData(currentRGBA, currentW, currentH), 0, 0);
  drawCorners(result);

  statsEl.textContent =
    `corners: ${n}\n` +
    `detect:  ${(t1 - t0).toFixed(1)} ms\n` +
    `size:    ${currentW}×${currentH}\n` +
    `upscale: ${upscaleFactor.value === '0' ? 'off' : upscaleFactor.value + '×'}`;
  setStatus(cameraStream ? 'webcam live' : 'ok');
}

function scheduleRun() {
  if (cameraStream) return;
  if (!autoRun.checked) return;
  run();
}

function setupAutoRerun() {
  const triggers = [
    threshold, nmsRadius, minCluster, broadMode,
    pyrLevels, pyrMinSize, upscaleFactor, refiner,
    showAxes, arrowLen, dotRadius,
  ];
  for (const el of triggers) {
    el.addEventListener('change', scheduleRun);
    if (el.type === 'range') el.addEventListener('input', scheduleRun);
  }
}

// ---- Image loading ----

async function loadSample(name) {
  setStatus(`loading ${name}.png…`);
  const resp = await fetch(`./testimages/${name}.png`);
  if (!resp.ok) throw new Error(`failed to fetch testimages/${name}.png`);
  const blob = await resp.blob();
  const bmp = await createImageBitmap(blob);
  currentImageBitmap = bmp;
  drawImage(bmp);
  setStatus('ready');
  scheduleRun();
}

async function loadFile(file) {
  setStatus(`loading ${file.name}…`);
  const bmp = await createImageBitmap(file);
  currentImageBitmap = bmp;
  drawImage(bmp);
  setStatus('ready');
  scheduleRun();
}

// ---- Webcam ----

async function toggleWebcam() {
  if (cameraStream) {
    stopWebcam();
    return;
  }
  try {
    setStatus('starting webcam…');
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    cameraVideo = document.createElement('video');
    cameraVideo.srcObject = cameraStream;
    cameraVideo.playsInline = true;
    await cameraVideo.play();
    camBtn.textContent = 'Stop webcam';
    runBtn.disabled = true;
    stepWebcam();
  } catch (err) {
    setStatus(`webcam error: ${err.message}`);
    stopWebcam();
  }
}

function stopWebcam() {
  if (cameraRAF) cancelAnimationFrame(cameraRAF);
  cameraRAF = 0;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  if (cameraVideo) {
    cameraVideo.pause();
    cameraVideo = null;
  }
  camBtn.textContent = 'Webcam';
  runBtn.disabled = !detector || !currentRGBA;
  setStatus(detector ? 'ready' : 'loading…');
}

function stepWebcam() {
  if (!cameraStream || !cameraVideo) return;
  const vw = cameraVideo.videoWidth;
  const vh = cameraVideo.videoHeight;
  if (vw === 0 || vh === 0) {
    cameraRAF = requestAnimationFrame(stepWebcam);
    return;
  }
  canvas.width = vw;
  canvas.height = vh;
  ctx.drawImage(cameraVideo, 0, 0);
  const data = ctx.getImageData(0, 0, vw, vh);
  currentRGBA = data.data;
  currentW = vw;
  currentH = vh;
  run();
  cameraRAF = requestAnimationFrame(stepWebcam);
}

// ---- Hover readout ----

function nearestCorner(px, py) {
  if (!lastCorners || lastCorners.length === 0) return null;
  let best = -1;
  let bestD = Infinity;
  const threshold2 = 15 * 15; // within 15 px (image space)
  for (let i = 0; i < lastCorners.length; i += STRIDE) {
    const dx = lastCorners[i + IDX_X] - px;
    const dy = lastCorners[i + IDX_Y] - py;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD && d2 < threshold2) {
      bestD = d2;
      best = i;
    }
  }
  return best >= 0 ? best : null;
}

function onCanvasMove(e) {
  if (!lastCorners) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = currentW / rect.width;
  const scaleY = currentH / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const i = nearestCorner(px, py);
  if (i === null) {
    hoverInfo.style.display = 'none';
    return;
  }
  const x = lastCorners[i + IDX_X];
  const y = lastCorners[i + IDX_Y];
  const resp = lastCorners[i + IDX_RESP];
  const contrast = lastCorners[i + IDX_CONTRAST];
  const rms = lastCorners[i + IDX_RMS];
  const a0 = lastCorners[i + IDX_A0_ANGLE];
  const s0 = lastCorners[i + IDX_A0_SIGMA];
  const a1 = lastCorners[i + IDX_A1_ANGLE];
  const s1 = lastCorners[i + IDX_A1_SIGMA];
  hoverInfo.textContent =
    `(${x.toFixed(2)}, ${y.toFixed(2)})\n` +
    `response: ${resp.toFixed(2)}\n` +
    `contrast: ${contrast.toFixed(1)}\n` +
    `fit rms:  ${rms.toFixed(2)}\n` +
    `axis 0:   ${a0.toFixed(3)} ± ${s0.toFixed(3)} rad\n` +
    `axis 1:   ${a1.toFixed(3)} ± ${s1.toFixed(3)} rad`;
  hoverInfo.style.display = 'block';
  hoverInfo.style.left = `${e.clientX - rect.left + 12}px`;
  hoverInfo.style.top = `${e.clientY - rect.top + 12}px`;
}

canvas.addEventListener('mousemove', onCanvasMove);
canvas.addEventListener('mouseleave', () => {
  hoverInfo.style.display = 'none';
});

// ---- Drag & drop ----

['dragenter', 'dragover'].forEach((ev) =>
  dropHint.addEventListener(ev, (e) => {
    e.preventDefault();
    dropHint.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach((ev) =>
  dropHint.addEventListener(ev, (e) => {
    e.preventDefault();
    dropHint.classList.remove('dragover');
  })
);
dropHint.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) loadFile(file);
});
document.body.addEventListener('dragover', (e) => e.preventDefault());

fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) loadFile(file);
});
sampleSelect.addEventListener('change', () => {
  const v = sampleSelect.value;
  if (v) loadSample(v);
});

runBtn.addEventListener('click', run);
camBtn.addEventListener('click', toggleWebcam);

// ---- Bootstrap ----

async function main() {
  setStatus('initialising WASM…');
  await init();
  detector = new ChessDetector();
  setupAutoRerun();
  try {
    await loadSample('small');
  } catch (err) {
    setStatus('ready (load an image to start)');
  }
  runBtn.disabled = false;
  camBtn.disabled = false;
}

main().catch((err) => {
  console.error(err);
  setStatus(`init failed: ${err.message || err}`);
});
