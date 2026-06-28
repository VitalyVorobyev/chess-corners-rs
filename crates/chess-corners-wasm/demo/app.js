import init, {
  ChessDetector,
  ChessConfig,
  ChessRefiner,
  ChessRing,
  CenterOfMassConfig,
  DetectionStrategy,
  DetectorConfig,
  ForstnerConfig,
  MultiscaleConfig,
  RadonConfig,
  SaddlePointConfig,
  UpscaleConfig,
} from '../pkg/chess_corners_wasm.js';

// Layout of each corner in the stride-7 Float32Array returned by `detect()`.
const STRIDE = 7;
const IDX_X = 0,
      IDX_Y = 1,
      IDX_RESP = 2,
      IDX_A0_ANGLE = 3,
      IDX_A0_SIGMA = 4,
      IDX_A1_ANGLE = 5,
      IDX_A1_SIGMA = 6;

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
const thresholdLbl = $('thresholdLbl');
const refinerRow = $('refinerRow');
const nmsRadius = $('nmsRadius');
const nmsRadiusVal = $('nmsRadiusVal');
const minCluster = $('minCluster');
const minClusterVal = $('minClusterVal');
const detectorMode = $('detectorMode');
const pyrLevels = $('pyrLevels');
const pyrLevelsVal = $('pyrLevelsVal');
const pyrMinSize = $('pyrMinSize');
const pyrMinSizeVal = $('pyrMinSizeVal');
const upscaleFactor = $('upscaleFactor');
const refiner = $('refiner');
const autoRun = $('autoRun');
const showHeatmap = $('showHeatmap');
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
let lastCorners = null;        // Float32Array with stride-7 layout
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

// Threshold display is mode-dependent; updated by updateThresholdUI().
threshold.addEventListener('input', () => {
  const isRadon = detectorMode.value === 'radon';
  thresholdVal.textContent = isRadon
    ? Number(threshold.value).toFixed(2)
    : Number(threshold.value).toFixed(0);
});
linkRange(nmsRadius, nmsRadiusVal);
linkRange(minCluster, minClusterVal);
linkRange(pyrLevels, pyrLevelsVal);
linkRange(pyrMinSize, pyrMinSizeVal);
linkRange(arrowLen, arrowLenVal);
linkRange(dotRadius, dotRadiusVal);

// ChESS threshold: absolute response floor (0–100, step 1, default 30).
// Radon threshold: fraction of per-frame max (0–0.5, step 0.01, default 0.01).
// Refiner is ChESS-only; hidden when Radon is active.
function updateThresholdUI() {
  const isRadon = detectorMode.value === 'radon';
  if (isRadon) {
    thresholdLbl.textContent = 'Relative threshold';
    threshold.min = '0';
    threshold.max = '0.5';
    threshold.step = '0.01';
    threshold.value = '0.01';
    thresholdVal.textContent = '0.01';
    refinerRow.style.display = 'none';
  } else {
    thresholdLbl.textContent = 'Threshold';
    threshold.min = '0';
    threshold.max = '100';
    threshold.step = '1';
    threshold.value = '30';
    thresholdVal.textContent = '30';
    refinerRow.style.display = '';
  }
}

// Call once on load to configure for the default mode (canonical / ChESS).
updateThresholdUI();

function buildRefiner(name) {
  switch (name) {
    case "forstner":
      return ChessRefiner.withForstner(new ForstnerConfig());
    case "saddle_point":
      return ChessRefiner.withSaddlePoint(new SaddlePointConfig());
    case "center_of_mass":
    default:
      return ChessRefiner.withCenterOfMass(new CenterOfMassConfig());
  }
}

function buildConfig() {
  // Map the dropdown to the typed config tree.
  const modeValue = detectorMode.value;
  const isRadon = modeValue === "radon";
  const cfg = new DetectorConfig();

  cfg.threshold = parseFloat(threshold.value);

  const upFactor = parseInt(upscaleFactor.value, 10);
  cfg.upscale =
    upFactor <= 1 ? UpscaleConfig.disabled() : UpscaleConfig.fixed(upFactor);

  if (isRadon) {
    const radon = new RadonConfig();
    radon.nmsRadius = parseInt(nmsRadius.value, 10);
    radon.minClusterSize = parseInt(minCluster.value, 10);
    cfg.strategy = DetectionStrategy.fromRadon(radon);
    // Radon detector is single-scale in the demo.
    cfg.multiscale = MultiscaleConfig.singleScale();
  } else {
    const chess = new ChessConfig();
    chess.ring = modeValue === "broad" ? ChessRing.Broad : ChessRing.Canonical;
    chess.nmsRadius = parseInt(nmsRadius.value, 10);
    chess.minClusterSize = parseInt(minCluster.value, 10);
    chess.refiner = buildRefiner(refiner.value);
    cfg.strategy = DetectionStrategy.fromChess(chess);

    const levels = parseInt(pyrLevels.value, 10);
    cfg.multiscale =
      levels <= 1
        ? MultiscaleConfig.singleScale()
        : MultiscaleConfig.pyramid(levels, parseInt(pyrMinSize.value, 10), 3);
  }

  return cfg;
}

function configureDetector(det) {
  det.applyConfig(buildConfig());
}

// Render a working-resolution Radon heatmap into the main canvas at
// input-image dimensions. The detector exposes the heatmap at
// (width * upscale * radon_image_upsample) — we downsample by
// `diagnostics_radon_heatmap_scale()` so the overlay aligns with corner overlays in
// input pixel space.
function drawHeatmap(heatmap, hmW, hmH, scale) {
  // Find max for normalization.
  let maxV = 0;
  for (let i = 0; i < heatmap.length; i++) {
    if (heatmap[i] > maxV) maxV = heatmap[i];
  }
  if (maxV <= 0) maxV = 1;

  // Render the heatmap directly at its working resolution onto an
  // offscreen canvas, then drawImage with downscaling so the result
  // matches input dimensions.
  const off = document.createElement('canvas');
  off.width = hmW;
  off.height = hmH;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(hmW, hmH);
  const inv = 255 / maxV;
  for (let i = 0, j = 0; i < heatmap.length; i++, j += 4) {
    const v = Math.min(255, Math.max(0, heatmap[i] * inv));
    img.data[j] = v;
    img.data[j + 1] = v;
    img.data[j + 2] = v;
    img.data[j + 3] = 255;
  }
  offCtx.putImageData(img, 0, 0);

  // The heatmap is at input * scale; draw it scaled to currentW × currentH.
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, hmW, hmH, 0, 0, currentW, currentH);
  ctx.imageSmoothingEnabled = true;
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
    ctx.arc(x + 0.5, y + 0.5, r, 0, Math.PI * 2);
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
  // Shift to pixel-center convention so arrows originate from the same point
  // as the dot overlay.
  x += 0.5;
  y += 0.5;
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

  // Background: original image, unless heatmap overlay is requested.
  let heatmapMs = 0;
  if (showHeatmap.checked) {
    const h0 = performance.now();
    const heatmap = detector.diagnostics_radon_heatmap_rgba(currentRGBA, currentW, currentH);
    const hmW = detector.diagnostics_radon_heatmap_width();
    const hmH = detector.diagnostics_radon_heatmap_height();
    const scale = detector.diagnostics_radon_heatmap_scale();
    heatmapMs = performance.now() - h0;
    // Black backdrop in case the heatmap dims fail to fully cover the canvas.
    canvas.width = currentW;
    canvas.height = currentH;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, currentW, currentH);
    drawHeatmap(heatmap, hmW, hmH, scale);
  } else {
    ctx.putImageData(new ImageData(currentRGBA, currentW, currentH), 0, 0);
  }
  drawCorners(result);

  const baseStats =
    `corners: ${n}\n` +
    `detect:  ${(t1 - t0).toFixed(1)} ms\n` +
    `size:    ${currentW}×${currentH}\n` +
    `upscale: ${upscaleFactor.value === '0' ? 'off' : upscaleFactor.value + '×'}`;
  statsEl.textContent = showHeatmap.checked
    ? `${baseStats}\nheatmap: ${heatmapMs.toFixed(1)} ms`
    : baseStats;
  setStatus(cameraStream ? 'webcam live' : 'ok');
}

function scheduleRun() {
  if (cameraStream) return;
  if (!autoRun.checked) return;
  run();
}

function setupAutoRerun() {
  const triggers = [
    threshold, nmsRadius, minCluster, detectorMode,
    pyrLevels, pyrMinSize, upscaleFactor, refiner,
    showHeatmap, showAxes, arrowLen, dotRadius,
  ];
  for (const el of triggers) {
    el.addEventListener('change', scheduleRun);
    if (el.type === 'range') el.addEventListener('input', scheduleRun);
  }
  // Update threshold label/range whenever the detector mode changes.
  detectorMode.addEventListener('change', updateThresholdUI);
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
  const a0 = lastCorners[i + IDX_A0_ANGLE];
  const s0 = lastCorners[i + IDX_A0_SIGMA];
  const a1 = lastCorners[i + IDX_A1_ANGLE];
  const s1 = lastCorners[i + IDX_A1_SIGMA];
  hoverInfo.textContent =
    `(${x.toFixed(2)}, ${y.toFixed(2)})\n` +
    `response: ${resp.toFixed(2)}\n` +
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
