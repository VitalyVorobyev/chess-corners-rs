# chess-corners-wasm

WebAssembly bindings for the [ChESS corner detector](https://github.com/VitalyVorobyev/chess-corners-rs). Detect chessboard corners with subpixel accuracy directly in the browser.

## Building

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

The npm-ready package is generated in `crates/chess-corners-wasm/pkg/`.

To target a bundler (Webpack, Vite, etc.) instead:

```bash
wasm-pack build crates/chess-corners-wasm --target bundler
```

## Installation

After building, install the package from the local `pkg/` directory:

```bash
npm install ./crates/chess-corners-wasm/pkg
```

Or copy `pkg/` into your project and reference it directly.

## Usage

### Initialization

```js
import init, { ChessDetector } from 'chess-corners-wasm';

// Initialize the WASM module (required once before any API calls).
await init();
```

### Detect corners from an image file

```js
const detector = new ChessDetector();

// Load an image onto a canvas to get pixel data.
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
const img = new Image();
img.src = 'board.png';
await img.decode();
canvas.width = img.width;
canvas.height = img.height;
ctx.drawImage(img, 0, 0);

const imageData = ctx.getImageData(0, 0, img.width, img.height);

// detect_rgba accepts RGBA pixels from canvas and converts to grayscale internally.
const corners = detector.detect_rgba(imageData.data, img.width, img.height);

// corners is a Float32Array with stride 4: [x, y, response, orientation, ...]
for (let i = 0; i < corners.length; i += 4) {
    const x = corners[i];
    const y = corners[i + 1];
    const response = corners[i + 2];
    const orientation = corners[i + 3]; // radians, in [0, PI)
    console.log(`Corner at (${x.toFixed(2)}, ${y.toFixed(2)}), strength=${response.toFixed(1)}`);
}
```

### Webcam streaming

```js
const detector = new ChessDetector();
detector.set_pyramid_levels(3); // enable multiscale for better detection

const video = document.querySelector('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

function processFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Reuses internal buffers across frames automatically.
    const corners = detector.detect_rgba(imageData.data, canvas.width, canvas.height);
    drawCorners(corners); // your rendering logic

    requestAnimationFrame(processFrame);
}

processFrame();
```

### Response map visualization

```js
const detector = new ChessDetector();

// Get the raw ChESS response as a Float32Array (row-major, width x height).
const response = detector.response_rgba(imageData.data, width, height);
const rWidth = detector.response_width();
const rHeight = detector.response_height();

// Render as a heatmap on a canvas.
const out = ctx.createImageData(rWidth, rHeight);
const maxVal = Math.max(...response);
for (let i = 0; i < response.length; i++) {
    const v = Math.floor(255 * response[i] / maxVal);
    out.data[4 * i] = v;       // R
    out.data[4 * i + 1] = 0;   // G
    out.data[4 * i + 2] = 255 - v; // B
    out.data[4 * i + 3] = 255; // A
}
ctx.putImageData(out, 0, 0);
```

### Configuration

```js
const detector = new ChessDetector();

// Or start with the multiscale preset:
// const detector = ChessDetector.multiscale();

// Threshold (fraction of max response, default 0.2).
detector.set_threshold(0.15);

// Non-maximum suppression radius (default 2).
detector.set_nms_radius(3);

// Broad mode uses the wider, more blur-tolerant detector sampling pattern.
detector.set_broad_mode(false);

// Minimum cluster size to accept a corner (default 2).
detector.set_min_cluster_size(2);

// Pyramid levels: 1 = single-scale, 3 = recommended multiscale.
detector.set_pyramid_levels(3);

// Minimum pyramid level size in pixels (default 128).
detector.set_pyramid_min_size(128);

// Subpixel refiner: "center_of_mass" (default), "forstner", or "saddle_point".
detector.set_refiner("forstner");
```

## API Reference

### `ChessDetector`

| Method | Description |
|--------|-------------|
| `new ChessDetector()` | Create detector with default single-scale config |
| `ChessDetector.multiscale()` | Create detector with 3-level pyramid preset |
| `detect(pixels, w, h)` | Detect corners from grayscale `Uint8Array` |
| `detect_rgba(pixels, w, h)` | Detect corners from RGBA `Uint8Array` |
| `response(pixels, w, h)` | Compute response map from grayscale pixels |
| `response_rgba(pixels, w, h)` | Compute response map from RGBA pixels |
| `response_width()` | Width of the last computed response map |
| `response_height()` | Height of the last computed response map |
| `set_threshold(rel)` | Set relative threshold (0.0-1.0) |
| `set_nms_radius(r)` | Set NMS radius |
| `set_broad_mode(v)` | Toggle broad detector mode |
| `set_min_cluster_size(v)` | Set min cluster size |
| `set_pyramid_levels(n)` | Set pyramid depth |
| `set_pyramid_min_size(v)` | Set min pyramid level size |
| `set_refiner(name)` | Set subpixel refiner |

### Output format

**Corners** (`detect` / `detect_rgba`): `Float32Array` with stride 4:

| Offset | Field | Description |
|--------|-------|-------------|
| `i + 0` | `x` | Subpixel x coordinate |
| `i + 1` | `y` | Subpixel y coordinate |
| `i + 2` | `response` | ChESS response strength |
| `i + 3` | `orientation` | Grid axis angle in radians [0, PI) |

**Response map** (`response` / `response_rgba`): `Float32Array` in row-major order, dimensions available via `response_width()` / `response_height()`.

## Binary size

~51 KB raw, ~23 KB gzipped (single-scale, no parallelism, no SIMD).

## License

MIT
