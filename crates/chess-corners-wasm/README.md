# @vitavision/chess-corners

WebAssembly bindings for the [ChESS corner detector](https://github.com/VitalyVorobyev/chess-corners-rs). Detect chessboard corners with subpixel accuracy directly in the browser.

> Previously published as **`chess-corners-wasm`** on npm (≤ 0.6.x). The
> package was renamed to `@vitavision/chess-corners` in 0.7.0; the
> legacy name is deprecated. Migrate by replacing your dependency name —
> the API is unchanged.

## Installation

```bash
npm install @vitavision/chess-corners
```

## Building from source

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

The npm-ready package is generated in `crates/chess-corners-wasm/pkg/`
under the `@vitavision/chess-corners` name (the published name is set
by the release workflow; locally `wasm-pack` derives it from the Rust
crate name `chess-corners-wasm`).

To target a bundler (Webpack, Vite, etc.) instead:

```bash
wasm-pack build crates/chess-corners-wasm --target bundler
```

## Usage

### Initialization

```js
import init, { ChessDetector } from '@vitavision/chess-corners';

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

// corners is a Float32Array with stride 9 per corner:
//   [x, y, response, contrast, fit_rms,
//    axis0_angle, axis0_sigma, axis1_angle, axis1_sigma, ...]
for (let i = 0; i < corners.length; i += 9) {
    const x = corners[i];
    const y = corners[i + 1];
    const response = corners[i + 2];
    const contrast = corners[i + 3];
    const axis0_angle = corners[i + 5]; // radians, in [0, PI)
    const axis1_angle = corners[i + 7]; // radians, in (axis0, axis0 + PI)
    console.log(`Corner at (${x.toFixed(2)}, ${y.toFixed(2)}), strength=${response.toFixed(1)}`);
}
```

### Webcam streaming

```js
import init, {
  ChessDetector,
  DetectorConfig,
  MultiscaleConfig,
} from '@vitavision/chess-corners';

await init();

// Multiscale ChESS preset for live webcam feeds.
const cfg = DetectorConfig.chessMultiscale();
const detector = ChessDetector.withConfig(cfg);

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

### Response map visualization (diagnostics)

The `diagnostics_*` methods expose intermediate detector data — raw
response maps and Radon heatmaps — for debugging and visualization.
They are opt-in and not part of the normal detection result, which is
the `Float32Array` returned by `detect` / `detect_rgba`.

```js
const detector = new ChessDetector();

// Get the raw ChESS response as a Float32Array (row-major, width x height).
const response = detector.diagnostics_response_rgba(imageData.data, width, height);
const rWidth = detector.diagnostics_response_width();
const rHeight = detector.diagnostics_response_height();

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

## Typed configuration

Every detector knob is reachable through a typed `DetectorConfig` tree.
Construct one with a preset and tweak only the fields you need:

```ts
import init, {
  ChessDetector,
  DetectorConfig,
  ChessConfig,
  ChessRefiner,
  ChessRing,
  DescriptorRing,
  DetectionStrategy,
  ForstnerConfig,
  MultiscaleConfig,
  OrientationMethod,
  PeakFitMode,
  RadonConfig,
  RadonRefiner,
  Threshold,
  UpscaleConfig,
} from '@vitavision/chess-corners';

await init();

const cfg = DetectorConfig.chessMultiscale();

// Top-level fields are simple getters / setters:
cfg.threshold = Threshold.relative(0.15);
cfg.multiscale = MultiscaleConfig.pyramid(4, 64, 3); // levels, minSize, refinementRadius
cfg.upscale = UpscaleConfig.fixed(2);
cfg.orientationMethod = OrientationMethod.DiskFit;
cfg.mergeRadius = 2.5;

// Strategy selects ChESS vs Radon and carries the detector tuning:
const chess = new ChessConfig();
chess.ring = ChessRing.Broad;
chess.descriptorRing = DescriptorRing.Canonical;
chess.nmsRadius = 3;
chess.refiner = ChessRefiner.withForstner(new ForstnerConfig());
cfg.strategy = DetectionStrategy.fromChess(chess);

const detector = ChessDetector.withConfig(cfg);
```

### Nested edits propagate

Getters return wrappers that share storage with the parent, so chained
mutation works without a round-trip:

```ts
cfg.strategy.chess.ring = ChessRing.Broad;
cfg.strategy.chess.refiner.forstner.maxOffset = 2.0;
cfg.strategy.chess.nmsRadius = 3;
cfg.multiscale = MultiscaleConfig.pyramid(4, 64, 3);
```

`getConfig()` returns an independent snapshot whose cells are detached
from the live detector. Use `applyConfig()` to commit edits made on the
snapshot:

```ts
const snapshot = detector.getConfig();
snapshot.strategy.chess.nmsRadius = 4;
detector.applyConfig(snapshot);
```

## API Reference

### `ChessDetector`

| Method | Description |
|--------|-------------|
| `new ChessDetector()` | Create detector with default single-scale config |
| `ChessDetector.multiscale()` | Create detector with 3-level pyramid preset |
| `ChessDetector.withConfig(cfg)` | Create detector seeded from a typed `DetectorConfig` |
| `detector.getConfig()` | Snapshot the live configuration as a `DetectorConfig` |
| `detector.applyConfig(cfg)` | Replace the configuration with the given `DetectorConfig` |
| `detect(pixels, w, h)` | Detect corners from grayscale `Uint8Array` |
| `detect_rgba(pixels, w, h)` | Detect corners from RGBA `Uint8Array` |

#### Diagnostics

Opt-in methods that expose intermediate detector data for debugging and
visualization. They are not part of the normal detection result.

| Method | Description |
|--------|-------------|
| `diagnostics_response(pixels, w, h)` | Compute response map from grayscale pixels |
| `diagnostics_response_rgba(pixels, w, h)` | Compute response map from RGBA pixels |
| `diagnostics_response_width()` | Width of the last computed response map |
| `diagnostics_response_height()` | Height of the last computed response map |
| `diagnostics_radon_heatmap(pixels, w, h)` | Compute the Radon heatmap from grayscale pixels |
| `diagnostics_radon_heatmap_rgba(pixels, w, h)` | Compute the Radon heatmap from RGBA pixels |
| `diagnostics_radon_heatmap_width()` | Width of the last computed Radon heatmap (working resolution) |
| `diagnostics_radon_heatmap_height()` | Height of the last computed Radon heatmap |
| `diagnostics_radon_heatmap_scale()` | Working-to-input scale factor for the last heatmap |

### Output format

**Corners** (`detect` / `detect_rgba`): `Float32Array` with stride 9 per corner:

| Offset | Field | Description |
|--------|-------|-------------|
| `i + 0` | `x` | Subpixel x coordinate |
| `i + 1` | `y` | Subpixel y coordinate |
| `i + 2` | `response` | ChESS response strength |
| `i + 3` | `contrast` | Fitted bright/dark amplitude |
| `i + 4` | `fit_rms` | RMS residual of the two-axis fit |
| `i + 5` | `axis0_angle` | First grid axis, radians in `[0, π)` |
| `i + 6` | `axis0_sigma` | 1σ uncertainty of `axis0_angle` |
| `i + 7` | `axis1_angle` | Second grid axis, radians in `(axis0, axis0 + π)` |
| `i + 8` | `axis1_sigma` | 1σ uncertainty of `axis1_angle` |

Rotating CCW from `axis0_angle` toward `axis1_angle` traverses a **dark** sector of the corner. The two grid axes are not assumed orthogonal, so the layout can represent projective warp instead of forcing a right-angle model.

**Response map** (`diagnostics_response` / `diagnostics_response_rgba`): `Float32Array` in row-major order, dimensions available via `diagnostics_response_width()` / `diagnostics_response_height()`.

## Binary size

~196 KB raw, ~70 KB gzipped (single-scale, no parallelism, no SIMD, measured on 0.10.0).

## License

MIT
