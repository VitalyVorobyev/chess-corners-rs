# @vitavision/chess-corners

WebAssembly bindings for the [ChESS corner detector](https://github.com/VitalyVorobyev/chess-corners-rs). Detect chessboard corners with subpixel accuracy directly in the browser.

> Previously published as **`chess-corners-wasm`** on npm (≤ 0.6.x). The
> package was renamed to `@vitavision/chess-corners` in 0.7.0; the
> legacy name is deprecated. Migrate by replacing your dependency name.
> Snake_case method names from earlier releases remain available as
> compatibility aliases; new code should prefer camelCase.

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

// detectRgba accepts RGBA pixels from canvas and converts to grayscale internally.
const corners = detector.detectRgba(imageData.data, img.width, img.height);

// corners is a Float32Array with stride 7 per corner:
//   [x, y, response,
//    axis0_angle, axis0_sigma, axis1_angle, axis1_sigma, ...]
for (let i = 0; i < corners.length; i += 7) {
    const x = corners[i];
    const y = corners[i + 1];
    const response = corners[i + 2];
    const axis0_angle = corners[i + 3]; // radians, in [0, PI)
    const axis1_angle = corners[i + 5]; // radians, in (axis0, axis0 + PI)
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
    const corners = detector.detectRgba(imageData.data, canvas.width, canvas.height);
    drawCorners(corners); // your rendering logic

    requestAnimationFrame(processFrame);
}

processFrame();
```

### Response map visualization (diagnostics)

The diagnostics methods expose intermediate detector data — raw
response maps and Radon heatmaps — for debugging and visualization.
They are opt-in and not part of the normal detection result, which is
the `Float32Array` returned by `detect` / `detectRgba`.

```js
const detector = new ChessDetector();

// Get the raw ChESS response as a Float32Array (row-major, width x height).
const response = detector.diagnosticsResponseRgba(imageData.data, width, height);
const rWidth = detector.diagnosticsResponseWidth();
const rHeight = detector.diagnosticsResponseHeight();

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
  DetectionStrategy,
  ForstnerConfig,
  MultiscaleConfig,
  OrientationMethod,
  PeakFitMode,
  RadonConfig,
  UpscaleConfig,
} from '@vitavision/chess-corners';

await init();

const cfg = DetectorConfig.chessMultiscale();

// Top-level fields are simple getters / setters:
cfg.threshold = 60.0; // ChESS: absolute floor on the raw response (default 30)
cfg.multiscale = MultiscaleConfig.pyramid(4, 64, 3); // levels, minSize, refinementRadius
cfg.upscale = UpscaleConfig.fixed(2);
cfg.orientationMethod = OrientationMethod.DiskFit;
cfg.mergeRadius = 2.5;
// cfg = cfg.withoutOrientation(); // skip the per-corner fit (axis values become NaN)

// Strategy selects ChESS vs Radon and carries the detector tuning:
const chess = new ChessConfig();
chess.ring = ChessRing.Broad;
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
| `detectRgba(pixels, w, h)` | Detect corners from RGBA `Uint8Array` |

#### Diagnostics

Opt-in methods that expose intermediate detector data for debugging and
visualization. They are not part of the normal detection result.

| Method | Description |
|--------|-------------|
| `diagnosticsResponse(pixels, w, h)` | Compute response map from grayscale pixels |
| `diagnosticsResponseRgba(pixels, w, h)` | Compute response map from RGBA pixels |
| `diagnosticsResponseWidth()` | Width of the last computed response map |
| `diagnosticsResponseHeight()` | Height of the last computed response map |
| `diagnosticsRadonHeatmap(pixels, w, h)` | Compute the Radon heatmap from grayscale pixels |
| `diagnosticsRadonHeatmapRgba(pixels, w, h)` | Compute the Radon heatmap from RGBA pixels |
| `diagnosticsRadonHeatmapWidth()` | Width of the last computed Radon heatmap (working resolution) |
| `diagnosticsRadonHeatmapHeight()` | Height of the last computed Radon heatmap |
| `diagnosticsRadonHeatmapScale()` | Working-to-input scale factor for the last heatmap |

### Output format

**Corners** (`detect` / `detectRgba`): `Float32Array` with stride 7 per corner:

| Offset | Field | Description |
|--------|-------|-------------|
| `i + 0` | `x` | Subpixel x coordinate |
| `i + 1` | `y` | Subpixel y coordinate |
| `i + 2` | `response` | ChESS response strength |
| `i + 3` | `axis0_angle` | First grid axis, radians in `[0, π)` |
| `i + 4` | `axis0_sigma` | 1σ uncertainty of `axis0_angle` |
| `i + 5` | `axis1_angle` | Second grid axis, radians in `(axis0, axis0 + π)` |
| `i + 6` | `axis1_sigma` | 1σ uncertainty of `axis1_angle` |

Rotating CCW from `axis0_angle` toward `axis1_angle` traverses a **dark** sector of the corner. The two grid axes are not assumed orthogonal, so the layout can represent projective warp instead of forcing a right-angle model.

The orientation fit is the dominant per-corner cost, and it is optional. When it is skipped — `const bare = cfg.withoutOrientation()` — the four axis values (`axis0_angle`, `axis0_sigma`, `axis1_angle`, `axis1_sigma`) are `NaN` for every corner and the stride stays 7. Skip it when a downstream stage recovers board geometry from corner positions alone.

**Response map** (`diagnosticsResponse` / `diagnosticsResponseRgba`): `Float32Array` in row-major order, dimensions available via `diagnosticsResponseWidth()` / `diagnosticsResponseHeight()`.

## Binary size

~186 KB raw, ~69 KB gzipped (single-scale, no parallelism, no SIMD; default features, no `--features` flags). Reproduce with:

```bash
wasm-pack build crates/chess-corners-wasm --target web --release
```

then measure `crates/chess-corners-wasm/pkg/*.wasm` directly and with `gzip -c <file>.wasm | wc -c`.

## License

MIT
