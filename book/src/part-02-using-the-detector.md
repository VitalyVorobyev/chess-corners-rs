# Part II: Using the library

This chapter is a walk through the public API on every binding
target. Code-first; algorithms are covered in
[Part III (ChESS)](part-03-chess-detector.md),
[Part IV (Radon)](part-04-radon-detector.md), and
[Part V (refiners)](part-05-refiners.md).

## 2.1 Rust

Add the facade crate:

```toml
[dependencies]
chess-corners = "0.10"
image = "0.25"          # optional, for GrayImage integration
```

### 2.1.1 Single-scale ChESS detection from an image file

```rust
use chess_corners::{DetectorConfig, Detector};
use image::io::Reader as ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    let cfg = DetectorConfig::single_scale();  // ChESS detector, defaults
    let mut detector = Detector::new(cfg)?;
    let corners = detector.detect(&img)?;

    println!("found {} corners", corners.len());
    Ok(())
}
```

`corners` is a `Vec<CornerDescriptor>` with subpixel positions and
per-corner intensity-fit metadata (Part I §1.4).

### 2.1.2 Radon detector instead of ChESS

```rust
use chess_corners::{DetectorConfig, Detector};

let cfg = DetectorConfig::radon();           // Radon detector, paper defaults
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;
```

`DetectorConfig::radon()` selects `DetectionStrategy::Radon` and populates
`RadonStrategy` with the paper's published defaults. The output type is
the same `Vec<CornerDescriptor>`.

Pick Radon when ChESS's 16-sample ring fails — heavy motion blur, strong
defocus, low contrast, or cells smaller than roughly `2·ring_radius`.
For throughput, ChESS is faster; see [Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon).

### 2.1.3 Swapping the subpixel refiner

```rust
use chess_corners::{DetectorConfig, RefinementMethod};

let mut cfg = DetectorConfig::multiscale();
cfg.refiner.kind = RefinementMethod::RadonPeak;  // or CenterOfMass / Forstner / SaddlePoint
```

The refiner is a one-line change. Per-refiner configuration lives
under `cfg.refiner.center_of_mass`, `cfg.refiner.forstner`,
`cfg.refiner.saddle_point`, `cfg.refiner.radon_peak`. Defaults match
[Part V](part-05-refiners.md).

### 2.1.4 Raw buffer API

If your pixels come from a camera SDK, FFI, or GPU pipeline, skip
the `image` crate and feed a packed `&[u8]`:

```rust
use chess_corners::{DetectorConfig, Detector, Threshold};

fn detect(img: &[u8], width: u32, height: u32) -> Result<(), chess_corners::ChessError> {
    let mut cfg = DetectorConfig::single_scale();
    cfg.threshold = Threshold::Relative(0.2);

    let mut detector = Detector::new(cfg)?;
    let corners = detector.detect_u8(img, width, height)?;
    println!("found {} corners", corners.len());
    Ok(())
}
```

Requirements:

- `img` is `width * height` bytes, row-major.
- `0` is black, `255` is white.

If your buffer has a stride or is interleaved RGB, copy the
luminance channel to a packed buffer first. The facade does not
resample stride; the only supported layout is tightly packed.

### 2.1.5 Inspecting corners

```rust
for c in &corners {
    println!(
        "({:.2}, {:.2})  response={:.1}  axes=({:.2}, {:.2}) rad",
        c.x, c.y, c.response,
        c.axes[0].angle, c.axes[1].angle,
    );
}
```

The `axes` field gives **two** directions, both in radians; they are
not assumed orthogonal. `c.axes[0].sigma` and `c.axes[1].sigma` are
1σ angular uncertainties. See [Part III §3.4](part-03-chess-detector.md#34-corner-descriptors)
for the fit and the polarity convention.

### 2.1.6 Key `DetectorConfig` fields

`DetectorConfig` shares cross-cutting fields at the top level and
groups detector-specific tuning under the `strategy` enum:

| Field                 | Meaning                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `strategy`            | `DetectionStrategy::Chess(ChessStrategy)` or `DetectionStrategy::Radon(RadonStrategy)`. Detector-specific parameters (ring width, NMS) live inside the variant. |
| `threshold`           | `Threshold::Absolute(f32)` or `Threshold::Relative(f32)`. `Absolute(0.0)` encodes the ChESS paper's `R > 0` contract; `Relative(0.01)` is the Radon preset default. |
| `multiscale`          | `Some(MultiscaleParams { pyramid_levels, pyramid_min_size, refinement_radius })` to enable coarse-to-fine; `None` runs single-scale. Honoured by both detectors. |
| `descriptor_mode`     | `FollowDetector`, `Canonical`, or `Broad`. Lets you run detection at one ring radius and descriptor sampling at another. |
| `refiner`             | Nested refiner selection + per-refiner configs (see Part V). Honoured by both detectors.          |
| `merge_radius`        | Duplicate-suppression distance (in base-level pixels) for the final merge step. Honoured by both detectors. |

Inside `ChessStrategy`:

| Field               | Meaning                                                                               |
|---------------------|---------------------------------------------------------------------------------------|
| `ring`              | `ChessRing::Canonical` or `ChessRing::Broad` — ring width used for ChESS response.   |
| `nms_radius`        | Non-maximum-suppression window half-radius (input pixels).                            |
| `min_cluster_size`  | Minimum positive-response neighbors inside the NMS window.                            |

Presets: `DetectorConfig::single_scale()`, `DetectorConfig::multiscale()`,
`DetectorConfig::radon()`, `DetectorConfig::radon_multiscale()`.

## 2.2 Python

Install from PyPI:

```bash
python -m pip install chess-corners
```

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.DetectorConfig.multiscale()
cfg.threshold = chess_corners.Threshold.relative(0.15)
cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

detector = chess_corners.Detector(cfg)
corners = detector.detect(img)
print(corners.shape)   # (N, 9)
```

`Detector(cfg).detect(image)` accepts a 2D `uint8` array shaped
`(H, W)` and returns a `float32` array with stride 9 per corner:

```
[x, y, response, contrast, fit_rms,
 axis0_angle, axis0_sigma, axis1_angle, axis1_sigma]
```

The Python `DetectorConfig` matches the Rust type field-for-field and
supports `to_dict()`, `from_dict()`, `to_json()`, `from_json()`,
`pretty()`, and `print()`.

The Radon detector is selected the same way as in Rust:

```python
cfg = chess_corners.DetectorConfig.radon()
```

If the wheel was built with `ml-refiner`, the ML pipeline is reached
through the same `Detector(cfg).detect(img)` call once
`cfg.refiner.kind` is set to the `Ml` variant.

## 2.3 JavaScript / WebAssembly

Build the wasm package from source:

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

Or consume the published npm package `chess-corners-wasm`. Usage
from a web app:

```js
import init, { ChessDetector } from 'chess-corners-wasm';

await init();
const detector = new ChessDetector();          // defaults to multiscale ChESS
detector.set_pyramid_levels(3);               // sets ChessStrategy.multiscale levels

// From a canvas (webcam frame, loaded image, etc.)
const imageData = ctx.getImageData(0, 0, width, height);
const corners = detector.detect_rgba(imageData.data, width, height);

// corners is Float32Array, stride 9 per corner — same layout as Python.
for (let i = 0; i < corners.length; i += 9) {
    console.log(`(${corners[i].toFixed(2)}, ${corners[i + 1].toFixed(2)})`);
}

// Raw response map, for heatmap visualisation.
const response = detector.response_rgba(imageData.data, width, height);
```

`ChessDetector` exposes convenience setters for the most common
detector configuration fields. The strategy is switched with
`detector.set_detector_mode("chess")` or `"radon"`, or by constructing
a typed `DetectorConfig` (via `DetectorConfig.singleScale()` /
`.multiscale()` / `.radon()` / `.radonMultiscale()`, or by editing the
nested `strategy` object) and passing it to
`ChessDetector.withConfig(cfg)`. See
`crates/chess-corners-wasm/README.md` for the full setter list.

## 2.4 CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

The CLI:

- Loads the image at the config's `image` field.
- Picks single-scale or multiscale from the top-level `multiscale` field.
- Picks ChESS or Radon from `strategy` (the top-level variant).
- Picks the refiner from the nested `refiner` block.
- Writes a JSON summary and a PNG overlay with one mark per corner.

Example configs under `config/`:

- `chess_algorithm_config_example.json` — just the algorithm fields,
  shared with the Rust and Python APIs.
- `chess_cli_config_example.json` — algorithm fields plus CLI
  I/O fields (`image`, `output_json`, `output_png`, `log_level`, `ml`).
- `chess_cli_config_example_ml.json` — same, with the ML refiner
  enabled. Requires a binary built with `--features ml-refiner`.

Overlay examples on the sample images in `testdata/`:

![](img/small_chess.png)

![](img/mid_chess.png)

![](img/large_chess.png)

## 2.5 ML refiner

The ML refiner is a separate, optional code path. Enable it by
building with `--features ml-refiner` (Rust) or by installing a
wheel built with the same feature (Python), then pick the ML
variant on the refiner config:

```rust
# #[cfg(feature = "ml-refiner")]
# {
use chess_corners::{DetectorConfig, Detector, RefinementMethod};

let mut cfg = DetectorConfig::multiscale();
cfg.refiner.kind = RefinementMethod::Ml;

let mut detector = Detector::new(cfg).unwrap();
let corners = detector.detect(&img).unwrap();
# }
```

The ML path:

- Runs the ChESS detector to produce seeds.
- Feeds each seed's 21×21 neighborhood through the embedded
  ONNX model (`chess_refiner_v4.onnx`, ~180 K params).
- Replaces the seed position with the model's predicted
  `(dx, dy)` offset.
- Falls back to the configured classical refiner if the ML path
  rejects or times out.

The algorithm and its limits are covered in
[Part V §5.6](part-05-refiners.md#56-ml-onnx-model). The ML refiner
is not a direct replacement for RadonPeak: on clean and lightly
blurred data RadonPeak is more accurate; ML wins on noise-heavy
scenes.

Pairing the ML refiner with `DetectionStrategy::Radon` is not supported:
the Radon detector does its own subpixel fit and does not emit the
integer seeds the ML refiner expects, so the facade falls back to
the Radon detector's native output in that case.

## 2.6 Radon heatmap (visualization)

The Radon detector computes a dense `(max_α S_α − min_α S_α)²`
response heatmap as an intermediate step. The heatmap is exposed
across all wrappers for visualization, debugging, and downstream
tooling — useful when tuning `ray_radius`, `image_upsample`, or the
threshold floor.

The heatmap is returned at *working resolution*: the input is
optionally upscaled (`DetectorConfig.upscale`) and then internally
supersampled by the Radon detector (`RadonStrategy.image_upsample`,
default 2). The working-to-input scale factor is therefore
`upscale_factor * image_upsample`. Multiply input-pixel coordinates by
this factor to land on heatmap pixels.

Rust:

```rust,no_run
use chess_corners::{radon_heatmap_u8, DetectorConfig};

# fn run(img: &[u8], width: u32, height: u32) {
let cfg = DetectorConfig::radon();
let map = radon_heatmap_u8(img, width, height, &cfg);
println!("heatmap {}×{}, max = {:.1}",
    map.width(), map.height(),
    map.data().iter().copied().fold(f32::NEG_INFINITY, f32::max));
# }
```

Python:

```python
import chess_corners
import numpy as np

cfg = chess_corners.DetectorConfig.radon()
heatmap = chess_corners.radon_heatmap(img, cfg)  # (H', W') float32
print(heatmap.shape, heatmap.dtype, float(heatmap.max()))
```

WebAssembly (JS):

```js
import init, { ChessDetector } from 'chess-corners-wasm';

await init();
const det = new ChessDetector();
det.use_radon();   // switch strategy to Radon

const heatmap = det.radon_heatmap(grayPixels, width, height);
const w = det.radon_heatmap_width();
const h = det.radon_heatmap_height();
const scale = det.radon_heatmap_scale();  // working-to-input factor
console.log('heatmap', w, 'x', h, 'scale', scale);
```

The heatmap is independent from corner detection: calling it does not
require the active strategy to be Radon, and it does not return
corners.

---

In this part we focused on the public faces of the detector: the
`image` helper, the raw buffer API, the CLI, and the Python bindings.
In the next parts we will look under the hood at how the ChESS
response is computed, how the detector turns responses into subpixel
corners, and how the multiscale pipeline is structured.

---

Next: [Part III](part-03-chess-detector.md) describes the ChESS
response kernel, the detection pipeline, and the corner descriptor
fit. [Part IV](part-04-radon-detector.md) covers the Radon detector.
