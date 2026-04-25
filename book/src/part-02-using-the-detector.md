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
chess-corners = "0.5"
image = "0.25"          # optional, for GrayImage integration
```

### 2.1.1 Single-scale ChESS detection from an image file

```rust
use chess_corners::{find_chess_corners_image, ChessConfig};
use image::io::Reader as ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    let cfg = ChessConfig::single_scale();  // ChESS detector, defaults
    let corners = find_chess_corners_image(&img, &cfg);

    println!("found {} corners", corners.len());
    Ok(())
}
```

`corners` is a `Vec<CornerDescriptor>` with subpixel positions and
per-corner intensity-fit metadata (Part I §1.4).

### 2.1.2 Radon detector instead of ChESS

```rust
use chess_corners::{find_chess_corners_image, ChessConfig};

let cfg = ChessConfig::radon();           // Radon detector, paper defaults
let corners = find_chess_corners_image(&img, &cfg);
```

`ChessConfig::radon()` sets `detector_mode = Radon` and populates
`radon_detector: RadonDetectorParams` with the paper's published
defaults. The output type is the same `Vec<CornerDescriptor>`.

### 2.1.3 Swapping the subpixel refiner

```rust
use chess_corners::{ChessConfig, RefinementMethod};

let mut cfg = ChessConfig::multiscale();
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
use chess_corners::{find_chess_corners_u8, ChessConfig, ThresholdMode};

fn detect(img: &[u8], width: u32, height: u32) {
    let mut cfg = ChessConfig::single_scale();
    cfg.threshold_mode = ThresholdMode::Relative;
    cfg.threshold_value = 0.2;

    let corners = find_chess_corners_u8(img, width, height, &cfg);
    println!("found {} corners", corners.len());
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

### 2.1.6 Key `ChessConfig` fields

The config is intentionally flat; all scalar knobs live at the top
level, only the refiner is nested:

| Field                 | Meaning                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `detector_mode`       | `Canonical` / `Broad` / `Radon`. `Canonical` and `Broad` pick ChESS ring widths; `Radon` switches detector entirely. |
| `descriptor_mode`     | `FollowDetector`, `Canonical`, or `Broad`. Lets you run detection at one ring radius and descriptor sampling at another. |
| `threshold_mode`      | `Absolute` or `Relative`. Applied to the detector response.                                       |
| `threshold_value`     | Threshold number, interpreted per `threshold_mode`.                                              |
| `nms_radius`          | Non-maximum-suppression window half-radius.                                                      |
| `min_cluster_size`    | Minimum number of positive-response neighbors inside the NMS window.                             |
| `refiner`             | Nested refiner selection + per-refiner configs (see Part V).                                      |
| `pyramid_levels`      | `1` means single-scale; `> 1` enables the coarse-to-fine multiscale pipeline.                    |
| `pyramid_min_size`    | Smallest pyramid level dimension allowed during construction.                                    |
| `refinement_radius`   | ROI size (in coarse-level pixels) used when refining coarse seeds at base resolution.            |
| `merge_radius`        | Duplicate-suppression distance (in base-level pixels) for the final merge step.                  |
| `radon_detector`      | Full `RadonDetectorParams` struct, only consulted when `detector_mode = Radon`.                   |

Presets: `ChessConfig::single_scale()`, `ChessConfig::multiscale()`,
`ChessConfig::radon()`, `ChessConfig::radon_peak()` (multiscale +
`RefinementMethod::RadonPeak`).

## 2.2 Python

Install from PyPI:

```bash
python -m pip install chess-corners
```

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.ChessConfig.multiscale()
cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

corners = chess_corners.find_chess_corners(img, cfg)
print(corners.shape)   # (N, 9)
```

`find_chess_corners` accepts a 2D `uint8` array shaped `(H, W)` and
returns a `float32` array with stride 9 per corner:

```
[x, y, response, contrast, fit_rms,
 axis0_angle, axis0_sigma, axis1_angle, axis1_sigma]
```

The Python `ChessConfig` matches the Rust type field-for-field and
supports `to_dict()`, `from_dict()`, `to_json()`, `from_json()`,
`pretty()`, and `print()`.

The Radon detector is selected the same way as in Rust:

```python
cfg = chess_corners.ChessConfig.radon()
```

If the wheel was built with `ml-refiner`, the ML entry point is
`chess_corners.find_chess_corners_with_ml(img, cfg)`.

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
const detector = new ChessDetector();
detector.set_pyramid_levels(3);

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

`ChessDetector` exposes setters for the same fields as `ChessConfig`,
plus `set_detector_mode('canonical' | 'broad' | 'radon')` and
tuning setters for each refiner. See
`crates/chess-corners-wasm/README.md` for the full setter list.

## 2.4 CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

The CLI:

- Loads the image at the config's `image` field.
- Picks single-scale or multiscale from `pyramid_levels`.
- Picks ChESS or Radon from `detector_mode`.
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
wheel built with the same feature (Python), then call the ML
entry points:

```rust
use chess_corners::{find_chess_corners_image_with_ml, ChessConfig};

let cfg = ChessConfig::multiscale();
let corners = find_chess_corners_image_with_ml(&img, &cfg);
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

Pairing the ML refiner with `detector_mode = Radon` is not supported:
the Radon detector does its own subpixel fit and does not emit the
integer seeds the ML refiner expects, so the facade falls back to
the Radon detector's native output in that case.

## 2.6 Radon heatmap (visualization)

The whole-image Radon detector (`detector_mode = "radon"`) computes a
dense `(max_α S_α − min_α S_α)²` response heatmap as an intermediate
step. The heatmap is exposed across all wrappers for visualization,
debugging, and downstream tooling — useful when tuning
`ray_radius`, `image_upsample`, or the threshold floor.

The heatmap is returned at *working resolution*: the input is
optionally upscaled (`ChessConfig.upscale`) and then internally
supersampled by the Radon detector (`radon_detector.image_upsample`,
default 2). The working-to-input scale factor is therefore
`upscale_factor * image_upsample`. Multiply input-pixel coordinates by
this factor to land on heatmap pixels.

Rust:

```rust,no_run
use chess_corners::{radon_heatmap_u8, ChessConfig};

# fn run(img: &[u8], width: u32, height: u32) {
let cfg = ChessConfig::radon();
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

cfg = chess_corners.ChessConfig.radon()
heatmap = chess_corners.radon_heatmap(img, cfg)  # (H', W') float32
print(heatmap.shape, heatmap.dtype, float(heatmap.max()))
```

WebAssembly (JS):

```js
import init, { ChessDetector } from 'chess-corners-wasm';

await init();
const det = new ChessDetector();
det.set_detector_mode('radon');

const heatmap = det.radon_heatmap(grayPixels, width, height);
const w = det.radon_heatmap_width();
const h = det.radon_heatmap_height();
const scale = det.radon_heatmap_scale();  // working-to-input factor
console.log('heatmap', w, 'x', h, 'scale', scale);
```

The heatmap is independent from corner detection: calling it does not
require the detector mode to actually be `"radon"`, and it does not
return corners.

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
