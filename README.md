# chess-corners-rs

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)

Fast Rust library for detecting chessboard corners — the X-junctions
where four alternating dark/bright cells meet — to sub-pixel
precision. The kind of detector that sits at the front of a camera
calibration, pose estimation, or AR alignment pipeline.

![](book/src/img/mid_chess.png)

Two independent detectors and five subpixel refiners sit behind one
`ChessConfig` type:

| Stage                | Options                                                                              | Book                                                                    |
|----------------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Detector             | `ChESS` (ring), `Radon` (rays)                                                       | [Part III](book/src/part-03-chess-detector.md), [Part IV](book/src/part-04-radon-detector.md) |
| Subpixel refinement  | `CenterOfMass`, `Förstner`, `SaddlePoint`, `RadonPeak`, `ML` (optional ONNX)         | [Part V](book/src/part-05-refiners.md)                                  |
| Scale handling       | Single-scale or coarse-to-fine 2× pyramid                                            | [Part VI](book/src/part-06-multiscale-and-pyramids.md)                  |

The full book is at
<https://vitalyvorobyev.github.io/chess-corners-rs/>, with a
measurement-driven refiner comparison in
[Part VII](book/src/part-07-benchmarks.md).

The workspace publishes four library crates and ships a CLI, a
Python package, and a WebAssembly package — all consuming the same
JSON schema.

## Workspace

```
chess-corners-py       PyO3 bindings         pip: chess-corners
chess-corners-wasm     wasm-bindgen          npm: chess-corners-wasm
       │
       ▼
chess-corners          high-level API, CLI, multiscale pipeline
       │
       ▼
chess-corners-core     ChESS + Radon detectors, refiners, descriptors

box-image-pyramid      standalone u8 pyramid builder (no chess coupling)
chess-corners-ml       optional ONNX refiner (`ml-refiner` feature)
```

Layering rules are enforced in CI — `core` does not depend on the
facade, `box-image-pyramid` is chess-agnostic and reusable elsewhere,
and every optional dependency is feature-gated. See `AGENTS.md` and
`CLAUDE.md` for the full rules.

## Rust quick start

```toml
[dependencies]
chess-corners = "0.5"
image = "0.25"
```

```rust
use chess_corners::{find_chess_corners_image, ChessConfig, RefinementMethod};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();

// Defaults: ChESS detector, multiscale pipeline, CenterOfMass refiner.
let mut cfg = ChessConfig::multiscale();
cfg.refiner.kind = RefinementMethod::Forstner;

let corners = find_chess_corners_image(&img, &cfg);
for c in &corners {
    println!("({:.2}, {:.2})  response={:.1}", c.x, c.y, c.response);
}
```

Switching to the Radon detector is one line:

```rust
let cfg = ChessConfig::radon();
let corners = find_chess_corners_image(&img, &cfg);
```

Switching the refiner is also one line:

```rust
cfg.refiner.kind = RefinementMethod::RadonPeak;
```

Algorithm details and the cost of each option are measured in
[Part VII](book/src/part-07-benchmarks.md).

## Python quick start

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
print(cfg)
```

`find_chess_corners` returns a `float32 [N, 9]` array with stride 9
per corner: `[x, y, response, contrast, fit_rms,
axis0_angle, axis0_sigma, axis1_angle, axis1_sigma]`.

Every public Python config object supports `to_dict()`,
`from_dict()`, `to_json()`, `from_json()`, `pretty()`, and `print()`.

## JavaScript / WebAssembly

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

```js
import init, { ChessDetector } from 'chess-corners-wasm';

await init();
const detector = new ChessDetector();
detector.set_pyramid_levels(3);

const imageData = ctx.getImageData(0, 0, width, height);
const corners = detector.detect_rgba(imageData.data, width, height);

// Same stride-9 layout as Python.
for (let i = 0; i < corners.length; i += 9) {
    console.log(`(${corners[i].toFixed(2)}, ${corners[i + 1].toFixed(2)})`);
}
```

See `crates/chess-corners-wasm/README.md` for the full setter list,
including `set_detector_mode('canonical' | 'broad' | 'radon')`.

## CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

The CLI loads an image plus JSON config, runs the detector, and
writes a JSON summary and an overlay PNG. Example configs:

- `config/chess_algorithm_config_example.json` — just the algorithm
  fields, shared by the Rust and Python APIs.
- `config/chess_cli_config_example.json` — algorithm + CLI I/O
  fields.
- `config/chess_cli_config_example_ml.json` — ML refiner enabled;
  requires `--features ml-refiner` at build time.

## `ChessConfig` schema

Flat schema; the same field names appear in Rust, Python, CLI JSON,
and WASM setters.

```json
{
  "detector_mode": "broad",
  "descriptor_mode": "canonical",
  "threshold_mode": "absolute",
  "threshold_value": 0.5,
  "nms_radius": 3,
  "min_cluster_size": 1,
  "refiner": {
    "kind": "forstner",
    "center_of_mass": { "radius": 2 },
    "forstner": {
      "radius": 3,
      "min_trace": 20.0,
      "min_det": 0.001,
      "max_condition_number": 60.0,
      "max_offset": 2.0
    },
    "saddle_point": {
      "radius": 3,
      "det_margin": 0.002,
      "max_offset": 1.75,
      "min_abs_det": 0.0002
    },
    "radon_peak": {
      "ray_radius": 2,
      "patch_radius": 3,
      "image_upsample": 2,
      "response_blur_radius": 1,
      "peak_fit": "gaussian",
      "min_response": 0.0,
      "max_offset": 1.5
    }
  },
  "radon_detector": {
    "ray_radius": 4,
    "image_upsample": 2,
    "response_blur_radius": 1,
    "peak_fit": "gaussian",
    "threshold_rel": 0.01,
    "nms_radius": 4,
    "min_cluster_size": 2
  },
  "pyramid_levels": 3,
  "pyramid_min_size": 96,
  "refinement_radius": 4,
  "merge_radius": 2.5
}
```

`detector_mode` picks a detector:

- `canonical` — ChESS with radius-5 ring (default).
- `broad` — ChESS with radius-10 ring; more blur-tolerant.
- `radon` — Duda-Frese Radon detector; handles small cells and
  heavy blur. Configuration lives under `radon_detector`.

`descriptor_mode` follows the detector by default; you can override
it to sample the descriptor ring at the other ChESS radius.

## Corner descriptor

Every detection returns a `CornerDescriptor`:

| Field        | Type                | Meaning                                                                        |
|--------------|---------------------|--------------------------------------------------------------------------------|
| `x`, `y`     | `f32`               | Subpixel position in input image pixels.                                       |
| `response`   | `f32`               | Raw detector response at the peak. Scale is detector-specific.                  |
| `contrast`   | `f32` (≥ 0)         | Bright/dark amplitude from the ring intensity fit (gray levels).                |
| `fit_rms`    | `f32` (≥ 0)         | RMS residual of that fit (gray levels).                                         |
| `axes[0, 1]` | `[AxisEstimate; 2]` | Two local grid axis directions with per-axis 1σ angular uncertainty.            |

The two axes are **not** assumed orthogonal — projective warp or lens
distortion tilts the sectors independently, and the fit captures both.
Axis 0 lives in `[0, π)`; axis 1 lies in `(axis0, axis0 + π)`, with
the CCW arc from axis 0 to axis 1 crossing a dark sector. Full
derivation in
[Part III §3.4](book/src/part-03-chess-detector.md#34-corner-descriptors).

## Feature flags

| Feature       | Crate                 | Effect                                                                      |
|---------------|-----------------------|-----------------------------------------------------------------------------|
| `image`       | facade (default)      | `image::GrayImage` convenience entry points.                                |
| `rayon`       | core, facade          | Parallel ChESS response + multiscale refinement over cores.                 |
| `simd`        | core, facade          | Portable SIMD for the ChESS kernel. Requires nightly Rust.                  |
| `par_pyramid` | box-image-pyramid     | SIMD / Rayon inside the pyramid downsampler.                                |
| `tracing`     | core, facade          | JSON spans for the detector pipeline.                                       |
| `ml-refiner`  | facade                | Enables the ONNX refiner (via `chess-corners-ml`).                          |
| `cli`         | facade                | Builds the `chess-corners` binary.                                          |
| `radon-sat-u32` | core                | Switches the Radon SAT accumulator to `u32` for lower memory, ~16 MP cap.   |

All feature combinations produce the same numerical results; flags
only affect performance and observability.

## Diligence statement

This project is developed with AI coding assistants as implementation
tools. The author validates algorithmic behaviour and numerical
results and enforces the quality gates below (fmt, clippy, tests,
docs, mdbook, Python checks) before every release.

## Pre-PR quality gates

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
mdbook build book
```
