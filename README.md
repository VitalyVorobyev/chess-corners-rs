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
`DetectorConfig` type:

| Stage                | Options                                                                              | Book                                                                    |
|----------------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Detector             | `ChESS` (ring), `Radon` (rays)                                                       | [Part III](book/src/part-03-chess-detector.md), [Part IV](book/src/part-04-radon-detector.md) |
| Subpixel refinement  | `CenterOfMass`, `Förstner`, `SaddlePoint`, `RadonPeak`, `ML` (optional ONNX)         | [Part V](book/src/part-05-refiners.md)                                  |
| Scale handling       | Single-scale or coarse-to-fine 2× pyramid                                            | [Part VII](book/src/part-07-multiscale-and-pyramids.md)                 |

The full book is at
<https://vitalyvorobyev.github.io/chess-corners-rs/>, with a
measurement-driven refiner comparison in
[Part VIII](book/src/part-08-benchmarks.md).

The workspace publishes four library crates and ships a CLI, a
Python package, and a WebAssembly package — all consuming the same
JSON schema.

## Workspace

```
chess-corners-py       PyO3 bindings         pip: chess-corners
chess-corners-wasm     wasm-bindgen          npm: @vitavision/chess-corners
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
chess-corners = "0.11"
image = "0.25"
```

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();

// Defaults: ChESS detector, multiscale pipeline, CenterOfMass refiner.
let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::forstner());

let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;
for c in &corners {
    println!("({:.2}, {:.2})  response={:.1}", c.x, c.y, c.response);
}
```

Four presets cover the most common setups:

| Preset                                 | Detector | Scale           |
|----------------------------------------|----------|-----------------|
| `DetectorConfig::chess()`              | ChESS    | Single-scale    |
| `DetectorConfig::chess_multiscale()`   | ChESS    | 3-level pyramid |
| `DetectorConfig::radon()`              | Radon    | Single-scale    |
| `DetectorConfig::radon_multiscale()`   | Radon    | 3-level pyramid |

Switching to the Radon detector is one line:

```rust
let cfg = DetectorConfig::radon();
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;
```

### Throughput on `testimages/mid.png` (1024×576)

Wall time for `Detector::detect()` on the [`mid.png`](testimages/mid.png)
test image, single-thread release build on an M-class CPU. Each
config uses the threshold that emits the same 77 true X-junctions
with zero false positives (precision = recall = 1), so the
wall-times are apples-to-apples — see note below. Median + p95 over
50 runs after a 5-run warmup so the pyramid / scratch buffers are
amortised:

| Config                                 | Corners | Median  | p95     |
|----------------------------------------|--------:|--------:|--------:|
| `DetectorConfig::chess()`              |      77 |  4.5 ms |  4.6 ms |
| `DetectorConfig::chess_multiscale()`   |      77 |  0.8 ms |  0.9 ms |
| `DetectorConfig::radon()`              |      77 |   28 ms |   29 ms |
| `DetectorConfig::radon_multiscale()`   |      77 |  3.0 ms |  3.1 ms |

ChESS-multiscale is the default and the fastest path. Radon-multiscale
is ~3.5× more expensive on this clean image because it accumulates a
full whole-image SAT before peak detection — its win is on hostile
imagery (heavy blur, low contrast), not on clean boards. Reproduce
with `python tools/render_readme_perf.py`.

Thresholds used in the table above: `chess()` and `chess_multiscale()`
both use `Threshold::Absolute(100.0)`; `radon()` uses
`Threshold::Relative(0.28)` and `radon_multiscale()` uses
`Threshold::Relative(0.34)`. Each value was tuned to recover exactly
the 77 X-junctions on this image. ChESS and Radon respond at
different scales, so the natural threshold type differs between
them.

Algorithm details, accuracy sweeps (vs blur, noise, cell size), and
the cost of each refiner are measured in
[Part VIII of the book](book/src/part-08-benchmarks.md) — including
the full feature-flag matrix (`simd`, `rayon`, `par_pyramid`) on
small / mid / large frames.

## Python quick start

```bash
python -m pip install chess-corners
```

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.DetectorConfig.chess_multiscale()
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.forstner()

detector = chess_corners.Detector(cfg)
corners = detector.detect(img)
print(corners.shape)   # (N, 9)
print(cfg)
```

`Detector.detect(image)` returns a `float32 [N, 9]` array with stride
9 per corner: `[x, y, response, contrast, fit_rms, axis0_angle,
axis0_sigma, axis1_angle, axis1_sigma]`.

Every public Python config object supports `to_dict()`,
`from_dict()`, `to_json()`, `from_json()`, `pretty()`, and `print()`.

## JavaScript / WebAssembly

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

```js
import init, {
  ChessDetector,
  ChessConfig,
  ChessRefiner,
  ChessRing,
  DetectionStrategy,
  DetectorConfig,
  ForstnerConfig,
} from '@vitavision/chess-corners';

await init();

const cfg = DetectorConfig.chessMultiscale();
const chess = new ChessConfig();
chess.ring = ChessRing.Broad;
chess.refiner = ChessRefiner.withForstner(new ForstnerConfig());
cfg.strategy = DetectionStrategy.fromChess(chess);

const detector = ChessDetector.withConfig(cfg);

const imageData = ctx.getImageData(0, 0, width, height);
const corners = detector.detect_rgba(imageData.data, width, height);

// Same stride-9 layout as Python.
for (let i = 0; i < corners.length; i += 9) {
    console.log(`(${corners[i].toFixed(2)}, ${corners[i + 1].toFixed(2)})`);
}
```

See `crates/chess-corners-wasm/README.md` for the full typed-config
surface.

## CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

The CLI loads an image plus JSON config, runs the detector, and
writes a JSON summary and an overlay PNG. Example configs:

- `config/chess_algorithm_config_example.json` — pure `DetectorConfig`
  shape, shared by the Rust and Python APIs.
- `config/chess_cli_config_example.json` — algorithm config plus the
  CLI runner envelope (`image`, `output_json`, `output_png`,
  `log_level`, `ml`).
- `config/chess_cli_config_example_ml.json` — ML refiner enabled;
  requires `--features ml-refiner` at build time.

## `DetectorConfig` schema

`DetectorConfig` has one place for every knob. Cross-cutting fields
(threshold, multiscale, upscale, orientation method, merge radius)
sit at the top level. Detector-specific tuning is nested inside the
active strategy variant. Refiners are per-detector: `ChessRefiner`
for ChESS, `RadonRefiner` for Radon — the type system rules out
invalid pairings.

```json
{
  "strategy": {
    "chess": {
      "ring": "canonical",
      "descriptor_ring": "canonical",
      "nms_radius": 3,
      "min_cluster_size": 1,
      "refiner": {
        "forstner": {
          "radius": 3,
          "min_trace": 20.0,
          "min_det": 0.001,
          "max_condition_number": 60.0,
          "max_offset": 2.0
        }
      }
    }
  },
  "threshold": { "absolute": 0.5 },
  "multiscale": {
    "pyramid": { "levels": 3, "min_size": 96, "refinement_radius": 4 }
  },
  "upscale": "disabled",
  "orientation_method": "ring_fit",
  "merge_radius": 2.5
}
```

Switch to the Radon strategy by replacing the `strategy` value:

```json
"strategy": {
  "radon": {
    "ray_radius": 4,
    "image_upsample": 2,
    "response_blur_radius": 1,
    "peak_fit": "gaussian",
    "nms_radius": 4,
    "min_cluster_size": 2,
    "refiner": { "radon_peak": {} }
  }
}
```

`orientation_method` controls the two-axis orientation fit applied to
each detected corner:

- `ring_fit` *(default)* — fits the parametric two-axis chessboard
  intensity model to 16 ring samples via Gauss-Newton, with per-axis
  1σ uncertainties calibrated by a piecewise-linear lookup table.
- `disk_fit` — full-disk crossing-line estimator. Samples all image
  pixels in a disk around the corner center and fits two possibly
  non-orthogonal axes. Falls back to `ring_fit` on clean orthogonal
  corners and near image borders. Use when corners are imaged under
  strong projective warp.

`strategy.chess.ring` selects the ChESS sampling ring:

- `canonical` — radius-5 ring (paper default).
- `broad` — radius-10 ring; wider support window for heavily blurred imagery.

`strategy.chess.descriptor_ring` defaults to `follow_detector`; set
it to `canonical` or `broad` to sample the descriptor ring at a
different radius than the detector.

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

![Projective-warp orientation overlays](book/src/img/readme_warp_overlays.png)

Reproduce with `python tools/render_readme_warp_overlays.py`.

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
