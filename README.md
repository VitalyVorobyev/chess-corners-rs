<h1>
  <a href="https://vitavision.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="book/src/img/vv-favicon-dark.svg">
      <img src="book/src/img/vv-favicon-light.svg" alt="vitavision.dev" height="48" align="left">
    </picture>
  </a>
  &nbsp;chess-corners-rs
</h1>

Part of the [vitavision.dev](https://vitavision.dev/) computer-vision atlas.

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)
[![Live demo](https://img.shields.io/badge/demo-live-2ea44f)](https://vitalyvorobyev.github.io/chess-corners-rs/demo/)

Rust workspace for detecting chessboard X-junctions: the points where
four alternating dark and bright cells meet. The main crate returns
subpixel corner positions plus a local two-axis descriptor that can be
used by calibration, pose-estimation, and overlay pipelines.

![](book/src/img/mid_chess.png)

The public Rust API, Python package, WebAssembly package, and CLI all
use the same `DetectorConfig` schema.

## What Ships

| Stage | Options | More detail |
|-------|---------|-------------|
| Detector | `ChESS` ring response, `Radon` ray-sum response | [ChESS](book/src/part-03-chess-detector.md) · [atlas](https://vitavision.dev/atlas/chess-corners), [Radon](book/src/part-04-radon-detector.md) · [atlas](https://vitavision.dev/atlas/duda-radon-corners) |
| Refiner | `CenterOfMass`, `Förstner`, `SaddlePoint`, optional `ML` | [Refiners](book/src/part-05-refiners.md) |
| Scale handling | Single-scale or coarse-to-fine 2× pyramid | [Multiscale](book/src/part-07-multiscale-and-pyramids.md) |
| Benchmarks | Synthetic sweeps and pipeline timings | [Benchmarks](book/src/part-08-benchmarks.md) |

Workspace crates:

```text
chess-corners-py       PyO3 bindings         pip: chess-corners
chess-corners-wasm     wasm-bindgen          npm: @vitavision/chess-corners
       │
       ▼
chess-corners          public Rust API, CLI, multiscale pipeline
       │
       ▼
chess-corners-core     low-level detectors, refiners, descriptors

box-image-pyramid      standalone u8 2× pyramid builder
chess-corners-ml       optional ONNX refiner (`ml-refiner` feature)
```

`chess-corners` is the compatibility boundary. `chess-corners-core`
contains lower-level primitives for callers that need raw response maps
or custom pipelines. `box-image-pyramid` has no chess-specific
dependencies.

## Rust Quick Start

```toml
[dependencies]
chess-corners = "0.11"
image = "0.25"
```

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();

let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::forstner());

let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;

for c in &corners {
    println!("({:.2}, {:.2})  response={:.1}", c.x, c.y, c.response);
}
```

Presets:

| Preset | Detector | Scale |
|--------|----------|-------|
| `DetectorConfig::chess()` | ChESS | Single-scale |
| `DetectorConfig::chess_multiscale()` | ChESS | 3-level pyramid |
| `DetectorConfig::radon()` | Radon | Single-scale |
| `DetectorConfig::radon_multiscale()` | Radon | 3-level pyramid |

Switching detector families is a config change:

```rust
let cfg = DetectorConfig::radon();
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;
```

## Output

Each Rust detection is a `CornerDescriptor`:

| Field | Meaning |
|-------|---------|
| `x`, `y` | Subpixel position in input-image pixels |
| `response` | Raw detector response at the detected peak |
| `axes` | `Option` of two local grid-axis directions with per-axis 1σ uncertainty; `None` when the orientation fit is disabled |

The two axes are not forced to be orthogonal. This lets the descriptor
represent perspective warp and lens distortion instead of collapsing the
corner to a right-angle model. See
[Part III §3.4](book/src/part-03-chess-detector.md#34-corner-descriptors)
for the convention and derivation.

The orientation fit is the dominant per-corner cost, and it is
optional. A pipeline that recovers board geometry from corner
*positions* alone can disable it with
`DetectorConfig::without_orientation()`, which skips the fit and leaves
each descriptor's `axes` as `None`.

![Projective-warp orientation overlays](book/src/img/readme_warp_overlays.png)

Reproduce the overlay with:

```bash
python tools/render_readme_warp_overlays.py
```

## Choosing Options

Use `DetectorConfig::chess_multiscale()` as the default entry point.
It is the fastest configuration in the README benchmark below and it
keeps the detector work at a smaller pyramid level before refining in
the input image.

Use the Radon presets when the ChESS ring response does not produce
enough reliable seeds. The repository has synthetic tests and benchmark
sections for small cells, blur, and low contrast; those are the cases
where the Radon ray-sum response was added. Treat this as a measured
configuration choice, not a universal statement about all camera data:
validate on your own images when the decision matters.

Refiner choice is also workload-dependent. The benchmark chapter reports
the fixture, timing loop, and error metric used for each claim. In that
fixture, `Förstner` has the lowest mean error on clean frames, `SaddlePoint`
is most robust to blur, and the optional ML refiner has the lowest mean error
in the heaviest noise condition; the geometric refiners are much cheaper per
corner than ML.

## Performance Snapshot

Wall time for `Detector::detect()` on [`testimages/mid.png`](testimages/mid.png)
(1024×576), single-thread release build on an M-class CPU. Each config
uses a threshold tuned to emit the same 77 true X-junctions with zero
false positives on this image. The median over 50 runs after a
5-run warmup:

| Config | Corners | Median |
|--------|--------:|-------:|
| `DetectorConfig::chess()` | 77 | 4.5 ms |
| `DetectorConfig::chess_multiscale()` | 77 | 0.8 ms |
| `DetectorConfig::radon()` | 77 | 28 ms |
| `DetectorConfig::radon_multiscale()` | 77 | 3.0 ms |

This table is a narrow snapshot, not a cross-dataset ranking. It is
useful for understanding the cost of each preset on one clean test
image. Reproduce it with:

```bash
python tools/render_readme_perf.py
```

The broader benchmark chapter includes blur, noise, cell-size, feature
flag, and pipeline measurements:
[Part VIII](book/src/part-08-benchmarks.md).

## Python

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
det = detector.detect(img)
print(det.xy.shape)      # (N, 2) float32
print(det.response.shape) # (N,)  float32
```

`detect()` returns a `Detections` object with named arrays:

- `det.xy` — `(N, 2)` float32, subpixel corner positions (x, y) in input pixels
- `det.response` — `(N,)` float32, raw detector response at each peak
- `det.angles` — `(N, 2)` float32, `[axis0_angle, axis1_angle]` in radians `[0, π)`, or `None` when orientation is disabled
- `det.sigmas` — `(N, 2)` float32, 1σ uncertainty per axis in radians, or `None` when orientation is disabled

With `cfg.without_orientation()`, `det.angles` and `det.sigmas` are `None`.

## JavaScript / WebAssembly

```bash
wasm-pack build crates/chess-corners-wasm --target web
```

```js
import init, { ChessDetector, DetectorConfig } from '@vitavision/chess-corners';

await init();

const cfg = DetectorConfig.chessMultiscale();
const detector = ChessDetector.withConfig(cfg);
const imageData = ctx.getImageData(0, 0, width, height);
const corners = detector.detect_rgba(imageData.data, width, height);
```

See [`crates/chess-corners-wasm/README.md`](crates/chess-corners-wasm/README.md)
for typed configuration and browser examples.

## CLI

```bash
cargo run -p chess-corners --release --bin chess-corners -- \
    run config/chess_cli_config_example.json
```

The CLI loads an image and JSON config, runs the detector, then writes
a JSON summary and optional overlay PNG. Useful config examples:

- [`config/chess_algorithm_config_example.json`](config/chess_algorithm_config_example.json)
  — pure `DetectorConfig`, shared by Rust and Python.
- [`config/chess_cli_config_example.json`](config/chess_cli_config_example.json)
  — `DetectorConfig` plus CLI input/output fields.
- [`config/chess_cli_config_example_ml.json`](config/chess_cli_config_example_ml.json)
  — ML refiner example; requires `--features ml-refiner`.

## Feature Flags

| Feature | Effect |
|---------|--------|
| `image` | `image::GrayImage` entry points |
| `rayon` | Parallel response/refinement work |
| `simd` | Optional high-performance path: `std::simd` ChESS kernels; nightly only |
| `par_pyramid` | SIMD/Rayon paths inside `box-image-pyramid` |
| `tracing` | Structured diagnostic spans |
| `ml-refiner` | ONNX refiner through `chess-corners-ml` |
| `cli` | Builds the `chess-corners` binary |
| `radon-sat-u32` | Uses `u32` Radon SATs for lower memory with an input-size cap |

Feature flags should affect performance or observability, not the
numerical output. Deterministic ordering is part of the public contract.
The stable scalar/autovectorized build is the supported, portable
baseline — correct on every target and fast enough for typical use; it
needs Rust 1.88 or newer. Only `simd` requires a nightly toolchain.

## Diligence Statement

This project uses AI coding assistants as implementation tools. The
author validates algorithmic behavior and numerical results before
release through formatting, linting, tests, documentation builds, the
book, and binding checks.

## References

- ChESS detector — Bennett, S. & Lasenby, J. (2014). *ChESS — Quick and
  Robust Detection of Chess-board Features*. Computer Vision and Image
  Understanding, 118:197–210.
  [arXiv:1301.5491](https://arxiv.org/abs/1301.5491).
- Radon detector — Duda, A. & Frese, U. (2018). *Accurate Detection and
  Localization of Checkerboard Corners for Calibration*. BMVC.

Further reading on the vitavision.dev atlas:
[chess-corners](https://vitavision.dev/atlas/chess-corners) ·
[duda-radon-corners](https://vitavision.dev/atlas/duda-radon-corners).

## Pre-PR Quality Gates

```bash
python3 tools/check_doc_versions.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
mdbook build book
```
