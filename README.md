# chess-corners-rs

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)

Fast Rust implementation of the [**ChESS**](https://arxiv.org/abs/1301.5491)
(Chess-board Extraction by Subtraction and Summation) corner detector.

![](book/src/img/mid_chess.png)

([image source](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures))

ChESS is a classical (not ML) **feature detector** for chessboard
**X-junction** corners. This project focuses on **convenient use**,
**subpixel accuracy**, and **real-time performance** (scalar, `rayon`,
and portable SIMD paths), with an optional ML-backed subpixel refiner
for higher precision on synthetic data.

## Quick start

```rust
use chess_corners::{ChessConfig, find_chess_corners_image};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let cfg = ChessConfig::single_scale();

let corners = find_chess_corners_image(&img, &cfg);
println!("found {} corners", corners.len());
if let Some(c) = corners.first() {
    println!(
        "corner at ({:.2}, {:.2}), response {:.1}, theta {:.2} rad",
        c.x, c.y, c.response, c.orientation
    );
}
```

## Configuring `ChessConfig` (Rust API)

`find_chess_corners_image` (enabled by the default `image` feature) is configured via `ChessConfig`, a small struct that groups:

- `cfg.params: ChessParams` — ChESS response + detector knobs (ring radius, thresholding, NMS/clustering, and the subpixel refiner).
- `cfg.multiscale: CoarseToFineParams` — coarse-to-fine pyramid settings (levels/min size) plus ROI/merge tuning for multiscale refinement.

Single-scale vs multiscale is controlled by `cfg.multiscale.pyramid.num_levels`:

- `<= 1`: single-scale detection (same as `ChessConfig::single_scale()`).
- `> 1`: coarse-to-fine detection (the default `ChessConfig::default()` uses 3 levels).

Common knobs you may want to adjust:

- Sensitivity: `cfg.params.threshold_rel` (recommended) or `cfg.params.threshold_abs` (absolute override).
- Blur / very small boards: `cfg.params.use_radius10 = true` (uses the r=10 ring instead of r=5).
- Subpixel refinement: `cfg.params.refiner = RefinerKind::Forstner(ForstnerConfig::default())` (alternatives: `CenterOfMass`, `SaddlePoint`).
- Multiscale trade-offs: `cfg.multiscale.pyramid.min_size`, `cfg.multiscale.refinement_radius`, `cfg.multiscale.merge_radius`.

Example:

```rust
use chess_corners::{ChessConfig, ForstnerConfig, RefinerKind, find_chess_corners_image};
use image::ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();

let mut cfg = ChessConfig::default(); // default: 3-level multiscale
cfg.params.threshold_rel = 0.15;
cfg.params.refiner = RefinerKind::Forstner(ForstnerConfig::default());

// Force single-scale:
// cfg.multiscale.pyramid.num_levels = 1;

let corners = find_chess_corners_image(&img, &cfg);
println!("found {} corners", corners.len());
```

## Performance snapshot

Measured on a MacBook Pro M4 (release build, averaged over 10 runs) using the
recommended 3-level multiscale pipeline:

| Features      | 720×540 | 1200×900 | 2048×1536 |
|---------------|--------:|---------:|----------:|
| none          | 0.6 ms | 0.7 ms  | 4.9 ms   |
| `simd`        | 0.4 ms | 0.4 ms  | 2.8 ms   |
| `rayon`       | 0.5 ms | 0.5 ms  | 1.9 ms   |
| `simd+rayon`  | 0.5 ms | 0.5 ms  | 1.6 ms   |

(`simd` uses portable SIMD and currently requires a nightly Rust toolchain.)

On a public stereo chessboard dataset, the mean nearest-neighbor distance to
OpenCV’s `findChessboardCornersSB` corners is below ≈ **0.2 px** (see the book for
methodology and plots).

See [`book/src/part-05-performance-and-integration.md`](book/src/part-05-performance-and-integration.md) for the full breakdown,
OpenCV comparisons, and how to reproduce the numbers with [`tools/perf_bench.py`](tools/perf_bench.py).

## ML refiner (optional)

The ML refiner is an ONNX-backed subpixel refinement stage. It runs on
small intensity patches centered at each candidate corner and predicts
`dx`, `dy` (the confidence output is ignored in the current version).
The input patches are normalized to `[0, 1]` by dividing `uint8` values
by `255.0`, and the model outputs `[dx, dy, conf_logit]`.

Use it in Rust by enabling the `ml-refiner` feature and calling the
explicit ML entry points:

```rust
use chess_corners::{ChessConfig, find_chess_corners_image_with_ml};

let cfg = ChessConfig::default();
let corners = find_chess_corners_image_with_ml(&img, &cfg);
```

ML refinement is **not** claimed to work well in all realistic cases
yet; it has been evaluated primarily on synthetic data. On that data,
it yields much tighter errors than OpenCV `cornerSubPix` (see the book
for plots). It is also slower: on `testimages/mid.png` (77 corners),
ML refinement takes about **23.5 ms** vs **0.6 ms** for the classic
refiner.

The current version applies all predicted offsets directly and does
not threshold by the model’s confidence output.

The published documentation includes:

- a guide-style book (API overview, internals, multiscale tuning), and
- generated Rust API docs for both `chess-corners-core` and `chess-corners`.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and pluggable subpixel refinement (center-of-mass, Förstner, saddle-point).
- Optional ML-backed refiner (`ml-refiner` feature) for higher synthetic accuracy.
- Optional `rayon` parallelism and portable SIMD acceleration (requires nightly Rust toolchain) on the dense response path.
- Crates: `chess-corners-core` (lean core) and `chess-corners` (ergonomic facade with optional `image`/multiscale integration and a CLI binary target).
- Multiscale coarse-to-fine pipeline with reusable pyramid buffers (fast + robust under blur/scale changes).
- Corner descriptors that include subpixel position, response, and orientation.
- JSON/PNG output plus Python helpers under `tools/` (synthetic dataset generator, accuracy benchmark, perf tooling).
- Python bindings (`chess_corners` module) built with PyO3 + maturin.

## Installation

Add the high-level facade crate to your `Cargo.toml`:

```toml
[dependencies]
chess-corners = "0.3.1"
```

If you need direct access to the low-level response / detector stages, you can also depend on the core crate:

```toml
[dependencies]
chess-corners-core = "0.3.1"
```

The `chess-corners` crate enables the `image` feature by default so you can work with `image::GrayImage`; disable it if you prefer to stay on raw buffers.

### Python bindings

Install the Python package (published as `chess_corners` and imported as `chess_corners`):

```bash
python -m pip install chess-corners
```

Minimal usage:

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)
corners = chess_corners.find_chess_corners(img)
print(corners.shape, corners.dtype)  # (N, 4), float32
```

The returned array columns are `[x, y, response, orientation]` in image pixels.
For configuration, construct `chess_corners.ChessConfig()` and set its fields
(`threshold_rel`, `nms_radius`, `pyramid_num_levels`, etc.).

### Examples

The `chess-corners` crate ships small examples that operate directly on `image::GrayImage` inputs and use the sample images under `testimages/`:

- Single-scale: `cargo run -p chess-corners --example single_scale_image -- testimages/mid.png`
- Multiscale: `cargo run -p chess-corners --example multiscale_image -- testimages/large.png`

These examples rely on the optional `image` feature on `chess-corners` (enabled by default). If you build with `--no-default-features`, pass `--features image` when running them.

## Development

- Run the workspace tests: `cargo test`
- Enable parallel response computation: `cargo test -p chess-corners-core --features rayon`
- Run docs locally: `cargo doc --workspace --all-features --no-deps`

### CLI

Run the bundled CLI for quick experiments:

```
cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config_example.json
```

To use the ML refiner from the CLI, enable the feature and set `ml: true` in the config:

```
cargo run -p chess-corners --release --features ml-refiner --bin chess-corners -- run config/chess_cli_config_example_ml.json
```

The config JSON drives both single-scale and multiscale runs:

```json
{
  "image": "testimages/mid.png",
  "pyramid_levels": 3,
  "min_size": 64,
  "refinement_radius": 3,
  "merge_radius": 3.0,
  "output_json": null,
  "output_png": null,
  "threshold_rel": 0.2,
  "threshold_abs": null,
  "refiner": "center_of_mass",
  "ml": true,
  "radius10": false,
  "descriptor_radius10": null,
  "nms_radius": 2,
  "min_cluster_size": 2,
  "log_level": "info"
}
```

- **Multiscale control**
  - `pyramid_levels`: number of pyramid levels (`<= 1` behaves as single-scale; larger values enable coarse-to-fine refinement)
  - `min_size`: smallest image size allowed in the pyramid (limits how deep the pyramid goes)
  - `refinement_radius`, `merge_radius`: multiscale refinement and deduplication
- **Detection and refinement**
  - `threshold_rel` / `threshold_abs`: response thresholding (relative thresholding is recommended in most cases)
  - `refiner`: subpixel refinement method (`center_of_mass`, `forstner`, `saddle_point`)
  - `ml`: set `true` to enable the ML refiner pipeline
  - `radius10`: use the larger r=10 ring instead of the canonical r=5 ring
  - `descriptor_radius10`: optional r=10 override for descriptors (when null, follows `radius10`)
  - `nms_radius`, `min_cluster_size`: suppression and clustering
- **Output**
  - `output_json` / `output_png`: override output paths (defaults next to the image)

You can override many fields via CLI flags (e.g., `--levels 1 --min_size 64 --output_json out.json`).

### Config cheat sheet

The defaults are tuned to be stable on real calibration images. In most cases you
only need to touch a few knobs:

- **Usually tune**
  - `pyramid_levels`: `1` = single-scale, `2–4` = multiscale (3 is a good default).
  - `min_size`: controls how deep the pyramid goes; smaller values allow detecting smaller boards but cost more.
  - `threshold_rel`: main sensitivity knob (lower = more corners, higher = fewer/stronger corners).
  - `refiner`: `forstner` / `saddle_point` can improve localization vs `center_of_mass` (trade-offs depend on blur/noise).
  - `radius10`: set `true` for heavy blur / very small boards; otherwise keep `false`.
  - `refinement_radius` + `merge_radius` (multiscale): increase `refinement_radius` if refinement misses corners; adjust `merge_radius` (typically `2–3`) if you see duplicates.

  - **Usually don’t touch**
  - `threshold_abs`: keep `null` unless you have a controlled pipeline and want a fixed absolute threshold (relative is more exposure/contrast robust).
  - `descriptor_radius10`: keep `null` (or omit) so descriptors follow the detector ring radius (avoids accidental mismatches).
  - `nms_radius` / `min_cluster_size`: leave defaults unless you’re deliberately trading recall vs noise; these are easy to overtune.

- SIMD and `rayon` are gated by Cargo features:
  - Enable SIMD (nightly only) on the core: `cargo test -p chess-corners-core --features simd`
  - Enable both SIMD and `rayon`: `cargo test -p chess-corners-core --features "simd,rayon"`
  - Add `par_pyramid` on `chess-corners` to accelerate pyramid downsampling when using those features.

- Tracing: enable structured spans for profiling by turning on the `tracing`
  feature in the libraries, and use env filters with the CLI:
  - `cargo test -p chess-corners-core --features tracing`
  - `RUST_LOG=info cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config_example.json`
  - `--json-trace` switches the CLI to emit JSON-formatted spans.

## Status

Stable, ready to use, published on [`crates.io`](https://crates.io/crates/chess-corners). Public API still may change, mostly by changing parameters set. User feedback is very welcome (create an issue or write me).

For contribution rules see [CONTRIBUTING.md](./CONTRIBUTING.md) and [AGENTS.md](./AGENTS.md).

## License

Licensed under the MIT license (see `Cargo.toml`).

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
