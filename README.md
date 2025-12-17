# chess-corners-rs

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)

Fast, deterministic Rust implementation of the [**ChESS**](https://arxiv.org/abs/1301.5491)
(Chess-board Extraction by Subtraction and Summation) corner detector.

![](book/src/img/mid_chess.png)

([image source](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures))

ChESS is a classical, ID-free **feature detector** for chessboard / checkerboard
**X-junction** corners (including ChArUco-style boards). This workspace focuses
on **deterministic outputs**, **subpixel accuracy**, and **real-time
performance** (scalar, `rayon`, and portable SIMD paths).

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
OpenCV’s `findChessboardCornersSB` corners is ≈ **0.21 px** (see the book for
methodology and plots).

See [`book/src/part-05-performance-and-integration.md`](book/src/part-05-performance-and-integration.md) for the full breakdown,
OpenCV comparisons, and how to reproduce the numbers with [`tools/perf_bench.py`](tools/perf_bench.py).

The published documentation includes:

- a guide-style book (API overview, internals, multiscale tuning), and
- generated Rust API docs for both `chess-corners-core` and `chess-corners`.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and pluggable subpixel refinement (center-of-mass, Förstner, saddle-point).
- Optional `rayon` parallelism and portable SIMD acceleration (requires nightly Rust toolchain) on the dense response path.
- Crates: `chess-corners-core` (lean core) and `chess-corners` (ergonomic facade with optional `image`/multiscale integration and a CLI binary target).
- Multiscale coarse-to-fine pipeline with reusable pyramid buffers (fast + robust under blur/scale changes).
- Corner descriptors that include subpixel position, response, and orientation.
- JSON/PNG output plus Python helpers under `tools/` (synthetic dataset generator, accuracy benchmark, perf tooling).

## Installation

Add the high-level facade crate to your `Cargo.toml`:

```toml
[dependencies]
chess-corners = "0.2"
```

If you need direct access to the low-level response / detector stages, you can also depend on the core crate:

```toml
[dependencies]
chess-corners-core = "0.2"
```

The `chess-corners` crate enables the `image` feature by default so you can work with `image::GrayImage`; disable it if you prefer to stay on raw buffers.

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

Need timings for profiling? Enable the `tracing` feature.

The multiscale path uses a coarse detector on the smallest pyramid level and
refines each seed in a base-image ROI. The ROI radius is specified in
coarse-level pixels and is automatically converted to a radius in base pixels,
with a minimum margin derived from the ChESS detector’s own border logic. Both
full-frame and ROI response computations honor the `rayon`/`simd` features so
patch refinement benefits from the same SIMD and parallelism as the dense
response path. Pyramid downsampling stays scalar unless the `par_pyramid`
feature is enabled alongside `simd` and/or `rayon`.

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
cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config.example.json
```

The config JSON drives both single-scale and multiscale runs:

```json
{
  "image": "testimages/mid.png",
  "pyramid_levels": 3,
  "min_size": 64,
  "roi_radius": 3,
  "merge_radius": 3.0,
  "output_json": null,
  "output_png": null,
  "threshold_rel": 0.2,
  "threshold_abs": null,
  "refiner": "forstner",
  "radius": 5,
  "descriptor_radius": null,
  "nms_radius": 2,
  "min_cluster_size": 2,
  "log_level": "info"
}
```

- `pyramid_levels`, `min_size`, `roi_radius`, `merge_radius`: multiscale controls (`pyramid_levels <= 1` behaves as single-scale; larger values request a multiscale coarse-to-fine run, with `min_size` limiting how deep the pyramid goes)
- `threshold_rel` / `threshold_abs`, `refiner`, `radius` / `descriptor_radius`, `nms_radius`, `min_cluster_size`: detector + descriptor tuning (`refiner` accepts `center_of_mass`, `forstner`, or `saddle_point`; `descriptor_radius` falls back to `radius` when null)
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
  - `radius`: use `10` for heavy blur / very small boards; otherwise `5`.
  - `roi_radius` + `merge_radius` (multiscale): increase `roi_radius` if refinement misses corners; adjust `merge_radius` (typically `2–3`) if you see duplicates.

- **Usually don’t touch**
  - `threshold_abs`: keep `null` unless you have a controlled pipeline and want a fixed absolute threshold (relative is more exposure/contrast robust).
  - `descriptor_radius`: keep `null` so descriptors use the same ring radius as detection (avoids accidental mismatches).
  - `nms_radius` / `min_cluster_size`: leave defaults unless you’re deliberately trading recall vs noise; these are easy to overtune.

- SIMD and `rayon` are gated by Cargo features:
  - Enable SIMD (nightly only) on the core: `cargo test -p chess-corners-core --features simd`
  - Enable both SIMD and `rayon`: `cargo test -p chess-corners-core --features "simd,rayon"`
  - Add `par_pyramid` on `chess-corners` to accelerate pyramid downsampling when using those features.

- Tracing: enable structured spans for profiling by turning on the `tracing`
  feature in the libraries, and use env filters with the CLI:
  - `cargo test -p chess-corners-core --features tracing`
  - `RUST_LOG=info cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config.example.json`
  - `--json-trace` switches the CLI to emit JSON-formatted spans.

## Status

Implemented:
- response kernel, ring tables, NMS + thresholding + cluster filter, 5x5 subpixel refinement, image helpers, data-free unit tests, tracing instrumentation
- multiscale pyramid builder with reusable buffers and coarse-to-fine corner refinement path
- SIMD acceleration and optional `rayon` parallelism on the response path; pyramid downsampling can opt into SIMD/parallelism via `par_pyramid`
- CLI tooling and plotting helper for JSON/PNG-based inspection

For contribution rules see [AGENTS.md](./AGENTS.md).

## License

Licensed under the MIT license (see `Cargo.toml`).

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
