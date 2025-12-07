# chess-corners-rs

[![CI](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-corners-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-corners-rs/)

Rust implementation of the [**ChESS**](https://arxiv.org/abs/1301.5491) (Chess-board Extraction by Subtraction and Summation) corner detector.

![](book/src/img/mid_chess.png)

([image source](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures))

ChESS is a classical, ID-free detector for chessboard **X-junction** corners. This workspace delivers a fast scalar kernel, corner extraction with non-maximum suppression and subpixel refinement, and convenient helpers for the `image` crate.

The published documentation includes:

- a guide-style book (API overview, internals, multiscale tuning), and
- generated Rust API docs for both `chess-corners-core` and `chess-corners`.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and 5x5 center-of-mass refinement.
- Optional `rayon` parallelism and portable SIMD acceleration (requires nightly RUST channel) on the dense response path.
- Crates: `chess-corners-core` (lean core) and `chess-corners` (ergonomic facade with optional `image`/multiscale integration and a CLI binary target).
- Multiscale coarse-to-fine helpers with reusable pyramid buffers.
- Corner descriptors that include subpixel position, scale, response, orientation, phase, and anisotropy.
- JSON/PNG output and Python helpers under `tools/` for benchmarking and visualization.

## Installation

Add the high-level facade crate to your `Cargo.toml`:

```toml
[dependencies]
chess-corners = "0.1"
```

If you need direct access to the low-level response / detector stages, you can also depend on the core crate:

```toml
[dependencies]
chess-corners-core = "0.1"
```

The `chess-corners` crate enables the `image` feature by default so you can work with `image::GrayImage`; disable it if you prefer to stay on raw buffers.

## Quick start

```rust
use chess_corners::{ChessConfig, find_chess_corners_image};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let cfg = ChessConfig::single_scale();

let corners = find_chess_corners_image(&img, &cfg);
println!("found {} corners", corners.len());
if let Some(c) = corners.first() {
    println!(
        "corner at ({:.2}, {:.2}), response {:.1}, theta {:.2} rad, phase {}",
        c.x, c.y, c.response, c.orientation, c.phase
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
  "roi_radius": 12,
  "merge_radius": 3.0,
  "output_json": null,
  "output_png": null,
  "threshold_rel": 0.2,
  "threshold_abs": null,
  "radius": 5,
  "descriptor_radius": null,
  "nms_radius": 2,
  "min_cluster_size": 2,
  "log_level": "info"
}
```

- `pyramid_levels`, `min_size`, `roi_radius`, `merge_radius`: multiscale controls (`pyramid_levels <= 1` behaves as single-scale; larger values request a multiscale coarse-to-fine run, with `min_size` limiting how deep the pyramid goes)
- `threshold_rel` / `threshold_abs`, `radius` / `descriptor_radius`, `nms_radius`, `min_cluster_size`: detector + descriptor tuning (`descriptor_radius` falls back to `radius` when null)
- `output_json` / `output_png`: override output paths (defaults next to the image)

You can override many fields via CLI flags (e.g., `--levels 1 --min_size 64 --output_json out.json`).

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

## License

Licensed under the MIT license (see `Cargo.toml`).

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
