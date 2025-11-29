# chess-rs

[![CI](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-rs/)

Rust implementation of the **ChESS** (Chess-board Extraction by Subtraction and Summation) corner detector.

ChESS is a classical, ID-free detector for chessboard **X-junction** corners. This workspace delivers a fast scalar kernel, corner extraction with non-maximum suppression and subpixel refinement, and convenient helpers for the `image` crate.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and 5x5 center-of-mass refinement.
- Optional `rayon` parallelism and portable SIMD acceleration on the dense response path and pyramid downsampling.
- Three crates:
  - `chess-core`: lean core (std optional) meant to stay SIMD/parallel-friendly.
  - `chess`: ergonomic facade that accepts `image::GrayImage`.
  - `chess-cli`: small CLI for single-scale and multiscale runs.
- Multiscale coarse-to-fine helpers with reusable pyramid buffers.
- JSON/PNG output and a small Python helper (`tools/plot_corners.py`) for overlay visualization.

## Quick start

```rust
use chess::{chess_response_image, find_corners_image, ChessParams};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let params = ChessParams::default();

let resp = chess_response_image(&img, &params);
println!("response map: {} x {}", resp.w, resp.h);

let corners = find_corners_image(&img, &params);
println!("found {} corners", corners.len());
```

Need timings for profiling? Swap in `find_corners_image_trace` to get per-stage milliseconds.

### Multiscale (coarse-to-fine)

```rust
use chess::{find_corners_coarse_to_fine_image_trace, ChessParams, CoarseToFineParams, PyramidBuffers};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let params = ChessParams::default();
let cf = CoarseToFineParams::default();
let mut buffers = PyramidBuffers::new();
buffers.prepare_for_image(&img, &cf.pyramid);

let res = find_corners_coarse_to_fine_image_trace(&img, &params, &cf, &mut buffers);
println!("coarse stage ran in {:.2} ms, refined {} corners", res.coarse_ms, res.corners.len());
```

The multiscale path uses a coarse detector on the smallest pyramid level and
refines each seed in a base-image ROI. The ROI radius is specified in
coarse-level pixels and is automatically converted to a radius in base pixels,
with a minimum margin derived from the ChESS detector’s own border logic. Both
full-frame and ROI response computations honor the `rayon`/`simd` features so
patch refinement benefits from the same SIMD and parallelism as the dense
response path.

## Development

- Run the workspace tests: `cargo test`
- Enable parallel response computation: `cargo test -p chess-core --features rayon`
- Run docs locally: `cargo doc --workspace --all-features --no-deps`

### CLI

Run the bundled CLI for quick experiments:

```
cargo run -p chess-cli -- run config/chess_cli_config.example.json
```

The config JSON drives both single-scale and multiscale runs:

```json
{
  "image": "testdata/images/Cam1.png",
  "mode": "multiscale",
  "downsample": null,
  "pyramid_levels": 3,
  "min_size": 12,
  "roi_radius": 12,
  "merge_radius": 2.0,
  "output_json": null,
  "output_png": null,
  "threshold_rel": 0.015,
  "threshold_abs": null,
  "radius": 5,
  "nms_radius": 1,
  "min_cluster_size": 2,
  "log_level": "info"
}
```

- `mode`: `single` or `multiscale`
- `downsample`: integer factor (single-scale only)
- `pyramid_levels`, `min_size`, `roi_radius`, `merge_radius`: multiscale controls
- `threshold_rel` / `threshold_abs`, `radius`, `nms_radius`, `min_cluster_size`: detector tuning
- `output_json` / `output_png`: override output paths (defaults next to the image)

You can override any field via CLI flags (e.g., `--mode single --downsample 2 --output_json out.json`).

- SIMD and `rayon` are gated by Cargo features:
  - Enable SIMD (nightly only) on the core: `cargo test -p chess-core --features simd`
  - Enable both SIMD and `rayon`: `cargo test -p chess-core --features "simd,rayon"`

## Status

Implemented:
- response kernel, ring tables, NMS + thresholding + cluster filter, 5x5 subpixel refinement, image helpers, data-free unit tests
- multiscale pyramid builder with reusable buffers and coarse-to-fine corner refinement path
- SIMD acceleration and optional `rayon` parallelism on the response and pyramid paths
- CLI tooling and plotting helper for JSON/PNG-based inspection

## License

Licensed under MIT.

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
