# chess-rs

[![CI](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/audit.yml)
[![Docs](https://github.com/VitalyVorobyev/chess-rs/actions/workflows/docs.yml/badge.svg)](https://vitalyvorobyev.github.io/chess-rs/)

Rust implementation of the **ChESS** (Chess-board Extraction by Subtraction and Summation) corner detector.

ChESS is a classical, ID-free detector for chessboard **X-junction** corners. This workspace delivers a fast scalar kernel, corner extraction with non-maximum suppression and subpixel refinement, and convenient helpers for the `image` crate.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and 5x5 center-of-mass refinement.
- Optional `rayon` parallelism on the response path and SIMD downsampling for pyramids.
- Two crates:
  - `chess-core`: lean core (std optional) meant to stay SIMD/parallel-friendly.
  - `chess`: ergonomic facade that accepts `image::GrayImage`.
- Multiscale coarse-to-fine helpers with reusable pyramid buffers.
- Example utilities that export detections as JSON/PNG overlays.

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
- Example run with JSON + overlay output:
  - `cargo run -p chess --example dump_corners --release -- path/to/board.png`
- Enable parallel response computation: `cargo test -p chess-core --features rayon`

## Status

Implemented: response kernel, ring tables, NMS + thresholding + cluster filter, 5x5 subpixel refinement, image helpers, data-free unit tests.

Implemented (multiscale): pyramid builder with reusable buffers and coarse-to-fine corner refinement path.

Planned: SIMD acceleration, CLI packaging.

## License

Dual-licensed under MIT or Apache-2.0.

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
