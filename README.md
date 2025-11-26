# chess-rs

Rust implementation of the **ChESS** (Chess-board Extraction by Subtraction and Summation) corner detector.

ChESS is a classical, ID-free detector for chessboard **X-junction** corners. This workspace delivers a fast scalar kernel, corner extraction with non-maximum suppression and subpixel refinement, and convenient helpers for the `image` crate.

## Highlights
- Canonical 16-sample rings (r=5 default, r=10 for heavy blur).
- Dense response computation plus NMS, minimum-cluster filtering, and 5x5 center-of-mass refinement.
- Optional `rayon` parallelism on the response path.
- Two crates:
  - `chess-core`: lean core (std optional) meant to stay SIMD/parallel-friendly.
  - `chess`: ergonomic facade that accepts `image::GrayImage`.
- Example utility `dump_corners` that exports detections as JSON/PNG overlays.

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

## Development

- Run the workspace tests: `cargo test`
- Example run with JSON + overlay output:
  - `cargo run -p chess --example dump_corners --release -- path/to/board.png`
- Enable parallel response computation: `cargo test -p chess-core --features rayon`

## Status

Implemented: response kernel, ring tables, NMS + thresholding + cluster filter, 5x5 subpixel refinement, image helpers, data-free unit tests.

Planned: SIMD acceleration, multi-scale pyramid, CLI packaging.

## License

Dual-licensed under MIT or Apache-2.0.

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
