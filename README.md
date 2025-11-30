# chess-corners-rs

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
- `chess-corners-core`: lean core (std optional) meant to stay SIMD/parallel-friendly.
- `chess-corners`: ergonomic facade (optionally with `image`/`multiscale` features). Internally uses a minimal u8 image buffer for pyramids; `image` is only pulled in when the feature is enabled.
- `chess-corners` binary: CLI for single-scale and multiscale runs.
- Multiscale coarse-to-fine helpers with reusable pyramid buffers.
- Corner descriptors that include subpixel position, scale, response,
  orientation, phase, and anisotropy.
- JSON/PNG output and a small Python helper (`tools/plot_corners.py`) for overlay visualization.

## Quick start

```rust
use chess_corners::{chess_response_image, find_corners_image, ChessParams};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let params = ChessParams::default();

let resp = chess_response_image(&img, &params);
println!("response map: {} x {}", resp.w, resp.h);

let corners = find_corners_image(&img, &params);
println!("found {} corners", corners.len());
if let Some(c) = corners.first() {
    println!(
        "corner at ({:.2}, {:.2}), response {:.1}, theta {:.2} rad, phase {}",
        c.x, c.y, c.response, c.orientation, c.phase
    );
}
```

The `image` and `multiscale` features on `chess-corners` are enabled by default; disable them if you only need the low-level `chess-corners-core` API.

Need timings for profiling? Swap in `find_corners_image_trace` to get per-stage milliseconds.

### Multiscale (coarse-to-fine)

```rust
use chess_corners::{
    find_corners_coarse_to_fine_image, ChessParams, CoarseToFineParams, PyramidBuffers,
};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let params = ChessParams::default();
let cf = CoarseToFineParams::default();
let mut buffers = PyramidBuffers::new();
buffers.prepare_for_image(&img, &cf.pyramid);

let res = find_corners_coarse_to_fine_image(&img, &params, &cf, &mut buffers);
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
  "descriptor_radius": null,
  "nms_radius": 1,
  "min_cluster_size": 2,
  "log_level": "info"
}
```

- `mode`: `single` or `multiscale`
- `downsample`: integer factor (single-scale only)
- `pyramid_levels`, `min_size`, `roi_radius`, `merge_radius`: multiscale controls
- `threshold_rel` / `threshold_abs`, `radius`, `descriptor_radius`, `nms_radius`, `min_cluster_size`: detector + descriptor tuning (`descriptor_radius` falls back to `radius` when null)
- `output_json` / `output_png`: override output paths (defaults next to the image)

You can override any field via CLI flags (e.g., `--mode single --downsample 2 --output_json out.json`).

- SIMD and `rayon` are gated by Cargo features:
  - Enable SIMD (nightly only) on the core: `cargo test -p chess-corners-core --features simd`
  - Enable both SIMD and `rayon`: `cargo test -p chess-corners-core --features "simd,rayon"`

- Tracing: enable structured spans for profiling by turning on the `tracing`
  feature in the libraries, and use env filters with the CLI:
  - `cargo test -p chess-corners-core --features tracing`
  - `RUST_LOG=info cargo run -p chess-corners --release --bin chess-corners -- run config/chess_cli_config.example.json`
  - `--json-trace` switches the CLI to emit JSON-formatted spans.

## Status

Implemented:
- response kernel, ring tables, NMS + thresholding + cluster filter, 5x5 subpixel refinement, image helpers, data-free unit tests, tracing instrumentation
- multiscale pyramid builder with reusable buffers and coarse-to-fine corner refinement path
- SIMD acceleration and optional `rayon` parallelism on the response and pyramid paths
- CLI tooling and plotting helper for JSON/PNG-based inspection

## License

Dual-licensed under MIT or Apache-2.0.

## References

- Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
