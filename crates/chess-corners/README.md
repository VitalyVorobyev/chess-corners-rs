# chess-corners

Ergonomic ChESS (Chess-board Extraction by Subtraction and Summation) detector on top of
`chess-corners-core`.

This crate:

- Re-exports the main types from `chess-corners-core` (`ChessParams`, `CornerDescriptor`, `ResponseMap`).
- Provides a unified `ChessConfig` for single-scale and multiscale detection.
- Adds optional `image::GrayImage` integration and a small CLI binary for batch runs.

By default the `image` feature is enabled so you can work directly with `GrayImage`:

```rust
use chess_corners::{ChessConfig, ChessParams, find_chess_corners_image};
use image::io::Reader as ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?
        .decode()?
        .to_luma8();

    let mut cfg = ChessConfig::single_scale();
    cfg.params = ChessParams::default();

    let corners = find_chess_corners_image(&img, &cfg);
    println!("found {} corners", corners.len());
    Ok(())
}
```

Feature flags:

- `image` *(default)* – enable `find_chess_corners_image` for `image::GrayImage`.
- `rayon` – parallelize response computation and multiscale refinement.
- `simd` – enable portable-SIMD acceleration in the core response kernel (nightly only).
- `par_pyramid` – opt into SIMD/`rayon` in the pyramid builder.
- `tracing` – emit structured spans from multiscale detection and the CLI when enabled.

The full guide-style documentation and API docs are published at:

- Book: https://vitalyvorobyev.github.io/chess-rs/part-01-orientation.html
- Rust docs: https://vitalyvorobyev.github.io/chess-rs/api/

