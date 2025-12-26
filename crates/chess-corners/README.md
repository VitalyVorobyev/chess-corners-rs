# chess-corners

Ergonomic ChESS (Chess-board Extraction by Subtraction and Summation) detector on top of
`chess-corners-core`.

This crate:

- Re-exports the main types from `chess-corners-core` (`ChessParams`, `CornerDescriptor`, `ResponseMap`).
- Provides a unified `ChessConfig` for single-scale and multiscale detection.
- Adds optional `image::GrayImage` integration and a small CLI binary for batch runs.
- Exposes pluggable subpixel refiners (`RefinerKind` via `ChessParams::refiner`) so you can choose
  between center-of-mass (default), Förstner, or saddle-point refinement.

## Examples

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

### Selecting a refiner

The default refiner matches the legacy center-of-mass behavior. To opt into the
Förstner or saddle-point refiners on image intensities:

```rust
use chess_corners::{ChessConfig, ChessParams, RefinerKind, find_chess_corners_image};

let mut cfg = ChessConfig::single_scale();
cfg.params = ChessParams::default();

let refiner = RefinerKind::Forstner(Default::default());
cfg.params.refiner = refiner;
let corners = find_chess_corners_image(&img, &cfg);
```

You can also override the refiner per call without mutating your config via
`find_chess_corners_image_with_refiner`.

You can also try the bundled examples on sample images in `testimages/`:

- Single-scale: `cargo run -p chess-corners --example single_scale_image -- testimages/mid.png`
- Multiscale: `cargo run -p chess-corners --example multiscale_image -- testimages/large.png`

Both examples require the `image` feature, which is enabled by default. If you build with `--no-default-features`, re-enable it when running examples: `--features image`.

Feature flags:

- `image` *(default)* – enable `find_chess_corners_image` for `image::GrayImage`.
- `rayon` – parallelize response computation and multiscale refinement.
- `simd` – enable portable-SIMD acceleration in the core response kernel (nightly only).
- `par_pyramid` – opt into SIMD/`rayon` in the pyramid builder.
- `tracing` – emit structured spans from multiscale detection and the CLI when enabled.

The full guide-style documentation and API docs are published at:

- Book: https://vitalyvorobyev.github.io/chess-corners-rs
- Rust docs: https://vitalyvorobyev.github.io/chess-corners-rs/api/

Python bindings are available in this workspace under `crates/chess-corners-py`
and are published as the `chess_corners` package on PyPI.
