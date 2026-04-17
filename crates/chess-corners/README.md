# chess-corners

Ergonomic ChESS (Chess-board Extraction by Subtraction and Summation) detector
on top of `chess-corners-core`.

This crate is the public Rust API:

- flat `ChessConfig` with explicit semantic modes, threshold mode, and refiner selection
- single-scale and multiscale detection entry points
- optional `image::GrayImage` helpers
- optional CLI binary and ML-backed refinement entry points

`chess-corners-core` and `box-image-pyramid` remain available as lower-level
sharp tools, but `chess-corners` is the intended compatibility boundary.

## Quick start

```rust
use chess_corners::{ChessConfig, RefinementMethod, find_chess_corners_image};
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    let mut cfg = ChessConfig::multiscale();
    cfg.threshold_value = 0.15;
    cfg.refiner.kind = RefinementMethod::Forstner;

    let corners = find_chess_corners_image(&img, &cfg);
    println!("found {} corners", corners.len());
    Ok(())
}
```

## Public config shape

`ChessConfig` is intentionally flat:

```rust
use chess_corners::{
    ChessConfig, DescriptorMode, DetectorMode, RefinementMethod, ThresholdMode,
};

let mut cfg = ChessConfig::single_scale();
cfg.detector_mode = DetectorMode::Canonical;
cfg.descriptor_mode = DescriptorMode::FollowDetector;
cfg.threshold_mode = ThresholdMode::Relative;
cfg.threshold_value = 0.2;
cfg.nms_radius = 2;
cfg.min_cluster_size = 2;
cfg.pyramid_levels = 1;
cfg.pyramid_min_size = 128;
cfg.refinement_radius = 3;
cfg.merge_radius = 3.0;
cfg.refiner.kind = RefinementMethod::CenterOfMass;
```

Use `ChessConfig::single_scale()` for the default one-level detector and
`ChessConfig::multiscale()` for the recommended 3-level preset.

`DetectorMode::Broad` enables the wider, blur-tolerant detector response mode.
`DescriptorMode` can either follow the detector or override the descriptor
ring radius explicitly (each descriptor is built by fitting a two-axis tanh
model to the ring samples — see the book's Part III, §3.4).

## Descriptor output

Each detection is a `CornerDescriptor` with:

- `x`, `y` — subpixel position.
- `response` — raw unnormalized ChESS response `R` (paper's score;
  `R > 0` is the default acceptance criterion).
- `contrast` — fitted bright/dark amplitude `|A|` in gray levels.
- `fit_rms` — RMS residual of the two-axis fit in gray levels.
- `axes[0]`, `axes[1]` — the two local grid axes with per-axis 1σ
  angular uncertainty from the Gauss–Newton covariance
  (`σθᵢ = √((SSR / 12) · (JᵀJ)⁻¹[i,i])`). Axes are not assumed
  orthogonal; `axes[0].angle ∈ [0, π)` and `axes[1].angle ∈
  (axes[0].angle, axes[0].angle + π)`, with the CCW arc between them
  spanning a dark sector.

## Refiner configuration

`cfg.refiner` always contains all supported leaf configs:

- `cfg.refiner.center_of_mass`
- `cfg.refiner.forstner`
- `cfg.refiner.saddle_point`

Only `cfg.refiner.kind` selects which one is active:

```rust
use chess_corners::{ChessConfig, RefinementMethod};

let mut cfg = ChessConfig::single_scale();
cfg.refiner.kind = RefinementMethod::Forstner;
cfg.refiner.forstner.max_offset = 2.0;
```

You can also bypass the configured refiner for a single call with
`find_chess_corners_image_with_refiner` or `find_chess_corners_with_refiner`.

## CLI config shape

The CLI uses the same flat algorithm schema at the top level, combined with
application fields such as `image`, `output_json`, `output_png`, `log_level`,
and `ml`.

See:

- `config/chess_algorithm_config_example.json` for the shared algorithm config
- `config/chess_cli_config_example.json` for a complete CLI input

## ML refiner

Enable the `ml-refiner` feature to use the separate ML-backed pipeline:

```rust
use chess_corners::{ChessConfig, find_chess_corners_image_with_ml};
use image::GrayImage;

let img = GrayImage::new(1, 1);
let cfg = ChessConfig::single_scale();
let corners = find_chess_corners_image_with_ml(&img, &cfg);
```

The ML path is slower than the classic refiners and intentionally stays outside
the canonical `ChessConfig` schema.

## Examples

- Single-scale: `cargo run -p chess-corners --example single_scale_image -- testimages/mid.png`
- Multiscale: `cargo run -p chess-corners --example multiscale_image -- testimages/large.png`

## Feature flags

- `image` (default): `image::GrayImage` integration
- `rayon`: parallel response/refinement
- `simd`: portable-SIMD acceleration in the core response path
- `par_pyramid`: SIMD/`rayon` in pyramid construction
- `tracing`: structured spans
- `ml-refiner`: ONNX-backed ML refinement
- `cli`: build the `chess-corners` binary
