# chess-corners

Ergonomic chessboard corner detector on top of `chess-corners-core`.

This crate is the public Rust API:

- strategy-typed `DetectorConfig` (`DetectionStrategy::Chess` /
  `DetectionStrategy::Radon`) with a unified `Threshold` enum and
  pluggable refiner selection
- top-level `multiscale: Option<MultiscaleParams>` and `refiner: RefinerConfig`
  honoured by both detectors
- single-scale and coarse-to-fine multiscale detection through a single
  `Detector` struct
- optional `image::GrayImage` helpers
- optional CLI binary and ML-backed refinement pipeline

`chess-corners-core` and `box-image-pyramid` remain available as lower-level
sharp tools, but `chess-corners` is the intended compatibility boundary.

## Quick start

```rust
use chess_corners::{DetectorConfig, Detector, RefinementMethod, Threshold};
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    let mut cfg = DetectorConfig::multiscale();
    cfg.threshold = Threshold::Relative(0.15);
    cfg.refiner.kind = RefinementMethod::Forstner;

    let mut detector = Detector::new(cfg)?;
    let corners = detector.detect(&img)?;
    println!("found {} corners", corners.len());
    Ok(())
}
```

`Detector` owns the pyramid and upscale scratch buffers, so calling
`detector.detect(&img)` repeatedly on successive frames does not
re-allocate.

## Presets

| Preset                              | Detector | Scale          |
|-------------------------------------|----------|----------------|
| `DetectorConfig::single_scale()`    | ChESS    | Single-scale   |
| `DetectorConfig::multiscale()`      | ChESS    | 3-level pyramid |
| `DetectorConfig::radon()`           | Radon    | Single-scale   |
| `DetectorConfig::radon_multiscale()`| Radon    | 3-level pyramid |

## Public config shape

`DetectorConfig` groups detector-specific tuning under a typed
[`DetectionStrategy`] enum and shares cross-cutting fields at the top
level:

```rust
use chess_corners::{
    DetectorConfig, ChessRing, ChessStrategy, DescriptorMode, DetectionStrategy,
    MultiscaleParams, RadonStrategy, RefinementMethod, Threshold,
};

let mut cfg = DetectorConfig::single_scale();   // ChESS, multiscale = None
cfg.threshold = Threshold::Relative(0.2);    // or Threshold::Absolute(0.0)
cfg.descriptor_mode = DescriptorMode::FollowDetector;
cfg.merge_radius = 3.0;
cfg.refiner.kind = RefinementMethod::CenterOfMass;

// Enable the coarse-to-fine pyramid (works for both ChESS and Radon):
cfg.multiscale = Some(MultiscaleParams {
    pyramid_levels: 3,
    pyramid_min_size: 128,
    refinement_radius: 3,
});

// Detector-specific knobs live inside the strategy variant:
if let DetectionStrategy::Chess(chess) = &mut cfg.strategy {
    chess.ring = ChessRing::Broad;            // wider, blur-tolerant ring
    chess.nms_radius = 2;
    chess.min_cluster_size = 2;
}

// Or switch to the Radon strategy:
cfg.strategy = DetectionStrategy::Radon(RadonStrategy::default());
```

`ChessRing::Broad` enables the wider, blur-tolerant detector response
mode. `DescriptorMode` can either follow the detector or override the
descriptor ring radius explicitly (each descriptor is built by fitting
a two-axis tanh model to the ring samples — see the book's Part III,
§3.4).

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
- `cfg.refiner.radon_peak`

Only `cfg.refiner.kind` selects which one is active:

```rust
use chess_corners::{DetectorConfig, RefinementMethod};

let mut cfg = DetectorConfig::single_scale();
cfg.refiner.kind = RefinementMethod::Forstner;
cfg.refiner.forstner.max_offset = 2.0;
```

To switch refiners on the fly without rebuilding the detector, use
`detector.config_mut().refiner.kind = ...`.

## CLI config shape

The CLI uses the same strategy-typed algorithm schema, combined with
application fields such as `image`, `output_json`, `output_png`,
`log_level`, and `ml`.

See:

- `config/chess_algorithm_config_example.json` for the shared algorithm config
- `config/chess_cli_config_example.json` for a complete CLI input

## ML refiner

Enable the `ml-refiner` feature, then pick the ML pipeline by setting
the refiner kind:

```rust
# #[cfg(feature = "ml-refiner")]
# {
use chess_corners::{DetectorConfig, Detector, RefinementMethod};
use image::GrayImage;

let img = GrayImage::new(1, 1);
let mut cfg = DetectorConfig::single_scale();
cfg.refiner.kind = RefinementMethod::Ml;

let mut detector = Detector::new(cfg).unwrap();
let _ = detector.detect(&img).unwrap();
# }
```

The ML path is slower than the classic refiners and falls back to the
classic CenterOfMass refiner at coarse pyramid levels.

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
