# chess-corners

Ergonomic chessboard corner detector on top of `chess-corners-core`.

This crate is the public Rust API:

- strategy-typed `DetectorConfig` (`DetectionStrategy::Chess(ChessConfig)`
  / `DetectionStrategy::Radon(RadonConfig)`) with a unified `Threshold`
  enum and per-detector refiner selection (`ChessRefiner`,
  `RadonRefiner`)
- top-level `MultiscaleConfig` (`SingleScale | Pyramid { ... }`) and
  `UpscaleConfig` (`Disabled | Fixed(factor)`), honoured by both
  detectors symmetrically
- single-scale and coarse-to-fine multiscale detection through a single
  `Detector` struct that reuses pyramid and scratch buffers across
  frames
- optional `image::GrayImage` helpers
- optional CLI binary and ML-backed refinement pipeline

`chess-corners-core` and `box-image-pyramid` remain available as
lower-level sharp tools, but `chess-corners` is the intended
compatibility boundary.

## Quick start

```rust
use chess_corners::{Detector, DetectorConfig, Threshold};
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    let mut cfg = DetectorConfig::multiscale();
    cfg.threshold = Threshold::Relative(0.15);

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
`DetectionStrategy` enum and shares cross-cutting fields at the top
level:

```rust
use chess_corners::{
    ChessConfig, ChessRefiner, ChessRing, DetectionStrategy, DetectorConfig,
    DescriptorRing, MultiscaleConfig, RadonConfig, Threshold, UpscaleConfig,
};

let mut cfg = DetectorConfig::single_scale();    // ChESS, no pyramid
cfg.threshold = Threshold::Relative(0.2);         // or Threshold::Absolute(0.0)
cfg.merge_radius = 3.0;

// Multiscale and upscale use the same enum-with-payload shape:
cfg.multiscale = MultiscaleConfig::Pyramid {
    levels: 3,
    min_size: 128,
    refinement_radius: 3,
};
cfg.upscale = UpscaleConfig::Fixed(2);

// Detector-specific knobs live inside the strategy variant:
let mut chess = ChessConfig::default();
chess.ring = ChessRing::Broad;                    // wider, blur-tolerant ring
chess.descriptor_ring = DescriptorRing::FollowDetector;
chess.nms_radius = 2;
chess.min_cluster_size = 2;
chess.refiner = ChessRefiner::default();          // CenterOfMass with defaults
cfg.strategy = DetectionStrategy::Chess(chess);

// Or switch to the Radon strategy:
cfg.strategy = DetectionStrategy::Radon(RadonConfig::default());
```

Three guarantees follow from this shape:

1. **One place per knob.** `cfg.strategy.chess.ring = ChessRing::Broad`
   is the only way to request the wider ChESS ring.
2. **Per-detector refiners.** `ChessRefiner` lists only refiners that
   operate on ChESS output; `RadonRefiner` lists only those that
   operate on Radon output.
3. **Symmetric encoding.** `Threshold`, `MultiscaleConfig`,
   `UpscaleConfig`, and both refiner enums use the same enum-with-
   payload shape, so the JSON and binding surface stays uniform.

## Descriptor output

Each detection is a `CornerDescriptor` with:

- `x`, `y` — subpixel position.
- `response` — raw unnormalized detector response (the ChESS paper's
  score for ChESS, `(max α S_α − min α S_α)²` for Radon).
- `contrast` — fitted bright/dark amplitude `|A|` in gray levels.
- `fit_rms` — RMS residual of the two-axis fit in gray levels.
- `axes[0]`, `axes[1]` — the two local grid axes with per-axis 1σ
  angular uncertainty from the Gauss–Newton covariance
  (`σθᵢ = √((SSR / 12) · (JᵀJ)⁻¹[i,i])`). Axes are not assumed
  orthogonal; `axes[0].angle ∈ [0, π)` and `axes[1].angle ∈
  (axes[0].angle, axes[0].angle + π)`, with the CCW arc between them
  spanning a dark sector.

## Refiner configuration

`ChessRefiner` and `RadonRefiner` are tagged enums; each variant
carries its tuning struct as a payload, so switching kinds cannot
leave a stale per-refiner config behind:

```rust
use chess_corners::{
    ChessConfig, ChessRefiner, DetectionStrategy, DetectorConfig, ForstnerConfig,
};

let mut cfg = DetectorConfig::single_scale();
let mut chess = ChessConfig::default();
chess.refiner = ChessRefiner::Forstner(ForstnerConfig {
    max_offset: 2.0,
    ..ForstnerConfig::default()
});
cfg.strategy = DetectionStrategy::Chess(chess);
```

The Radon equivalent uses `RadonRefiner::RadonPeak(_)` or
`RadonRefiner::CenterOfMass(_)`. A `ChessRefiner::RadonPeak` (or
vice versa) mismatch is unrepresentable.

## CLI config shape

The CLI uses the same `DetectorConfig` schema, combined with
application fields such as `image`, `output_json`, `output_png`,
`log_level`, and `ml`.

See:

- `config/chess_algorithm_config_example.json` for the pure
  `DetectorConfig` shape (round-trips through the Rust and Python
  APIs).
- `config/chess_cli_config_example.json` for a complete CLI runner
  input (algorithm config + envelope).

## ML refiner

Enable the `ml-refiner` feature, then pick the `Ml` variant on the
ChESS strategy's refiner:

```rust
# #[cfg(feature = "ml-refiner")]
# {
use chess_corners::{
    ChessConfig, ChessRefiner, DetectionStrategy, Detector, DetectorConfig,
};
use image::GrayImage;

let img = GrayImage::new(1, 1);
let mut cfg = DetectorConfig::single_scale();
let mut chess = ChessConfig::default();
chess.refiner = ChessRefiner::Ml;
cfg.strategy = DetectionStrategy::Chess(chess);

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
