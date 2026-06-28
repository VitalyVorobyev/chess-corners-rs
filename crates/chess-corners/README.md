# chess-corners

Ergonomic chessboard corner detector on top of `chess-corners-core`.

This crate is the public Rust API:

- strategy-typed `DetectorConfig` (`DetectionStrategy::Chess(ChessConfig)`
  / `DetectionStrategy::Radon(RadonConfig)`) with a single numeric
  `threshold` and per-detector refiner selection (`ChessRefiner`,
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
use chess_corners::{Detector, DetectorConfig};
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open("board.png")?.decode()?.to_luma8();

    // ChESS reads `threshold` as an absolute floor on the raw response
    // (default 30); raise it to suppress more textured-region noise.
    let cfg = DetectorConfig::chess_multiscale()
        .with_threshold(60.0);

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

| Preset                                 | Detector | Scale           |
|----------------------------------------|----------|-----------------|
| `DetectorConfig::chess()`              | ChESS    | Single-scale    |
| `DetectorConfig::chess_multiscale()`   | ChESS    | 3-level pyramid |
| `DetectorConfig::radon()`              | Radon    | Single-scale    |
| `DetectorConfig::radon_multiscale()`   | Radon    | 3-level pyramid |

Use `chess()` and `chess_multiscale()` for single-scale and multiscale
ChESS detection respectively.

## Public config shape

`DetectorConfig` groups detector-specific tuning under a typed
`DetectionStrategy` enum and shares cross-cutting fields at the top
level:

```rust
use chess_corners::{
    ChessRefiner, ChessRing, DetectorConfig,
    MultiscaleConfig, UpscaleConfig,
};

let cfg = DetectorConfig::chess()
    .with_threshold(60.0) // ChESS: absolute floor on the raw response (default 30)
    .with_merge_radius(3.0)
    .with_multiscale(MultiscaleConfig::pyramid_default())
    .with_upscale(UpscaleConfig::Fixed(2))
    .with_chess(|c| {
        c.ring = ChessRing::Broad;
        c.refiner = ChessRefiner::default();
    })
    .with_detection(|d| {
        d.nms_radius = 2;
        d.min_cluster_size = 2;
    });
// Or switch to the Radon strategy:
let cfg = cfg.with_radon(|_| {});
```

Three guarantees follow from this shape:

1. **One place per knob.** `cfg.strategy.chess.ring = ChessRing::Broad`
   is the only way to request the wider ChESS ring.
2. **Per-detector refiners.** `ChessRefiner` lists only refiners that
   operate on ChESS output; `RadonRefiner` lists only those that
   operate on Radon output.
3. **Symmetric encoding.** `MultiscaleConfig`, `UpscaleConfig`, and both
   refiner enums use the same enum-with-payload shape, so the JSON and
   binding surface stays uniform. (`threshold` is a plain number, so it
   carries no tag.)

## Descriptor output

Each detection is a `CornerDescriptor` with:

- `x`, `y` — subpixel position.
- `response` — raw unnormalized detector response (the ChESS paper's
  score for ChESS, `(max α S_α − min α S_α)²` for Radon).
- `axes` — `Some([axis0, axis1])` carries the two local grid axes with
  per-axis 1σ angular uncertainty from the Gauss-Newton covariance
  (`σθᵢ = √((SSR / 12) · (JᵀJ)⁻¹[i,i])`). Axes are not assumed
  orthogonal; `axis0.angle ∈ [0, π)` and `axis1.angle ∈
  (axis0.angle, axis0.angle + π)`, with the CCW arc between them
  spanning a dark sector. It is `None` when the orientation fit is
  disabled.

The orientation fit is the dominant per-corner cost. If a downstream
stage recovers board geometry from corner positions alone, disable it
with `DetectorConfig::without_orientation()`: detection still runs and
every descriptor's `axes` comes back `None`.

## Refiner configuration

`ChessRefiner` and `RadonRefiner` are tagged enums; each variant
carries its tuning struct as a payload, so switching kinds cannot
leave a stale per-refiner config behind:

```rust
use chess_corners::{ChessRefiner, DetectorConfig, ForstnerConfig};

let cfg = DetectorConfig::chess().with_chess(|c| {
    // `ForstnerConfig` is `#[non_exhaustive]`: start from `Default` and
    // set the fields you need.
    let mut forstner = ForstnerConfig::default();
    forstner.max_offset = 2.0;
    c.refiner = ChessRefiner::Forstner(forstner);
});
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
use chess_corners::{ChessRefiner, Detector, DetectorConfig};
use image::GrayImage;

let img = GrayImage::new(1, 1);
let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::Ml);

let mut detector = Detector::new(cfg).unwrap();
let _ = detector.detect(&img).unwrap();
# }
```

The ML path is slower than the classic refiners in the shipped
benchmark and falls back to the classic CenterOfMass refiner at coarse
pyramid levels.

## Examples

- Single-scale: `cargo run -p chess-corners --example single_scale_image -- testimages/mid.png`
- Multiscale: `cargo run -p chess-corners --example multiscale_image -- testimages/large.png`

## Feature flags

- `image` (default): `image::GrayImage` integration
- `rayon`: parallel response/refinement
- `simd`: optional high-performance path — portable `std::simd` in the core response path (nightly only)
- `par_pyramid`: SIMD/`rayon` in pyramid construction
- `tracing`: structured spans
- `ml-refiner`: ONNX-backed ML refinement
- `cli`: build the `chess-corners` binary

The stable scalar/autovectorized build is the supported, portable
baseline — correct and fast enough for typical use. It requires Rust
**1.88** or newer (`rust-version` in `Cargo.toml`). The `simd` feature
uses `std::simd` and is the only feature that needs a nightly
toolchain; every other feature builds on stable.
