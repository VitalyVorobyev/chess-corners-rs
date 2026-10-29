# chess-corners

Ergonomic chessboard corner detector on top of `chess-corners-core`.

This crate is the public Rust API:

- strategy-typed `DetectorConfig` (`DetectionStrategy::Chess(ChessConfig)`
  / `DetectionStrategy::Radon(RadonConfig)`) with a single numeric
  `threshold` and one optional, detector-independent image refiner
  (`ImageRefiner`)
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
| `DetectorConfig::radon_multiscale()`   | Radon    | 2-level pyramid |

Use `chess()` and `chess_multiscale()` for single-scale and multiscale
ChESS detection respectively.

## Public config shape

`DetectorConfig` groups detector-specific tuning under a typed
`DetectionStrategy` enum and shares cross-cutting fields at the top
level:

```rust
use chess_corners::{
    ChessRing, DetectorConfig, ImageRefiner, MultiscaleConfig, UpscaleConfig,
};

let cfg = DetectorConfig::chess()
    .with_threshold(60.0) // ChESS: absolute floor on the raw response (default 30)
    .with_merge_radius(3.0)
    .with_multiscale(MultiscaleConfig::pyramid_default())
    .with_upscale(UpscaleConfig::Fixed(2))
    .with_chess(|c| {
        c.ring = ChessRing::Broad;
        c.center_of_mass.radius = 2;
    })
    .with_refiner(ImageRefiner::saddle_point())
    .with_detection(|d| {
        d.nms_radius = 2;
        d.min_cluster_size = 2;
    });
// Or switch to the Radon strategy (the image refiner stays):
let cfg = cfg.with_radon(|_| {});
```

Three guarantees follow from this shape:

1. **One place per knob.** `cfg.strategy.chess.ring = ChessRing::Broad`
   is the only way to request the wider ChESS ring.
2. **Every combination is valid.** Each detector tunes its own subpixel
   localizer in its strategy config (ChESS center of mass, Radon peak
   fit); the optional image refiner (`DetectorConfig::refiner`) reads
   only the image, so every `ImageRefiner` works with both detectors.
3. **Symmetric encoding.** `MultiscaleConfig`, `UpscaleConfig`, and
   `ImageRefiner` use the same enum-with-payload shape, so the JSON and
   binding surface stays uniform. (`threshold` is a plain number, so it
   carries no tag.)

## Descriptor output

Each detection is a `CornerDescriptor` with:

- `x`, `y` — subpixel position.
- `response` — raw unnormalized detector response (the ChESS paper's
  score for ChESS, `(max α S_α − min α S_α)²` for Radon).
- `axes` — `Some([axis0, axis1])` carries the two local grid axes, each
  with `fit_std_error`: the residual-based least-squares standard error of
  the axis fit in radians, `√([(JᵀJ)⁻¹]θθ · SSR / (N − 4))` over the ring
  samples (`RingFit`) or disk pixels (`DiskFit`). It is a fit-quality
  score for ranking and gating (larger means a less reliable axis), not a
  calibrated error bound, and independent of the noise level and
  `Uncertainty`. Axes are not assumed
  orthogonal; `axis0.angle ∈ [0, π)` and `axis1.angle ∈
  (axis0.angle, axis0.angle + π)`, with the CCW arc between them
  spanning a dark sector. It is `None` when the orientation fit is
  disabled.
- `position_covariance` — `Some` 2×2 covariance of `(x, y)` in px², for
  weighting the corner in a calibration: the scatter due to white pixel
  noise plus the localizer's systematic error, which includes a floor
  measured on real images that usually dominates there (see
  `Corner::position_covariance` in `chess-corners-core`; weighting by it
  has not been shown to beat uniform weighting on real data). Every localizer
  reports it when `DetectorConfig::uncertainty` is
  `Uncertainty::Propagated { noise }`; it is `None` under the default
  `Off`. The noise level is estimated from each image with
  `NoiseLevel::Estimate` (`estimate_noise_sigma_u8` returns the same
  estimate), or supplied with `NoiseLevel::Known(σ)`; it is `None` on
  upscaled input.
  `Detector::last_noise_sigma` returns the σ the most recent detection
  used.

The orientation fit is the dominant per-corner cost. If a downstream
stage recovers board geometry from corner positions alone, disable it
with `DetectorConfig::without_orientation()`: detection still runs and
every descriptor's `axes` comes back `None`.

## Subpixel localization

Each detector places its corners with its own localizer on its response
map: ChESS with the center of mass (`ChessConfig::center_of_mass`),
Radon with the 3-point peak fit (`RadonConfig::peak_fit`). By default
that position is final. An image refiner (`DetectorConfig::refiner`)
refines the corners of either detector on the image; each variant
carries its tuning struct:

```rust
use chess_corners::{DetectorConfig, ImageRefiner, SaddlePointConfig};

// `SaddlePointConfig` is `#[non_exhaustive]`: start from `Default` and
// set the fields you need.
let mut saddle = SaddlePointConfig::default();
saddle.max_offset = 2.0;
let cfg = DetectorConfig::chess().with_refiner(ImageRefiner::SaddlePoint(saddle));
let cfg = DetectorConfig::radon().with_refiner(ImageRefiner::saddle_point());
```

On ChESS the image refiner replaces the center of mass, starts from the
integer peak and drops a corner it rejects, unless
`ChessConfig::on_refiner_reject` is `OnRefinerReject::KeepCenterOfMass`,
which keeps it at its center-of-mass position; on Radon it starts from the
peak fit and keeps the peak fit where it does not converge.

## CLI config shape

The CLI config holds application fields (`image`, `output_json`,
`output_png`, `log_level`) at the top level and a `DetectorConfig`
under `detector`. An unknown key at either level is an error that
names it. The CLI reads PNG, JPEG, TIFF, BMP, WebP and GIF
input images.

See:

- `config/chess_algorithm_config_example.json` for the pure
  `DetectorConfig` shape (round-trips through the Rust and Python
  APIs).
- `config/chess_cli_config_example.json` for a complete CLI runner
  input (I/O fields + `detector`).

## Examples

- Single-scale: `cargo run -p chess-corners --example single_scale_image -- testimages/mid.png`
- Multiscale: `cargo run -p chess-corners --example multiscale_image -- testimages/large.png`

## Feature flags

- `image` (default): `image::GrayImage` integration, no image-format codecs
- `rayon`: parallel response/refinement
- `simd`: optional high-performance path — portable `std::simd` in the core response path (nightly only)
- `par_pyramid`: SIMD/`rayon` in pyramid construction
- `tracing`: structured spans
- `cli`: build the `chess-corners` binary

`image` only wires up in-memory buffer types (`GrayImage`,
`ImageBuffer`, `Luma`) — it enables no image-format codec, so this
crate never decodes or encodes a file itself. If your code calls
`image::open` or `ImageReader::decode`, declare the formats you need
on your own `image` dependency:

```toml
image = { version = "0.25", features = ["png", "jpeg"] }
```

The stable scalar/autovectorized build is the supported, portable
baseline — correct and fast enough for typical use. It requires Rust
**1.91** or newer (`rust-version` in `Cargo.toml`). The `simd` feature
uses `std::simd` and is the only feature that needs a nightly
toolchain; every other feature builds on stable.
