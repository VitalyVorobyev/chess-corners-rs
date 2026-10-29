# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The ML refiner refines Radon corners too: `ImageRefiner::Ml` works with
  either detector. On Radon it starts from the peak fit, and a corner it
  cannot refine keeps the peak fit and, under
  `Uncertainty::Propagated`, its covariance.

### Changed

- Subpixel refinement is one optional, detector-independent stage. Each
  detector keeps the localizer that works on its own response map — the
  ChESS center of mass (`ChessConfig::center_of_mass: CenterOfMassConfig`)
  and the Radon peak fit (`RadonConfig::peak_fit`) — and
  `DetectorConfig::refiner: Option<ImageRefiner>` adds an image refiner,
  `ImageRefiner::SaddlePoint(SaddlePointConfig)` or (feature
  `ml-refiner`) `ImageRefiner::Ml`, for either detector. `None`, the
  default, keeps the detector's own localizer. On ChESS an image refiner
  replaces the center of mass, starts from the integer peak and drops a
  corner it rejects; on Radon it starts from the peak fit and keeps it
  where it does not converge. The refiner survives `with_chess` /
  `with_radon` strategy switches. `ChessRefiner`, `RadonRefiner`,
  `ChessConfig::refiner` and `RadonConfig::refiner` are removed. Outputs
  of every configuration that maps across are unchanged bit for bit.
  Migration:

  | Binding | Before | After |
  |---------|--------|-------|
  | Rust | `.with_chess(\|c\| c.refiner = ChessRefiner::SaddlePoint(cfg))` | `.with_refiner(ImageRefiner::SaddlePoint(cfg))` |
  | Rust | `.with_radon(\|r\| r.refiner = RadonRefiner::saddle_point())` | `.with_refiner(ImageRefiner::saddle_point())` |
  | Rust | `ChessRefiner::CenterOfMass(cfg)` | `refiner: None` (default) plus `.with_chess(\|c\| c.center_of_mass = cfg)` |
  | Rust | `RadonRefiner::PeakFit` | `refiner: None` (default) |
  | Rust | `ChessRefiner::Ml` | `.with_refiner(ImageRefiner::Ml)` |
  | JSON | `"strategy": {"chess": {"refiner": {"center_of_mass": {...}}}}` | `"strategy": {"chess": {"center_of_mass": {...}}}` |
  | JSON | `"strategy": {"chess" \| "radon": {"refiner": {"saddle_point": {...}}}}` | top-level `"refiner": {"saddle_point": {...}}` |
  | JSON | `"strategy": {"radon": {"refiner": "peak_fit"}}` | `"refiner": null` or absent |
  | JSON | `"strategy": {"chess": {"refiner": "ml"}}`, CLI `"ml": true` | `"refiner": "ml"` |
  | Python | `cfg.strategy.chess.refiner = ChessRefiner.saddle_point(c)` | `cfg.refiner = ImageRefiner.saddle_point(c)` or `cfg.with_refiner(...)` |
  | Python | `RadonRefiner.saddle_point()` / `.peak_fit()`, `with_radon(refiner=...)` | `ImageRefiner.saddle_point()` / `None`, `with_refiner(...)` / `without_refiner()` |
  | Python | `ChessRefiner.center_of_mass(c)`, `ChessRefiner.ml()` | `cfg.strategy.chess.center_of_mass = c`, `ImageRefiner.ml()` |
  | JS | `withChessRefiner(ChessRefiner.withSaddlePoint(c))`, `withRadonRefiner(RadonRefiner.withSaddlePoint(c))` | `withRefiner(ImageRefiner.saddlePointWith(c))` |
  | JS | `ChessRefiner.withCenterOfMass(c)`, `RadonRefiner.withPeakFit()` | `cfg.strategy.chess.centerOfMass = c`, `withoutRefiner()` |
  | C | `CC_REFINER_CENTER_OF_MASS`, `CC_REFINER_PEAK_FIT` | `CC_IMAGE_REFINER_NONE` (`0`, zero-initialised default) |
  | C | `CC_REFINER_SADDLE_POINT` | `CC_IMAGE_REFINER_SADDLE_POINT`, now valid for either strategy |
  | CLI | `--chess-refiner center_of_mass\|saddle_point` | `--refiner none\|saddle_point\|ml` |
  | core | `ChessParams::refiner`, `RefinerKind::CenterOfMass(cfg)` | `ChessParams::center_of_mass = cfg`; `RefinerKind` holds only image refiners, and `Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg))` builds the localizer |
- `ChessConfig` and `RadonConfig` reject unknown JSON keys (with
  `serde_json` and in Python), so a configuration written for the removed
  per-strategy `refiner` field fails to load instead of silently dropping
  the refiner.

- The uncertainty setting carries its noise level:
  `Uncertainty::{Off, Propagated { noise: NoiseLevel }}` with
  `NoiseLevel::{Estimate, Known(f32)}` (both now in `chess-corners`).
  `DetectorConfig::noise_sigma` and `with_noise_sigma` are removed; a
  `Known(σ)` that is negative or not finite is still rejected by
  `Detector::new` (`ChessError::InvalidNoiseSigma`). Positions and
  covariances are unchanged. Migration:
  - Rust: `.with_uncertainty(Uncertainty::Propagated)` →
    `.with_uncertainty(Uncertainty::Propagated { noise: NoiseLevel::Estimate })`;
    adding `.with_noise_sigma(σ)` → `noise: NoiseLevel::Known(σ)`.
  - JSON config: `"uncertainty": "propagated"` →
    `{"propagated": {"noise": "estimate"}}`; with `"noise_sigma": σ` →
    `{"propagated": {"noise": {"known": σ}}}`. `"off"` is unchanged.
  - Python: `Uncertainty` is an immutable value: `Uncertainty.OFF` →
    `Uncertainty.off()`, `Uncertainty.PROPAGATED` →
    `Uncertainty.propagated()`, plus `with_noise_sigma(σ)` /
    `cfg.noise_sigma = σ` → `Uncertainty.propagated(noise_sigma=σ)`.
    `DetectorConfig.from_dict` rejects the removed `noise_sigma` key.
  - JS: `Uncertainty.Off` → `Uncertainty.off()`, `Uncertainty.Propagated`
    → `Uncertainty.propagated()`, plus `withNoiseSigma(σ)` /
    `cfg.noiseSigma = σ` → `Uncertainty.propagated(σ)`.
  - C: `CC_UNCERTAINTY_OFF` is now `0` and `CC_UNCERTAINTY_PROPAGATED` `1`,
    so a zero-initialised `cc_config` selects off, like every preset.
    `has_noise_sigma` / `noise_sigma` are read only under
    `CC_UNCERTAINTY_PROPAGATED`. C++: `Config::uncertainty` is a
    `chess_corners::Uncertainty` (`Uncertainty::off()`,
    `Uncertainty::propagated(sigma)`) in place of the tag plus
    `Config::noise_sigma`.
  - `chess-corners-core`: `Uncertainty` moved to `chess-corners`;
    `Refiner::with_uncertainty(tier)` → `Refiner::with_covariance(bool)`.
- Each orientation axis reports `fit_std_error`, the residual-based
  least-squares standard error of the axis fit in radians, in place of
  `sigma`. It is a fit-quality score for ranking and gating axes, not a
  calibrated error bound. Both methods now report this same quantity:
  `RingFit` values are the previous ones divided by 1.0–1.25 (the empirical
  correction factor is gone); `DiskFit` computes it over the disk pixels
  instead of the previous residual heuristic, and its two axes now differ.
  The 10° minimum when `DiskFit` and `RingFit` disagree and π as "no
  information" are unchanged, and so are positions, covariances and axis
  angles. The two values of a `RingFit` axis pair are no longer exchanged
  when the fit relabels its axes. A gate tuned on the previous values
  rejects about half as many axes; tighten it by about ×0.65 to keep its
  rejection rate. Renamed:
  - Rust: `AxisEstimate::sigma` → `AxisEstimate::fit_std_error` (and the
    second parameter of `AxisEstimate::new`); core `AxisFitResult::
    sigma_theta1` / `sigma_theta2` → `fit_std_error_theta1` /
    `fit_std_error_theta2`.
  - Python: `Detections.sigmas` → `Detections.fit_std_errors`.
  - JS: the `axis0_sigma` / `axis1_sigma` slots of the stride-7 corner
    array are `axis0_fitStdError` / `axis1_fitStdError` (same offsets).
  - C: `cc_axis::sigma` → `cc_axis::fit_std_error`; C++ `Axis::sigma` →
    `Axis::fit_std_error`.
  - CLI JSON output: `axes[].sigma` → `axes[].fit_std_error`.
- C ABI: `cc_abi_version` is 8. `cc_config::refiner` is a
  `cc_image_refiner_t` (`CC_IMAGE_REFINER_NONE = 0`,
  `CC_IMAGE_REFINER_SADDLE_POINT = 1`; see the migration table above).
  Rebuild against the new header.
- With the `rayon` feature, the subpixel refinement and position covariance
  of large corner sets, the Radon peak-fit covariance, and the multiscale
  per-seed refinement run in parallel. Results are unchanged bit for bit;
  multiscale ChESS frames take about half the time or less.

### Removed

- The Förstner refiner, everywhere: `ChessRefiner::Forstner` and
  `ChessRefiner::forstner()`, core `RefinerKind::Forstner`,
  `Refiner::Forstner`, `ForstnerConfig` and `ForstnerRefiner`, Python
  `ForstnerConfig` and `ChessRefiner.forstner()`, JS `ForstnerConfig` and
  `ChessRefiner.withForstner`, C `CC_REFINER_FORSTNER`, the CLI
  `--chess-refiner forstner` choice and the `"forstner"` refiner in the JSON
  config. Use the saddle-point image refiner (`ImageRefiner::saddle_point()`,
  `"saddle_point"`, `CC_IMAGE_REFINER_SADDLE_POINT`), which is more accurate on
  real calibration images and on almost every synthetic condition.
- The `covariance_per_noise_variance` fields of `Corner` and
  `RefineResult`: read `noise_covariance_per_unit_variance()`, or
  `position_covariance(noise_sigma)` for the full covariance in px².
- `chess_corners_core::attach_radon_peak_covariance`: use
  `DenseDetector::attach_peak_covariance` on `RadonDetector`.
- `DenseDetector::refines_on_image`: both detectors always refine on the
  image; drop the call.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

- [1.7.0](docs/changelog/1.7.0.md) — 2026-10-29
- [1.6.0](docs/changelog/1.6.0.md) — 2026-10-24
- [1.5.0](docs/changelog/1.5.0.md) — 2026-10-22
- [1.4.0](docs/changelog/1.4.0.md) — 2026-10-22
- [1.3.1](docs/changelog/1.3.1.md) — 2026-10-19
- [1.3.0](docs/changelog/1.3.0.md) — 2026-10-03
- [1.2.0](docs/changelog/1.2.0.md) — 2026-07-23
- [1.1.0](docs/changelog/1.1.0.md) — 2026-07-22
- [1.0.0](docs/changelog/1.0.0.md) — 2026-07-02
- [0.11.0](docs/changelog/0.11.0.md) — 2026-05-17
- [0.10.0](docs/changelog/0.10.0.md) — 2026-05-14
- [0.9.0](docs/changelog/0.9.0.md) — 2026-05-10
- [0.8.0](docs/changelog/0.8.0.md) — 2026-04-28
- [0.7.0](docs/changelog/0.7.0.md) — 2026-04-26
- [0.6.0](docs/changelog/0.6.0.md)
- [0.5.0](docs/changelog/0.5.0.md)
- [0.4.2](docs/changelog/0.4.2.md)
- [0.4.1](docs/changelog/0.4.1.md)
- [0.4.0](docs/changelog/0.4.0.md)
- [0.3.2](docs/changelog/0.3.2.md)
- [0.3.1](docs/changelog/0.3.1.md)
- [0.3.0](docs/changelog/0.3.0.md)
- [0.2.1](docs/changelog/0.2.1.md)
- [0.2.0](docs/changelog/0.2.0.md)
- [0.1.2](docs/changelog/0.1.2.md)
