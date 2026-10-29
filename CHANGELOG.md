# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
- C ABI: `cc_abi_version` is 8. The `CC_REFINER_*` tags are renumbered to
  stay contiguous: `CC_REFINER_CENTER_OF_MASS = 0`,
  `CC_REFINER_SADDLE_POINT = 1`, `CC_REFINER_PEAK_FIT = 2`. Rebuild against
  the new header; code that uses the named constants needs no other change.
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
  config. Use `ChessRefiner::SaddlePoint` (`ChessRefiner::saddle_point()`,
  `"saddle_point"`, `CC_REFINER_SADDLE_POINT`), which is more accurate on
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
