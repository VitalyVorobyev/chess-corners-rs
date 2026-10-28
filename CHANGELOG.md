# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `PositionCovariance` implements `Add`: the covariance of the sum of two
  independent position errors.
- Every binding reports the image noise σ a detection's position
  covariances were computed with (the supplied `noise_sigma` or the
  per-image estimate; absent when no corner has a covariance):
  `Detector::last_noise_sigma()` (Rust), `Detections.noise_sigma`
  (Python), `ChessDetector.noiseSigma()` (JS), `cc_result.has_noise_sigma`
  / `noise_sigma` (C) and an optional `noise_sigma` out-parameter of
  `chess_corners::detect` (C++). The CLI writes it as a top-level
  `noise_sigma` in its JSON output. `cc_result` grows by these two fields,
  so `cc_abi_version()` is now 7: rebuild C and C++ code against the new
  header.
- `Corner::noise_covariance_per_unit_variance()` and
  `RefineResult::noise_covariance_per_unit_variance()` read the noise term
  of the position covariance; prefer them to the public
  `covariance_per_noise_variance` fields, whose change drops the
  systematic term.
- `DenseDetector::attach_peak_covariance` attaches the Radon peak fit's own
  position covariance to the peaks the image refiner left in place.

### Deprecated

- `chess_corners_core::attach_radon_peak_covariance`: use
  `DenseDetector::attach_peak_covariance` on `RadonDetector`.

### Changed

- The default uncertainty tier is now `Off` in every binding: corners carry
  no `position_covariance` and no per-image noise estimate is made unless
  you select `Uncertainty::Propagated` (Python `Uncertainty.PROPAGATED`,
  JS `Uncertainty.Propagated`, C `CC_UNCERTAINTY_PROPAGATED`), which keeps
  the covariance unchanged. In C, `cc_config_default()` and the strategy
  presets return `CC_UNCERTAINTY_OFF`, while a zero-initialised `cc_config`
  still selects `CC_UNCERTAINTY_PROPAGATED`.

### Documentation

- `noise_sigma` (Rust, Python, WASM, C) is the total noise of the 8-bit
  image, rounding included: what `estimate_noise_sigma_u8` measures, or the
  standard deviation of the difference of two frames of a static scene
  divided by √2.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
