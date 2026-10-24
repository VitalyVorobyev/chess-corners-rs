# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DetectorConfig::uncertainty` (`Uncertainty`): which position uncertainty
  each corner reports. `Propagated` (the default) gives every localizer a
  `position_covariance` — the ChESS center-of-mass, Förstner and
  saddle-point refiners, the Radon peak fit and saddle-point refinement of
  Radon peaks (not the ML refiner): image noise propagated through the
  localizer, plus a per-localizer systematic term (from the corner's offset
  to the localizer's fit window and the width of its edges) with a floor
  measured on real calibration images, about 0.05 px per axis for the
  saddle-point fit and the Radon peak fit, 0.07 for center of mass and 0.11
  for Förstner. On real images the reported σ then matches each
  localizer's own error; error that every corner of an image shares
  (defocus, depth, lens-model error) is outside any single-corner
  covariance and shows up in a downstream calibration's variance factor.
  Weighting a calibration by these covariances is neutral (held-out error
  unchanged within 0.5% for the saddle-point fit); the covariance is
  conservative on ideal synthetic corners and overconfident for the Radon
  peak fit on blurred corners near 45° and for the center of mass on sharp
  boards rotated a few degrees. Cost of the default tier: single-scale
  ChESS frame time +3–9% (the test images), coarse-to-fine +7–22%,
  Radon +0.4–2.5%. `Off` reports no covariance and makes no noise
  estimate; positions are identical under both. Also
  `Refiner::with_uncertainty` in `chess-corners-core` (turns the covariance
  off in custom refinement loops), Python (`Uncertainty`,
  `DetectorConfig.uncertainty`, `with_uncertainty`), WASM (`Uncertainty`,
  `cfg.uncertainty`, `withUncertainty`), the JSON schema, and the C ABI
  (`cc_config::uncertainty`: `CC_UNCERTAINTY_PROPAGATED` (0, so a
  zero-initialised config is the default) / `CC_UNCERTAINTY_OFF`).
- `RadonConfig::refiner` (`RadonRefiner`): the final subpixel position of
  a Radon corner — `PeakFit` (the default, unchanged behaviour) keeps the
  response-map peak fit; `SaddlePoint(SaddlePointConfig)` refines each peak
  with the saddle-point fit on the input image. A corner whose refinement
  does not converge keeps its peak-fit position. On synthetic boards with
  8 px cells the refinement lowers the RMS error per axis from
  0.02–0.17 px to under 0.01 px without noise (lens blur σ 1.5–2.5, any
  rotation) and by 2–6× at noise σ 2; on 4 px cells it is no better.
  Available in Python (`RadonRefiner`, `RadonConfig.refiner`,
  `with_radon(refiner=...)`), WASM (`RadonRefiner`, `withRadonRefiner`),
  the JSON schema and the C ABI (`CC_REFINER_PEAK_FIT` /
  `CC_REFINER_SADDLE_POINT` under `CC_STRATEGY_RADON`).
- `estimate_noise_sigma_u8` in the `chess-corners` facade: the per-image
  noise estimate the detector uses when `noise_sigma` is `None`. Also
  Python `estimate_noise_sigma`, WASM `estimateNoiseSigma` /
  `estimateNoiseSigmaRgba`, C `cc_estimate_noise_sigma_u8` and C++
  `chess_corners::estimate_noise_sigma`.

### Changed

- Every localizer now reports a `position_covariance` by default, not only
  the saddle-point refiner, and the saddle-point covariance's fixed
  0.006 px floor is replaced by its systematic term (larger: the covariance
  is conservative on ideal synthetic corners). Select `Uncertainty::Off`
  for no covariance and no per-image noise estimate.
- `RadonDetector::refine_peaks_on_image` (`chess-corners-core`) replaces a
  peak's position whenever the refiner accepts it and reaches its own fixed
  point: every single-shot refiner (center of mass, Förstner) now moves the
  peaks; the saddle-point refiner still keeps the peak fit where its
  re-centring does not converge.
- C ABI: `cc_abi_version` is 6 (`cc_config::uncertainty` appended).
  `cc_config::refiner` is validated per
  strategy: Radon takes `CC_REFINER_PEAK_FIT` (the `cc_config_radon*`
  presets' value) or `CC_REFINER_SADDLE_POINT`, and rejects the other tags
  with `CC_ERR_INVALID_CONFIG` (it ignored them before).
- `DetectorConfig::radon_multiscale()` (and C `cc_config_radon_multiscale`
  / `multiscale = 1` under Radon) uses a 2-level pyramid instead of 3: a
  quarter-resolution Radon level seeds corners too far off, or not at all
  on boards with fine cell detail. With the fix below, its output on the
  test images is single-scale Radon corners at the same positions:
  `small.png` 1087 → 61 corners (single-scale 72; the other 11 are
  junctions inside ChArUco markers and line or blur peaks), `mid.png`
  78 → 77 (all),
  `large.png` 7004 → 279 (all). It costs about 2× the 3-level time
  (`mid.png` 3.0 → 9.9 ms), still 3–4× faster than single-scale Radon.

### Fixed

- `DetectorConfig::radon_multiscale()` no longer returns hundreds to
  thousands of weak non-corner peaks: the relative `threshold` now applies
  against the strongest corner of the frame, not of each refinement window.
- The Radon detector no longer reports peaks beside straight edges that
  run along one of its ray directions — on sharp boards rotated near 45°,
  hundreds per frame about half a cell from the corners. A peak is now
  kept only where the image is a saddle. Single-scale Radon output on
  `mid.png` and `large.png` is unchanged; on `small.png` three non-corner
  peaks are dropped (75 → 72). Corners on 60°-sheared 4–6 px cells under
  lens blur σ ≥ 1.5 px, where blur leaves little saddle, are dropped too.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
