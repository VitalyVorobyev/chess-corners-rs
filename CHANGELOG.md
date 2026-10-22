# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CornerDescriptor::position_covariance`: the 2×2 covariance of each
  corner's position in px², for weighting corners in a calibration. It is
  the scatter due to white pixel noise (variance at least 1/12 gray level²,
  the 8-bit rounding) plus a model-error floor of 0.006 px per axis. On
  synthetic boards formed with lens blur and 8–12 px cells at any rotation,
  whitened errors have variance 0.6–1.75 at noise σ 1 and 0.7–1.15 at
  σ ≥ 4. Reported for the `SaddlePoint` refiner; `None` for the
  other refiners, the Radon detector, and upscaled input. Python
  `Detections.position_covariance` (`(N, 2, 2)`, `NaN` rows where absent),
  WASM `ChessDetector.positionCovariance()` (stride 3), C `cc_corner`
  `position_covariance` / `has_position_covariance` (`cc_abi_version` 5),
  and the CLI's corner JSON carry it.
- `DetectorConfig::noise_sigma` (`with_noise_sigma`): the image's pixel
  noise in gray levels used for that covariance; `None` (default)
  estimates it from each image. Available in Python, WASM, the C ABI
  (`has_noise_sigma` / `noise_sigma`) and the JSON schema.
- `estimate_noise_sigma_u8` (`chess-corners-core`): estimates the white
  pixel-noise σ of an 8-bit image, ignoring edges, corners, smooth shading
  and saturated pixels.
- `Corner::covariance_per_noise_variance` and `Corner::position_covariance`
  (`chess-corners-core`): the refiner's covariance carried through
  detection, and its px² form for a given noise level.
- Book Part VIII, "Uncertainty": what the position covariance describes,
  how it is computed, where it matches the measured error and where it
  does not, and how to weight a least-squares fit with it.

### Changed

- The `SaddlePoint` refiner refits on a patch re-centred at its own
  estimate until the fitted saddle sits at the window centre. A single
  fit on the pixel grid is pulled toward the seed pixel; refined
  positions now move toward the true corner (noise-free RMS error on
  synthetic grid-aligned corners 0.12 px → under 0.01 px). Which corners
  are accepted is unchanged. A corner costs about 0.2 µs.
  `covariance_per_noise_variance` describes the noise scatter of the
  converged position. The position keeps a small error that depends on
  where the corner sits between pixels: up to about 0.009 px RMS on boards
  rotated 30–45° from the pixel grid under a lens blur of σ ≥ 0.5 px, and
  up to 0.02 px on boards rotated 2–15° or 45° with ideally sharp optics.
  When re-centring
  does not converge, the refiner returns the single fit with no
  covariance. On cells of about 5 px with strong shear and blur the
  re-centred position can be less accurate than the single fit.

### Fixed

- `CenterOfMassRefiner` returns `RefineStatus::IllConditioned` instead of
  `Accepted` when its window holds no positive response, so a seed it
  could not refine is no longer reported as refined. Detector output is
  unchanged: every peak the detector seeds it with has positive response.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
