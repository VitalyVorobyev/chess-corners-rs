# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ChessConfig::on_refiner_reject` (`OnRefinerReject::{Drop, KeepCenterOfMass}`):
  what a ChESS detection does with a corner the image refiner rejects.
  `Drop`, the default, is the existing behaviour. `KeepCenterOfMass` keeps
  the corner at the ChESS center-of-mass position, with that localizer's
  position covariance; the corners the image refiner accepts are unchanged,
  so a single-scale detection returns the `Drop` output plus the kept
  corners. On cluttered or blurred boards the saddle-point refiner rejects
  real corners as well as clutter; this keeps the center of mass's recall
  while every corner the refiner can fit gets its accuracy. Ignored without
  an image refiner and by the Radon detector, which never drops a corner.
  - Python `OnRefinerReject.{DROP, KEEP_CENTER_OF_MASS}` and
    `ChessConfig.on_refiner_reject` / `with_chess(on_refiner_reject=...)`;
    JS `OnRefinerReject` and `cfg.strategy.chess.onRefinerReject`; JSON
    `"on_refiner_reject": "drop" | "keep_center_of_mass"` under
    `strategy.chess`; CLI `--on-refiner-reject`.
  - C: `cc_config::on_refiner_reject` (`CC_ON_REFINER_REJECT_DROP`, the
    zero-initialised value, or `CC_ON_REFINER_REJECT_KEEP_CENTER_OF_MASS`),
    appended to `cc_config`, so `cc_abi_version()` is 9: rebuild C and C++
    code against the new header. C++: `Config::on_refiner_reject`.
  - core: `refine_corners_on_image_with_fallback` and
    `DenseDetector::refine_peaks_on_image_with_fallback`.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

- [2.0.0](docs/changelog/2.0.0.md) — 2026-10-29
- [1.7.1](docs/changelog/1.7.1.md) — 2026-10-29
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
