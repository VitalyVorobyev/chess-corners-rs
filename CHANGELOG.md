# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Top-level config renamed: `ChessConfig` → `DetectorConfig`. The type
  configures both ChESS and Radon detectors; the new name reflects this.
  `ChessConfig` remains a deprecated type alias in Rust and a Python-level
  alias in the `chess_corners` package, to be removed in 0.11.0. The
  WASM `ChessConfig` class is renamed in-place to `DetectorConfig` (no
  JS-side alias). Multiscale settings (`multiscale: Option<MultiscaleParams>`)
  and refiner settings (`refiner: RefinerConfig`) are now top-level fields
  on `DetectorConfig` — both ChESS and Radon detectors honour them
  uniformly. The previously-per-strategy `ChessStrategy.multiscale` field
  has been removed.

### Added

- New preset `DetectorConfig::radon_multiscale()` (Python:
  `DetectorConfig.radon_multiscale()`, WASM: `DetectorConfig.radonMultiscale`)
  for coarse-to-fine Radon detection. The pyramid pipeline now drives both
  detectors symmetrically via the new `DenseDetector` trait in
  `chess-corners-core`.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
