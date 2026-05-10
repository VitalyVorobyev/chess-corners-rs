# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`OrientationMethod` enum** (`chess-corners-core`, `chess-corners`)
  with two user-facing variants: `RingFit` (default) and `DiskFit`.
  The API is intentionally compact — it describes what each method does,
  not its implementation lineage.

- **`AxisFitResult` type** — public struct carrying `theta1`, `theta2`,
  `sigma_theta1`, `sigma_theta2`, `amp`, and `rms` from a single
  two-axis orientation fit.

- **`fit_axes_at_point`** and **`fit_axes_from_samples`** — public
  entry points for the orientation fit. `fit_axes_at_point` samples the
  ChESS ring from an image and dispatches to the chosen
  `OrientationMethod`; `fit_axes_from_samples` accepts pre-sampled ring
  values directly, useful for tests and benchmarks where ring sampling is
  decoupled.

- **`corners_to_descriptors_with_method`** — entry point in
  `chess-corners-core` (and re-exported from `chess-corners`) that lets
  callers pick the orientation method explicitly.

### Changed

- **`OrientationMethod` API simplified to two variants.** `RingFit`
  (default) and `DiskFit` replace the previous multi-variant design.
  `RingFit` is bit-identical to the previous `SigmaCorrectionLut` output
  (angles, amplitude, `fit_rms`, and per-axis 1σ uncertainties).
  `DiskFit` is bit-identical to the previous `FullDiskSector` output.
  Serde keys are `"ring_fit"` and `"disk_fit"`.

### Removed

- **`corners_to_descriptors` deprecated shim** — removed; use
  `corners_to_descriptors_with_method` directly.

- **Python `find_chess_corners` legacy JSON-string config argument.**
  The `&str` fallback path that serialized the config through JSON across
  the FFI boundary is removed. Pass a typed `ChessConfig` directly.
  The one-release deprecation notice in 0.7.0 is now fulfilled.

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
