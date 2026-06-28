# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Breaking:** dropped the `contrast` and `fit_rms` fields from
  `CornerDescriptor`. The public descriptor is now
  `{ x, y, response, axes }`, and a single `response` score is the
  detection-strength contract. The Python `detect()` array is now
  `(N, 7)` and the WebAssembly `Float32Array` uses stride 7
  (`[x, y, response, axis0_angle, axis0_sigma, axis1_angle,
  axis1_sigma]`); the CLI JSON `corners` entries drop the two fields.
  Migration: use `response` for per-corner strength and `axes[i].sigma`
  for per-axis confidence.
- **Breaking:** the low-level Radon primitives (`ANGLES`, `DIR_COS`,
  `DIR_SIN`, `fit_peak_frac`, `box_blur_inplace`, and the `SatElem`
  summed-area-table element type) are no longer re-exported from
  `chess_corners_core::unstable`; they are now crate-internal.
  Detection results are unchanged.
- **Breaking:** removed the `descriptor_ring` ChESS config option
  (Rust `DescriptorRing` enum, Python/WASM `descriptor_ring` field,
  CLI `--descriptor-ring` flag). Descriptors now always sample at the
  detector ring radius (`ring` / `ChessRing`).

### Changed

- **Breaking:** `ChessParams` and `RefinerKind` moved off the
  `chess-corners-core` crate root into the unstable, no-semver-guarantee
  `chess_corners_core::unstable` namespace, where they are documented as
  implementation-level translation types. The `chess-corners` facade
  still re-exports both unchanged at `chess_corners::low_level`.
- **Breaking:** the classic refiner config structs `CenterOfMassConfig`,
  `ForstnerConfig`, and `SaddlePointConfig` (matching `RadonPeakConfig`),
  and the `ChessBuffers` scratch carrier, are now `#[non_exhaustive]`, so
  future tuning knobs and scratch fields stay additive. Construct them
  from `Default::default()` and assign fields instead of using
  struct-literal or `..` update syntax across the crate boundary
  (`ChessBuffers` is built via `ChessBuffers::default()`; its `response`
  field stays readable).
- **Breaking:** the `DenseDetector` and `CornerRefiner` traits are now
  sealed and cannot be implemented outside `chess-corners-core`; they
  are not public extension points. Select a refinement backend through
  `RefinerKind` / the detector configuration.
- Documented the minimum supported Rust version: the default (stable)
  build needs Rust ≥ 1.88; the `simd` feature needs a nightly toolchain.

- **Breaking:** the `nms_radius` and `min_cluster_size` detection knobs
  are no longer duplicated on each strategy config. They now live once
  on `DetectorConfig.detection` as a shared `DetectionParams`, honoured
  by both the ChESS and Radon detectors. Tune them with
  `DetectorConfig::with_detection(|d| …)` (Rust),
  `cfg.with_detection(nms_radius=…, min_cluster_size=…)` or
  `cfg.detection.nms_radius` (Python), and
  `cfg.withDetection({ nmsRadius, minClusterSize })` or
  `cfg.detection.nmsRadius` (WebAssembly). JSON / dict configs move the
  two keys out of `strategy.{chess,radon}` into a top-level `detection`
  object. Detection results are unchanged.
- Bump `numpy` to `0.29` and `pyo3` to `0.29`

## Past releases

Detailed notes for prior releases live under `docs/changelog/`.

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
