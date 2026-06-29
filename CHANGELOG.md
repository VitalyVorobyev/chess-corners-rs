# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Orientation can now be skipped. `DetectorConfig::without_orientation()`
  (Rust), `without_orientation()` (Python), `withoutOrientation()` (WASM),
  and `CC_ORIENTATION_NONE` (C) skip the per-corner axis fit entirely —
  useful when a downstream stage recovers board geometry and does not need
  per-corner orientation. With the fit skipped, detection returns the same
  corner positions with no axis data.

### Removed

- **Breaking:** removed the `RadonRefiner` config — the Rust enum, the
  `RadonConfig.refiner` field, the Python/WASM `RadonRefiner` type, and the
  CLI `--radon-refiner` flag. It was a no-op: the Radon detector never
  applied a pluggable refiner, so the choice had no effect on output.
  Radon's subpixel accuracy comes from its built-in Gaussian peak fit
  (`peak_fit`), which is unchanged. Migration: delete any `refiner` set on
  a Radon config — Radon detection results are identical.
- **Breaking:** removed the internal `RadonPeak` refiner — the core
  `RefinerKind::RadonPeak` variant, the `RadonPeakConfig` type (including
  the Python and WebAssembly `RadonPeakConfig` classes), and the C
  `CC_REFINER_RADON_PEAK` tag. It was never selectable through the public
  `ChessRefiner` configuration and was reachable only from the removed
  `RadonRefiner` path; the Radon detector's Gaussian peak fit is unchanged.
- **Breaking:** dropped the `contrast` and `fit_rms` fields from
  `CornerDescriptor`. The public descriptor is now
  `{ x, y, response, axes }`, and a single `response` score is the
  detection-strength contract. The WebAssembly `Float32Array` uses
  stride 7 (`[x, y, response, axis0_angle, axis0_sigma, axis1_angle,
  axis1_sigma]`) and the CLI JSON `corners` entries drop the two fields.
  (The Python result is a structure-of-arrays `Detections` object — see
  *Changed* below.) Migration: use `response` for per-corner strength and
  `axes[i].sigma` for per-axis confidence.
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

- **Breaking (Python):** `Detector.detect()` returns a `Detections` object
  with named numpy arrays — `xy` (N×2), `response` (N), and `angles` /
  `sigmas` (N×2, or `None` when orientation is disabled) — instead of a
  dense `(N, 7)` array. Clearer than positional columns and makes
  "orientation off" explicit (`None` vs `NaN`). Migration: `arr[:, :2]` →
  `det.xy`, `arr[:, 2]` → `det.response`, and the axis columns →
  `det.angles` / `det.sigmas`.
- **Breaking:** the detector acceptance threshold is now a single number
  (`DetectorConfig.threshold: f32`) instead of an `Absolute` / `Relative`
  enum. ChESS reads it as an absolute floor on the response; Radon reads it
  as a fraction in `[0, 1]` of the per-frame maximum. The CLI exposes one
  `--threshold` flag, the Python/WASM configs take a plain number, and JSON
  configs write `"threshold": <number>`. Migration: replace
  `Threshold::Absolute(v)` with `v` (ChESS) and `Threshold::Relative(f)`
  with `f` (Radon).
- **Breaking:** the default ChESS threshold is now `30` (was `0`). The
  previous default accepted every positive response and produced large
  numbers of spurious detections on textured backgrounds; `30` keeps
  well-formed corners while suppressing that noise. Useful values run
  roughly `30`–`300` depending on image contrast; set `threshold`
  explicitly to recover the previous behaviour.
- **Breaking:** the Radon detector's default acceptance threshold is now
  `0.30` (was `0.01`). It is read as a fraction of the per-frame maximum
  response; raising the default eliminates low-response texture hits and
  keeps a clean board-corner set. To restore the previous permissive
  behaviour, set the threshold explicitly — e.g.
  `DetectorConfig::radon().with_threshold(0.01)`.
- **Breaking:** `CornerDescriptor.axes` is now `Option<[AxisEstimate; 2]>`,
  `None` when the orientation fit was skipped. The Python `Detections`
  object then reports `angles` / `sigmas` as `None`; the WASM
  `Float32Array` puts `NaN` in the four axis columns; the CLI JSON
  writes `axes: null`.
- **Breaking (C ABI, version 3):** `cc_config` drops the threshold-kind tag
  and the `CC_THRESHOLD_*` constants (it now carries a single `threshold`
  field), and `cc_corner` gains a `has_orientation` flag (`0` when the axis
  fields are unset). Check `cc_abi_version()`.
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

### Migration to 1.0.0

Quick reference when upgrading from 0.11.2:

- **Threshold**: replace `Threshold::Absolute(v)` with `v` and
  `Threshold::Relative(f)` with `f`; the plain `f32` is now the only
  form. Note raised defaults: ChESS `0` → `30`, Radon `0.01` → `0.30`.
- **Detection knobs**: move `nms_radius` and `min_cluster_size` from
  `ChessConfig` / `RadonConfig` into `DetectorConfig::with_detection(…)`.
- **Removed no-ops**: delete `RadonConfig.refiner`, `ChessConfig.descriptor_ring`,
  and any use of `RadonRefiner`, `RadonPeakConfig`, `RadonPeakRefiner`, or
  `RefinerKind::RadonPeak`.
- **`CornerDescriptor`**: remove reads of `.contrast` and `.fit_rms`; use
  `.response` for detection strength and `.axes[i].sigma` for per-axis
  confidence. Handle `axes: Option<…>` — it is `None` when orientation is
  disabled.
- **Moved internals**: `ChessParams` and `RefinerKind` are now in
  `chess_corners_core::unstable` / `chess_corners::low_level`; they remain
  accessible but carry no semver guarantee.
- **Sealed traits**: `DenseDetector` and `CornerRefiner` can no longer be
  implemented outside the crate; select a backend through `RefinerKind`.
- **Python**: `detect()` now returns a `Detections` object; replace
  column-index access (`arr[:, 2]`) with named attributes (`det.response`,
  `det.xy`, `det.angles`, `det.sigmas`).

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
