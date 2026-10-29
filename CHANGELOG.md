# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

2.0 is a major release. To upgrade from 1.x, see
[Migrating from 1.x to 2.0](book/src/part-12-migrating-to-2.0.md), which
has before and after code for every binding.

### Changed

- Subpixel refinement is one optional, detector-independent stage. Each
  detector keeps the localizer that works on its own response map, the
  ChESS center of mass (`ChessConfig::center_of_mass`) and the Radon peak
  fit (`RadonConfig::peak_fit`), and `DetectorConfig::refiner:
  Option<ImageRefiner>` adds an image refiner for either detector.
  `None`, the default, keeps the detector's own localizer. On ChESS an
  image refiner replaces the center of mass, starts from the integer peak
  and drops a corner it rejects; on Radon it starts from the peak fit and
  keeps it where it does not converge. The refiner survives `with_chess` /
  `with_radon` strategy switches. Outputs of every configuration that
  maps across are unchanged bit for bit. Replaces `ChessRefiner`,
  `RadonRefiner`, `ChessConfig::refiner` and `RadonConfig::refiner`:
  - Rust: `ChessRefiner::SaddlePoint(c)` / `RadonRefiner::SaddlePoint(c)`
    → `.with_refiner(ImageRefiner::SaddlePoint(c))`;
    `ChessRefiner::CenterOfMass(c)` → no refiner plus
    `.with_chess(|s| s.center_of_mass = c)`; `RadonRefiner::PeakFit` → no
    refiner.
  - Python: `ImageRefiner.saddle_point(c)` set on `cfg.refiner` or passed
    to `cfg.with_refiner(...)`; `cfg.without_refiner()`;
    `cfg.strategy.chess.center_of_mass`.
  - JS: `cfg.withRefiner(ImageRefiner.saddlePoint())` /
    `ImageRefiner.saddlePointWith(c)` in place of `withChessRefiner` /
    `withRadonRefiner`; `cfg.withoutRefiner()`;
    `cfg.strategy.chess.centerOfMass`.
  - JSON: the refiner moves from `strategy.chess.refiner` /
    `strategy.radon.refiner` to a top-level `"refiner"` (`null` or
    `{"saddle_point": {...}}`); the center-of-mass tuning is
    `strategy.chess.center_of_mass`.
  - CLI: `--chess-refiner` → `--refiner none|saddle_point`.
  - core: `ChessParams::refiner` → `ChessParams::center_of_mass`;
    `RefinerKind` holds only image refiners.
- The uncertainty setting carries its noise level:
  `Uncertainty::{Off, Propagated { noise: NoiseLevel }}` with
  `NoiseLevel::{Estimate, Known(f32)}`, both in `chess-corners`.
  `DetectorConfig::noise_sigma` and `with_noise_sigma` are removed; a
  `Known(σ)` that is negative or not finite is still rejected by
  `Detector::new`. Positions and covariances are unchanged.
  - Python and JS: `Uncertainty` is an immutable value,
    `Uncertainty.off()` / `Uncertainty.propagated(noise_sigma=σ)` (JS
    `Uncertainty.propagated(σ)`), in place of the `OFF` / `PROPAGATED`
    (`Off` / `Propagated`) enum plus `noise_sigma` (`noiseSigma`).
  - JSON: `"propagated"` → `{"propagated": {"noise": "estimate"}}`, or
    with `"noise_sigma": σ` → `{"propagated": {"noise": {"known": σ}}}`.
  - core: `Uncertainty` moved to `chess-corners`;
    `Refiner::with_uncertainty(tier)` → `Refiner::with_covariance(bool)`.
- Each orientation axis reports `fit_std_error`, the residual-based
  least-squares standard error of the axis fit in radians, in place of
  `sigma`: a fit-quality score for ranking and gating axes, not a
  calibrated error bound. Both methods now report this quantity. `RingFit`
  values are the previous ones divided by 1.0–1.25; `DiskFit` computes it
  over the disk pixels, and its two axes now differ. The 10° minimum when
  `DiskFit` and `RingFit` disagree, π as "no information", and the axis
  angles are unchanged. A gate tuned on the previous values rejects about
  half as many axes; tighten it by about ×0.65 to keep its rejection
  rate. Renamed: Rust `AxisEstimate::sigma` → `fit_std_error` (and core
  `AxisFitResult::sigma_theta1` / `sigma_theta2` →
  `fit_std_error_theta1` / `fit_std_error_theta2`), Python
  `Detections.sigmas` → `fit_std_errors`, JS `axis0_sigma` /
  `axis1_sigma` slots → `axis0_fitStdError` / `axis1_fitStdError` (same
  offsets), C `cc_axis::sigma` → `fit_std_error`, C++ `Axis::sigma` →
  `fit_std_error`, CLI JSON output `axes[].sigma` →
  `axes[].fit_std_error`.
- C ABI version 8; rebuild C and C++ code against the new header.
  `cc_config::refiner` is a `cc_image_refiner_t`: `CC_IMAGE_REFINER_NONE`
  (`0`, replacing `CC_REFINER_CENTER_OF_MASS` and `CC_REFINER_PEAK_FIT`)
  or `CC_IMAGE_REFINER_SADDLE_POINT` (`1`, for either strategy).
  `CC_UNCERTAINTY_OFF` is now `0` and `CC_UNCERTAINTY_PROPAGATED` `1`, so
  a zero-initialised `cc_config` selects no covariance and no image
  refiner, like every preset; `has_noise_sigma` / `noise_sigma` are read
  only under `CC_UNCERTAINTY_PROPAGATED`. `cc_axis::sigma` is
  `fit_std_error`. C++: `Config::uncertainty` is a
  `chess_corners::Uncertainty` (`Uncertainty::off()`,
  `Uncertainty::propagated(sigma)`) in place of the tag plus
  `Config::noise_sigma`.
- Every config loader rejects an unknown key at every level, with an
  error that names it: Rust serde (`DetectorConfig` and every type in
  it), the CLI, Python `DetectorConfig.from_dict` / `from_json`, and JS
  `DetectorConfig.fromJson`, which ignored unknown keys before. A stale
  1.x key such as `noise_sigma` or a per-strategy `refiner` fails to
  load instead of being dropped. The JSON Schema sets
  `additionalProperties: false` on every object.
- The CLI config file nests the detector settings under `"detector"`,
  beside `image`, `output_json`, `output_png` and `log_level`; the
  JSON summary echoes them under `detector` instead of `algorithm`.
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
  `--chess-refiner forstner` choice and the `"forstner"` refiner in the
  JSON config. Use the saddle-point image refiner
  (`ImageRefiner::saddle_point()`, `"saddle_point"`,
  `CC_IMAGE_REFINER_SADDLE_POINT`), which is more accurate on real
  calibration images and on almost every synthetic condition.
- The ML refiner: `ChessRefiner::Ml`, the `ml-refiner` feature of
  `chess-corners` and of the Python package, Python `ChessRefiner.ml()`,
  JS `ChessRefiner.withMl()`, and the CLI `"ml"` setting. Use
  `ImageRefiner::SaddlePoint`: it is more accurate in 15 of the 16
  synthetic conditions of the refiner benchmark (the exception is the
  heaviest noise row, 0.100 against 0.093 px) and on rotated boards, where
  the ML model was 0.64–0.90 px RMS against at most 0.057 px, and it is
  about 800× cheaper per corner. A build that enabled `ml-refiner` drops
  the feature. The `chess-corners-ml` crate is no longer developed; 0.13.0
  stays on crates.io.
- The `covariance_per_noise_variance` fields of `Corner` and
  `RefineResult`: read `noise_covariance_per_unit_variance()`, or
  `position_covariance(noise_sigma)` for the full covariance in px².
- `chess_corners_core::attach_radon_peak_covariance`: use
  `DenseDetector::attach_peak_covariance` on `RadonDetector`.
- `DenseDetector::refines_on_image`: both detectors always refine on the
  image; drop the call.

### Fixed

- `RingFit`: the per-axis fit errors (`fit_std_error`, previously
  `sigma`) now follow their axes. When the fit relabelled the two axes to
  put them in canonical order, the two values stayed in place and were
  reported on the wrong axis.

### Documentation

- A migration guide from 1.x to 2.0 in the book, with before and after
  code for Rust, Python, JavaScript, C and C++, and the JSON
  configuration.

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
