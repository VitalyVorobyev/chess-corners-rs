# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `RadonPeakRefiner` now implements the full Duda-Frese (2018) pipeline:
  image supersampling, response-map box blur, and Gaussian (log-space)
  peak fit. Default accuracy is ~0.04 px mean on clean anti-aliased
  chessboards, competitive with and often beating `SaddlePoint`. The
  module doc no longer calls this a "first cut" implementation.

- `RadonPeakConfig` field renames and additions: the previous
  `upsample` (response-grid density only) is replaced by
  `image_upsample` (controls both ray-sample spacing and
  response-grid density); new `response_blur_radius` and `peak_fit`
  fields gate the post-blur and Gaussian-fit stages. Defaults
  (`ray_radius = 2`, `patch_radius = 3`, `image_upsample = 2`,
  `response_blur_radius = 1`, `peak_fit = Gaussian`) match the paper.

- New cross-refiner accuracy integration test
  (`crates/chess-corners-core/tests/refiner_accuracy.rs`) prints a
  summary table for `RadonPeak` / `SaddlePoint` / `Forstner` across
  subpixel offset, blur, and noise sweeps.

- New unified accuracy+throughput benchmark
  (`crates/chess-corners/tests/refiner_benchmark.rs`) covering all
  five refiners — `CenterOfMass`, `Forstner`, `SaddlePoint`,
  `RadonPeak`, and (under the `ml-refiner` feature) the embedded
  ONNX `ML` refiner — over clean, blurred, and noisy sweeps with
  per-refiner timing. Run via
  `cargo test --release -p chess-corners --test refiner_benchmark \
   --features ml-refiner -- --nocapture --test-threads=1`.

## [0.6.0]

### Breaking

- **Default threshold now matches the ChESS paper's contract.**
  `ChessConfig::default()` now sets
  `threshold_mode = Absolute, threshold_value = 0.0`, and
  `ChessParams::default()` sets `threshold_abs = Some(0.0)`. The
  threshold comparison in `detect_corners_from_response` is now
  strict (`v > thr`), so the default detector accepts exactly the
  paper's "strictly positive `R`" criterion. The `threshold_rel`
  machinery remains available as an opt-in adaptive policy. Callers
  that relied on the previous 20%-of-max default should set
  `threshold_mode = Relative, threshold_value = 0.2` explicitly.

- **`CornerDescriptor` shape.** The single-axis `orientation: f32` field
  is replaced with a two-axis descriptor:

  ```rust
  pub struct CornerDescriptor {
      pub x: f32,
      pub y: f32,
      pub response: f32,
      pub contrast: f32,       // NEW: bright/dark amplitude from the fit
      pub fit_rms: f32,        // NEW: residual RMS of the two-axis fit
      pub axes: [AxisEstimate; 2],
  }

  pub struct AxisEstimate {
      pub angle: f32,          // radians, see convention below
      pub sigma: f32,          // 1σ angular uncertainty (radians)
  }
  ```

  `axes[0].angle ∈ [0, π)` and `axes[1].angle ∈ (axes[0].angle,
  axes[0].angle + π)` — rotating CCW from `axes[0]` to `axes[1]`
  traverses a **dark** sector of the corner. The two axes are **not**
  assumed orthogonal, which correctly captures projective warp and lens
  distortion.

- **Python `find_chess_corners` return shape:** `(N, 4)` →
  `(N, 9)` with columns
  `[x, y, response, contrast, fit_rms, axis0_angle, axis0_sigma,
  axis1_angle, axis1_sigma]`. See
  `crates/chess-corners-py/README.md` for documentation.

- **WASM `ChessDetector::detect` `Float32Array` stride:** 4 → 9 with
  the same column layout as the Python binding.

- **CLI JSON output per-corner schema:** `{x, y, response, orientation}`
  → `{x, y, response, contrast, fit_rms, axes: [{angle, sigma},
  {angle, sigma}]}`. Tools that consume `*.corners.json` need to be
  updated — `tools/detection/chesscorner.py` expects the new schema
  with a backwards-compatible `orientation` property that maps to
  `axes[0].angle`.

### Added

- **Parametric two-axis corner fit.** `fit_two_axes` replaces the 2nd-
  harmonic orientation kernel in `chess-corners-core/src/descriptor.rs`.
  A Gauss-Newton fit of `I(φ) = μ + A · tanh(β·sin(φ−θ₁)) ·
  tanh(β·sin(φ−θ₂))` over the 16-point ring yields both grid-axis
  directions independently plus per-axis 1σ uncertainty from the
  residual-scaled Hessian inverse. Typical runtime: ~1.5 µs per corner
  on Apple Silicon (M-series).

- **Optional pre-pipeline image upscaling.** New `ChessConfig.upscale`
  field enables an integer-factor bilinear upscaling stage ahead of
  the pyramid. Disabled by default; supports factors 2, 3, 4. Corner
  coordinates are always returned in the original input pixel frame
  (divided back by the factor), so callers do not need to be aware the
  stage ran. Motivating use case: low-resolution ChArUco crops where
  target corners fall inside the ChESS ring margin.

  Public surface: `UpscaleMode`, `UpscaleConfig::{disabled, fixed,
  effective_factor, validate}`, `UpscaleBuffers`, and
  `upscale::upscale_bilinear_u8` (reusable-buffer API suitable for
  frame-to-frame use).

  WASM: `ChessDetector::set_upscale_factor(u32)` accepts 0/1 (off) or
  2/3/4.

- **Benchmarks.**
  - `chess-corners-core/benches/descriptor_fit.rs` — Criterion
    microbenchmark of `corners_to_descriptors` at 64 / 256 / 1024
    corners.
  - `chess-corners/benches/upscale.rs` — throughput of
    `upscale_bilinear_u8` at 320×240, 640×480, 1280×720 for factors 2
    and 3 (~720 MiB/s on M-series).

- **Integration tests** for the upscaling pipeline at
  `crates/chess-corners/tests/upscale_pipeline.rs`.

### Changed

- Version bump across all workspace crates, the Python package, and the
  WASM package to `0.6.0`.
- `tools/plot_output.py` and `tools/plotting/chess_plot.py` now draw
  both grid axes per corner (two arrows, one per axis). The legacy
  `--no-orientation` flag still disables overlay arrows. Book overlay
  images regenerated with the new renderer.
- `tools/detection/chesscorner.py` schema updated to load the new JSON
  fields; exposes a backwards-compatible `orientation` property for
  callers that only need one axis.

## [0.5.0]

### Added

- `chess-corners-wasm` crate: WebAssembly bindings via `wasm-bindgen` for running
  the ChESS corner detector in the browser. Exposes a `ChessDetector` class with
  `detect`/`detect_rgba` (returns corners as `Float32Array`) and
  `response`/`response_rgba` (returns the raw ChESS response map). Supports
  webcam RGBA frames and grayscale input, reusable pyramid buffers for streaming,
  and configurable threshold, NMS radius, detector mode, pyramid levels, and refiner.
  Ships as an npm package via `wasm-pack` (~23 KB gzipped).
- Shared canonical algorithm config example at `config/chess_algorithm_config_example.json`.
- New ADR-007 documenting the Python-native public API with a private native module.

### Changed

- Version bump across all workspace crates, the Python package, and the WASM
  package to `0.5.0` for the upcoming release.
- `chess-corners` facade: CLI-only dependencies (`clap`, `anyhow`, `serde_json`,
  `tracing-subscriber`) are now feature-gated behind the `cli` feature,
  reducing the dependency tree for library consumers and WASM builds.
- Breaking public API redesign across Rust and Python:
  - `chess-corners::ChessConfig` is now a flat public config with explicit
    enums for detector mode, descriptor mode, threshold mode, and refiner kind.
  - `RefinerConfig` now always contains default-initialized
    `center_of_mass`, `forstner`, and `saddle_point` leaf configs.
  - The CLI now uses the same canonical flat algorithm schema via flattened
    `ChessConfig` fields in JSON.
- Public high-level ring selection now uses semantic modes instead of exposing
  radius-specific names:
  - Rust and Python use `DetectorMode::{Canonical,Broad}` /
    `DescriptorMode::{FollowDetector,Canonical,Broad}`
  - CLI and JSON use `canonical`, `broad`, and `follow_detector`
  - the numeric radius implementation detail remains in `chess-corners-core`
- `chess-corners-py` is now a mixed Rust/Python package:
  - public API lives in pure Python under `chess_corners`
  - native PyO3 extension moved behind the private `chess_corners._native` module
  - public Python config objects now provide typed enums/dataclasses, strict
    JSON helpers, readable printing, and proper wrapper signatures.
- Python example and docs now use Pillow plus the shared canonical config schema.
- Repository docs and guidance now use the `uv` + `.venv` workflow for local
  Python verification.
- The npm release workflow now uses GitHub OIDC trusted publishing instead of
  `secrets.NPM_TOKEN`, and builds with Node 24 to satisfy npm's current trusted
  publishing requirements.

## [0.4.2]

### Added

- `ChessConfig::multiscale()` as an explicit 3-level coarse-to-fine preset for
  projected, textured, or otherwise cluttered chessboard-like targets.

### Changed

- `ChessConfig::default()` / `ChessConfig::single_scale()` now remain in the
  stable single-scale regime, while multiscale usage is opt-in via
  `ChessConfig::multiscale()`.
- README, crate docs, and the multiscale example now point callers at the
  explicit multiscale preset instead of implying that `default()` is
  coarse-to-fine.
- Version bump across all workspace crates and the Python package to 0.4.2.
- GitHub Actions workflows now use `actions/checkout@v5`, and crates.io
  trusted publishing is pinned to `rust-lang/crates-io-auth-action@v1.0.4`
  for Node 24 compatibility.

### Fixed

- Rust crate publishing now checks the crates.io sparse index instead of the
  web API, avoiding false negatives from API `403` responses during release.
- Release reruns now treat `cargo publish` "already exists on crates.io index"
  responses as success, making both release paths idempotent.

## [0.4.1]

### Changed

- Version bump across all workspace crates and the Python package to 0.4.1.
- The crates.io release workflow for `vX.Y.Z` tags now verifies
  `box-image-pyramid`, `chess-corners-core`, `chess-corners-ml`, and
  `chess-corners` against the release tag before publishing.

### Fixed

- Rust crate publishing now happens in dependency order from the shared release
  tag: `box-image-pyramid`, then `chess-corners-core`, then
  `chess-corners-ml`, and finally `chess-corners`.
- The release workflow now waits for each published crate version to become
  visible on crates.io before attempting to publish dependent crates, fixing
  the CI failure where `box-image-pyramid` was not yet available before the
  chess crates were published.
- Release reruns and flows that publish `box-image-pyramid` first via the
  dedicated `box-image-pyramid-vX.Y.Z` tag now skip crates whose target version
  is already visible on crates.io instead of failing on an "already exists"
  publish error.

## [0.4.0]

### Changed

- Version bump across the breaking `chess-corners*` crates and the Python package to 0.4.0.
- `Corner` and `RefineResult` now use separate `x`, `y` fields instead of
  `xy: [f32; 2]`, matching `CornerDescriptor` for consistent coordinate
  representation across the codebase.
- `ResponseMap` fields (`w`, `h`, `data`) are now private; use `new()`,
  `width()`, `height()`, `data()`, `data_mut()` instead.
- `Roi` fields are now private with validated construction via `Roi::new()`
  (returns `None` if `x0 >= x1` or `y0 >= y1`) and `x0()`, `y0()`, `x1()`,
  `y1()` accessors.
- Deduplicated the ML and classic coarse-to-fine pipelines in `multiscale.rs`,
  extracting shared helpers (`RoiContext`, `refine_seed_in_roi`,
  `single_scale_detect`, `merge_and_describe`).
- `ForstnerConfig` default values are now documented with derivation rationale.

### Fixed

- ML refiner path no longer has a dead `#[cfg(feature = "rayon")]` block that
  silently skipped parallelization. The ML path now explicitly runs
  sequentially (required by `&mut MlRefinerState`).
- Embedded ONNX model extraction (`chess-corners-ml`) now compares bytes, not
  only file size, before reusing extracted temp files.

## [0.3.2]

### Added

- README diligence statement clarifying that AI coding assistants are implementation tools used alongside human validation and release quality gates.

### Changed

- Security audit workflow now uses `actions-rust-lang/audit` with read-only repository permissions and ignores `RUSTSEC-2024-0436`.
- Refreshed locked dependency versions in `Cargo.lock`.
- Version bump across workspace crates and the Python package to 0.3.2.

## [0.3.1]

### Added

- `chess-corners-ml` README + crates.io metadata (documentation/homepage/keywords).

### Changed

- ML refiner configuration is now internal-only; the public API uses ML entry points with built-in defaults.
- CLI uses `ml: true` to enable ML refinement; Python bindings mirror the simplified ML entry point.
- Version bump across all crates and the Python package to 0.3.1.

### Fixed

- `chess-corners-ml` now bundles the embedded ONNX model and fixtures inside the crate to support publishing.

## [0.3.0]

### Added

- ML-backed subpixel refiner (feature `ml-refiner`) with ONNX model support and explicit ML entry points (confidence output is ignored in this release).
- CLI support for ML refinement via `ml: true` in the config (feature-gated).
- Python bindings for full config structs, classic refiners, and the ML refiner entry point (using embedded defaults).
- Tracing diagnostics for ML refiner timings.
- Book + README coverage of ML refiner methodology, results, and usage.

### Changed

- Version bump across all crates and Python package to 0.3.0.

## [0.2.1]

### Added

- Python bindings via the `chess-corners-py` crate (`chess_corners` module).

### Changed

- Default `PyramidParams::num_levels` is now `1` instead of `3`. It improves detection stability with the default config by trading off some performance.
- Documentation updates covering Python usage (README, book, and crate docs).

## [0.2.0]

### Added

- Pluggable subpixel refinement trait (`CornerRefiner`) with three built-ins: center-of-mass (legacy default), Förstner, and saddle-point quadratic fit, plus reusable runtime selector (`RefinerKind`).
- Refinement selection now lives on `ChessParams::refiner`, shared across the core and facade crates.
- Refinement docs and README guidance on choosing/configuring refiners; core README updated with refined examples.
- Unit tests covering each refiner and a regression test ensuring the default matches prior COM behavior.

### Changed

- The `CornerDescriptor` structure simplifed: fields `phase` and `anysotropy` are gone. Only essential `x`, `y`, `response`, and `orientation` stay.
- CLI config schema simplified: `radius` / `descriptor_radius` are removed in favor of boolean `radius10` / `descriptor_radius10`.
- CLI config naming: `roi_radius` is renamed to `refinement_radius`.

## [0.1.2]

### Added

- Test images in `./testimages`
- Examples are tested and documented
- `tools/README.md` describing the Python benchmarking and visualization helpers.
- Basic contributor guidance (`CONTRIBUTING.md`) for running tests, docs, and feature-matrix checks.

### Changed

- Default `ChessParams` and `CoarseToFineParams` are adjusted to work in most practical cases out of the box
- CLI config examples and documentation now use numeric `radius` / `descriptor_radius` fields; these are validated and mapped to the underlying ChESS ring radii while still allowing boolean overrides from CLI flags.
- Public detector helpers (`find_chess_corners_u8`, `find_chess_corners`, `find_chess_corners_image`) are now marked `#[must_use]` and document their preconditions.
- Crate metadata now points to the `chess-corners-rs` docs site and exposes search keywords on crates.io.
