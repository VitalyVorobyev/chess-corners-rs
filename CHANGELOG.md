# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
