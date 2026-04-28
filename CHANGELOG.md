# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance

- **Radon detector pipeline rewritten for parallelism.** The
  Duda–Frese path landed in 0.7.0 with no parallelism and no SIMD.
  Every hot stage now has a feature-gated parallel and/or SIMD
  variant: row-parallel `compute_response` plus a portable
  `Simd<i64, 8>` inner kernel under `simd`; row-parallel
  `box_blur_inplace`, `upsample_bilinear_2x`, and
  `detect_corners_from_radon` (NMS + peak-fit) under `rayon`;
  `rayon::join` across the four mutually-independent
  `build_cumsums` directions. Output is deterministic across all
  feature combinations.

- **`merge_corners_simple` is now O(N) typical via spatial grid.**
  The naive O(N²) pairwise scan was the dominant pipeline cost on
  Radon-detected frames where candidate counts run into the
  thousands. Cell size equals the merge radius, so each incoming
  corner only consults a 3×3 cell neighbourhood. Output is
  order-equivalent to the old scan (verified via a randomized
  equivalence test).

  Cumulative speedup on the Radon pipeline (vs. 0.7.0 scalar
  reference, 8-core M-class CPU):

  | Bench                                  | 0.7.0   | This rev   | Speedup |
  |----------------------------------------|--------:|-----------:|--------:|
  | `radon_response` 1920×1080 up=2        | 130.6 ms| 28.0 ms    | 4.7×    |
  | `radon_pipeline` 1920×1080 synth       | 692 ms  | 162 ms     | 4.3×    |
  | `radon_pipeline` `large.png`           | 393 ms  | 188 ms     | 2.1×    |
  | `radon_pipeline` `mid.png`             | 47 ms   | 19.6 ms    | 2.4×    |

  See `book/src/part-07-benchmarks.md` §7.7.1 for the full table
  and the `tools/profile.sh` invocations behind it.

- **Side win: scalar `box_blur_inplace` is 5–28% faster** purely
  from the row-major rewrite, regardless of `rayon` / `simd`.

### Added

- Full-pipeline Criterion benches:
  `crates/chess-corners/benches/radon_pipeline.rs`,
  `crates/chess-corners/benches/chess_pipeline.rs`, and
  `crates/chess-corners-core/benches/refiners.rs` (per-refiner
  ns/corner microbench). Each runs on synthetic chessboards plus
  the `testimages/{small,mid,large}.png` real frames so absolute
  timings are comparable across machines.

- `tools/profile.sh` — wrapper around `cargo-flamegraph` and
  `samply` with `chess` / `radon` / `refiner <kind>` / `samply`
  subcommands. Pairs with a new
  `crates/chess-corners/examples/profile_target.rs` so the
  profiler sees a clean hot loop without harness noise. Outputs
  land under `testdata/out/profiles/`.

- `tools/perf_bench.py --radon` — exercises the Radon detector
  alongside the existing ChESS `multi`/`single` configs. Reads
  `config/config_radon.json` if present, otherwise patches
  `config_single.json` with `detector_mode = "radon"`.

- `crates/chess-corners/tests/perf_accuracy_guard.rs` — recall /
  precision / p95-error floors per detector preset, calibrated
  against the scalar reference. Runs under
  `cargo test --workspace --all-features` so optimization patches
  that change correctness fail at test time rather than silently
  drifting in production.

### Fixed

- SIMD `compute_response` inner loop now exits one chunk earlier
  (`x + RADON_LANES < interior_end` instead of `<=`) so the
  diagonal-negative ray's `lo` lookup never reads past the row at
  the right edge. The scalar tail then handles the boundary pixel,
  matching the original kernel bit-for-bit. Reported by Codex on
  PR #49 (P1). A new test
  (`simd_kernel_matches_scalar_at_every_interior_pixel`) sweeps
  several widths to lock the boundary down.

- `upsample_bilinear_2x` and `box_blur_inplace` no longer panic on
  zero-extent inputs (`w == 0` or `h == 0`). The row-parallel
  rewrites used `chunks_mut(w)` / `par_chunks_mut(w)`, which
  panic on zero chunk size; both now early-return to preserve the
  pre-perf no-op contract. Reported by Codex on PR #49 (P2).

### Documentation

- New §7.7.1 in `book/src/part-07-benchmarks.md` covering the
  Radon detector pipeline: stage-by-stage rayon / SIMD speedups,
  the memory-bound caveat that makes SIMD stack weakly with rayon
  on M-class CPUs, and an honest note on which optimization
  unlocked which gain. §7.11 expanded with `cargo bench` /
  `tools/profile.sh` reproducibility commands.

## [0.7.0] - 2026-04-26

### Highlights

- New whole-image Radon corner detector for hard frames (heavy
  blur, low contrast, small cells).
- Typed `ChessConfig` surfaces in WASM and Python — every facade
  field is reachable through type-safe getters/setters; nested
  edits (`cfg.refiner.forstner.max_offset = 2.0`) work directly,
  no JSON across the FFI.
- Improved ML refiner (`v4`); now competitive with classical
  refiners under noise.

### Changed

- **npm package renamed to `@vitavision/chess-corners`** (was
  `chess-corners-wasm`). The exported JS API is identical —
  migrate by updating the dependency name in `package.json` /
  your `import` statements. The legacy package is deprecated on
  npm with a tombstone release pointing at the new name.

- `RadonPeakRefiner` now implements the full Duda-Frese (2018)
  pipeline (image supersampling, response-map box blur, Gaussian
  peak fit). Default accuracy is ~0.04 px mean on clean
  anti-aliased boards. `RadonPeakConfig` field renames: the
  previous `upsample` is replaced by `image_upsample` (controls
  both ray-sample spacing and response-grid density); new
  `response_blur_radius` and `peak_fit` fields gate the post-blur
  and Gaussian-fit stages.

### Added

- **`DetectorMode::Radon`** — the whole-image Duda-Frese Radon
  detector is now a first-class detector mode on `ChessConfig`,
  with a new `ChessConfig::radon()` preset and a
  `radon_detector: RadonDetectorParams` field. Useful when ChESS
  fails (heavy blur, low contrast, cells smaller than
  `~2·ring_radius`). See
  [`docs/detector-comparison.md`](docs/detector-comparison.md)
  for when to pick each detector. Available across Rust, Python
  (`DetectorMode.RADON`, `ChessConfig.radon()`), and WASM
  (`detector.set_detector_mode("radon")` and the new typed
  config).

- **Radon heatmap public API.** The dense
  `(max_α S_α − min_α S_α)²` Radon response is exposed on every
  layer for visualization and downstream tooling: Rust
  `chess_corners::radon_heatmap_u8` (and `radon_heatmap_image`
  under `image`); Python `chess_corners.radon_heatmap()` returning
  a `(H, W) float32` numpy array; WASM `ChessDetector.radon_heatmap*`
  family with `radon_heatmap_scale()` for input-to-heatmap pixel
  alignment. The interactive demo gains a "Radon heatmap" overlay
  toggle.

- **Native typed `ChessConfig` in WASM.** Every public facade
  field is now reachable through `#[wasm_bindgen]` typed wrappers
  (`ChessConfig`, `RefinerConfig`, `RadonDetectorParams`,
  `UpscaleConfig`, refiner subconfigs, plus enums) with
  TypeScript types in the generated `.d.ts`. New
  `ChessDetector.withConfig(cfg)` / `getConfig()` /
  `applyConfig(cfg)`. Nested edits propagate without round-trip
  (`cfg.refiner.kind = X`, `cfg.radonDetector.rayRadius = 5`,
  …). The legacy `set_*` shortcut methods continue to work.

- **Native typed PyO3 `ChessConfig` surface.** Replaces the
  previous Python dataclass + JSON-string FFI with native PyO3
  classes. The Python user surface is unchanged — attribute
  access, classmethod factories, `to_dict`/`from_dict`/`to_json`/
  `from_json`/`pretty`/`print`, identity-comparable enum members
  — but `find_chess_corners(image, cfg)` no longer serializes
  through JSON across the FFI boundary. The JSON-string fallback
  is retained for one release.

- Feature `radon-sat-u32` (opt-in) switches the Radon detector's
  summed-area-table element type from `i64` to `u32`. Halves SAT
  memory and widens SIMD lanes at the cost of a ~16 MP image-size
  cap (`255·W·H ≤ u32::MAX`).

- ML refiner **v4** (`chess_refiner_v4.onnx`) — retrained on a
  mixed (hard-cell + tanh-saddle) distribution so it handles both
  regimes. v4 is the first ML refiner to land inside the shipping
  band on clean data (~0.09 px mean) and **wins under heavy
  noise** (~0.10 px at σ=10). It does not beat `RadonPeak` on
  clean data — see
  [`docs/refiner-comparison.md`](docs/refiner-comparison.md) for
  the trade-off.

- Book Part V rewritten with a six-refiner comparison
  (CenterOfMass, Förstner, SaddlePoint, RadonPeak, ML v4,
  `cv2.cornerSubPix`).

### Fixed

- Out-of-bounds indexing in the Radon detector's box-blur on
  non-square inputs (reproduced on every real-world frame).

- Release-build panic when `radon_detector.image_upsample ≥ 3`.
  Values are now clamped to the supported set `{1, 2}` at the
  entry points.

- Under `radon-sat-u32`, `radon_response_u8` now validates
  `255·W·H ≤ u32::MAX` up front instead of silently wrapping
  the cumsum accumulators in release builds.

- `chess-corners` now re-exports `PeakFitMode` so downstream
  consumers can set `RadonPeakConfig.peak_fit` without depending
  directly on `chess-corners-core`.

### Removed

- Obsolete `chess_refiner_v2.onnx` and the interim
  `chess_refiner_v3.onnx`. Neither shipped in a released version;
  `v4.onnx` is now the only embedded artifact.

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
