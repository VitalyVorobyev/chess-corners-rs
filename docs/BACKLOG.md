# Backlog

Task registry for the program in [`ROADMAP.md`](ROADMAP.md). Design detail
lives in the linked `design/` docs; this file is the task list.

**Legend.** ID `<WS>-NN`, `WS ∈ {PERF, API, SITE, CPP, DEBT, DOCS, SWEEP, SOLID}`
(+ carried-over `ML/ALGO/PY`). Priority `P0`(blocker)–`P3`(nice-to-have).
Status `todo | in-progress | blocked | done | wontfix`. **Milestone** links to
the ROADMAP; **Deps** lists prerequisite IDs.

**Status.** All milestone work (M1–M6) is done; the 1.0 surface is hardened and
coherent. No `P0`/`P1` blocks the 1.0 freeze. The only remaining
release-critical item is `API-09` (version bump + tag + publish), deferred by
choice. Everything under *Post-1.0 / future* is `P2`/`P3`, off the 1.0 critical
path. The pre-release campaign on `release/v1.0.0-prep` (see
`docs/API_REVISION.md`) is folded into this state.

## PERF — profiling & optimization  ·  M2  ·  [design](design/perf-profiling.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| PERF-01 | P1 | done | M2 | — | Criterion microbench: ChESS ring kernel (scalar + SIMD). |
| PERF-02 | P1 | done | M2 | — | Bench: Radon SAT build, i64 vs u32 element. |
| PERF-03 | P2 | done | M2 | — | Bench RingFit orientation-fit solver. |
| PERF-04 | P2 | done | M2 | — | Bench DiskFit vs RingFit. |
| PERF-05 | P2 | done | M2 | — | Bench NMS scaling (radius sweep). |
| PERF-06 | P1 | done | M2 | — | Allocation audit: hot paths allocation-free; multiscale per-seed vecs kept (optional `*_into` variants → PERF-13). |
| PERF-07 | P1 | done | M2 | — | Flamegraph/profiling automation (`tools/profile.sh`); doc residual → DOCS-03. |
| PERF-08 | P1 | done | M2 | PERF-01..05 | CI bench-regression gate (`tools/perf/` + `.github/workflows/bench-gate.yml`): same-runner head-vs-base `critcmp`, fail on >2% median drift. |
| PERF-09 | P2 | done | M2 | PERF-08 | Committed perf baseline snapshot (`tools/perf/baseline-metrics.json`); the live gate is head-vs-base, not vs this file. |
| PERF-10 | P2 | done | M2 | PERF-01 | Vectorized ChESS SIMD μₗ/write-back tail (4×→~7×, bit-identical); remaining levers → PERF-13. |
| PERF-12 | P2 | done | M2 | — | Added soft-edge + warped fixtures; confirmed DiskFit lazy gate short-circuits on typical corners (not a flat 47× tax). |
| PERF-13 | P3 | todo | — | PERF-12 | Optional further optimizations if profiling justifies: Radon angular sampling @up=2; NMS O(W·H) scan; rayon 2D tiling for ChESS; SIMD prefix-sum for Radon SAT. |

## API — v1.0 stabilization  ·  M3  ·  [design](design/api-v1.0.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| API-01 | P0 | done | M3 | — | Dropped `contrast`/`fit_rms` from `CornerDescriptor` across Rust/Py/WASM/CLI + snapshot. Ripple → TOOL-01. |
| API-02 | P1 | done | M3 | — | Lifted `nms_radius`/`min_cluster_size` into a shared `DetectionParams` on `DetectorConfig.detection`; `with_detection` builders added. |
| API-03 | P1 | done | M3 | — | Moved `ChessParams`/`RefinerKind` to `chess_corners_core::unstable`; demoted Radon primitives to `pub(crate)`. (`unstable` move superseded by DEBT-01; Radon demotions stand.) |
| API-04 | P2 | done | M3 | — | Feature-gated `ChessRefiner::Ml`; removed the silent-mapping helper. |
| API-05 | P1 | done | M3 | — | `.pyi` parity confirmed complete and factory names match Rust; added a stub-vs-runtime parity guard test. |
| API-06 | P1 | done | M3 | — | Sealed `DenseDetector`/`CornerRefiner`; added `#[non_exhaustive]` to remaining public configs; documented MSRV (stable ≥ 1.88, `simd` = nightly). |
| API-07 | P2 | done | M3 | API-06 | Documented core→binding unknown-variant → default mapping (forward-compat); caller input stays strict; pinned WASM enum discriminants. |
| API-08 | P0 | done | M3 | API-01..07 | Advisory `cargo-semver-checks` CI vs `v0.11.2` (`continue-on-error`); flips to blocking at API-09. |
| API-09 | P0 | blocked | M3 | API-08, M4, M5, M6 | **Release act (deferred by choice).** Bump 0.11.2→1.0.0, move `[Unreleased]`→`docs/changelog/1.0.0.md`, tag + publish (crates.io/PyPI/npm), finalize vcpkg (CPP-05), flip semver-checks to blocking. Awaiting the go decision. |

## SITE — GitHub Pages  ·  M4 (dep M3)  ·  [design](design/site-architecture.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SITE-01 | P1 | done | M4 | — | Landing page (`.github/pages/index.html`) adapted to chess-corners (4 cards → /book//api//demo//performance/). |
| SITE-02 | P1 | done | M4 | SITE-01,03,04 | Reworked `docs.yml` to assemble `public/` = landing + api + book + demo + perf. |
| SITE-03 | P1 | done | M4 | M3 | Vite + React + Bun WASM demo (`demo/`) over `@vitavision/chess-corners`; live detect + overlays + controls; Playwright-verified. |
| SITE-04 | P2 | done | M4 | — | `gen-perf-data.sh` + `perf_overlay` example emit `performance/data.json` + overlays; rewrote the perf page. |
| SITE-05 | P2 | done | M4 | SITE-02 | `scripts/build-site.sh` reproduces the deployed tree locally. |
| SITE-06 | P3 | done | M4 | SITE-02 | `/book/`-move audit; README badges repointed to the site root + demo. |

## CPP — C++ bindings via vcpkg  ·  M5 (dep M3)  ·  [design](design/cpp-vcpkg-bindings.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| CPP-01 | P2 | done | M5 | M3 | `chess-corners-capi` crate: flat `cc_config` + presets, `cc_detect_u8`, lib-owned `cc_result`, panic-trapped boundary; Rust parity + C smoke tests. |
| CPP-02 | P2 | done | M5 | CPP-01 | `cbindgen.toml` + generator bin (with `--check` drift mode) → committed `include/chess_corners.h`. |
| CPP-03 | P2 | done | M5 | CPP-02 | Header-only `chess_corners.hpp` (C++17): value types, RAII `ResultGuard`, throwing `detect()`, compile/run ABI guard. |
| CPP-04 | P2 | done | M5 | CPP-01 | `CMakeLists.txt` + package config: `find_package(chess-corners CONFIG)`; static + shared; relocatable pkg-config `.pc`. |
| CPP-05 | P3 | done (draft) | M5 | CPP-04 | Overlay vcpkg port (`ports/chess-corners/`). Release-finalization (real `v1.0.0` tag + SHA512, cross-platform `vcpkg install`, registry PR) tracked in `ports/README.md`; not install-verified here. |
| CPP-06 | P3 | done | M5 | CPP-03 | Self-contained C++ example + C smoke wired as CTest; `.github/workflows/cpp.yml` matrix (static + shared). |
| CPP-07 | P3 | done | M5 | CPP-03 | Book Part IX "C++ bindings". |

## DEBT — design hardening before the freeze  ·  M6

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| DEBT-01 | P1 | done | M6 | API-03 | Deleted `chess_corners_core::unstable` (root-public fns required `ChessParams`, which lived only there — incoherent); promoted the genuinely-needed types to the crate root, demoted the rest to `pub(crate)`. Visibility-only; detection bit-stable. |
| DEBT-02 | P1 | done | M6 | DEBT-01 | Deleted the facade `chess_corners::low_level` escape hatch; exposed lowering as `DetectorConfig::chess_params()` / `radon_detector_params()` / `coarse_to_fine_params()`. |
| DEBT-03 | P2 | done | M6 | — | Retired the `too_many_arguments` `#[allow]` cluster by bundling `(img,w,h)` into `ImageView`; the 3 justified allows kept with comments. |
| DEBT-04 | P3 | done | M6 | — | Replaced the disk-sector argmax `-1.0f32` sentinel with `Option` (make-illegal-states-unrepresentable); bit-exact. |
| DEBT-05 | P3 | done | M6 | — | Split the facade `config.rs` into a cohesive `config/` module; public paths byte-identical. |
| DEBT-06 | P2 | todo | — | — | Parked (post-1.0). py/wasm config duplication is largely inherent to PyO3 vs wasm-bindgen; a shared codegen layer isn't worth its complexity before 1.0. Revisit if a third binding or a schema change forces it. |
| DEBT-07 | P3 | todo | — | — | Parked (post-1.0, likely wontfix). Strategy-dispatch parallel `match` arms — only two variants and new detectors are out of scope, so refactoring dispatch now is YAGNI. |
| DEBT-08 | P3 | todo | — | — | Parked (optional). Splitting `detect/radon/response.rs` by variant can hurt shared-math readability; defer unless the file grows further. |

## SWEEP — dev-history/internal reference cleanup  ·  M3

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SWEEP-01 | P2 | done | M3 | — | Swept `.rs` doc comments + book + README for dev-history/lineage/origin refs (fixes in `tools/book/plot_benchmark.py`). |
| SWEEP-02 | P3 | done | M3 | — | Fixed stale `tools/perf_bench.py`; non-doc comment audit otherwise clean. |
| SWEEP-03 | P3 | done | — | — | Fixed sibling-repo references in `.claude/agents/{deep,quick}-implementer.md`. |

## SOLID — DRY / cohesion cleanup  ·  M2

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SOLID-01 | P2 | done | M2 | — | New zero-dep `chess-corners-testutil` crate homes the shared fixtures/blur/noise helpers (was duplicated up to 8×); byte-identical. |
| SOLID-02 | P3 | todo | — | — | Evaluate `Refiner` enum dispatch boilerplate (`refine/mod.rs`) — refactor or accept. |
| SOLID-03 | P3 | done | M2 | SOLID-01 | Merged duplicate synthetic-chessboard generators into `chess-corners-testutil`. |
| SOLID-04 | P2 | done | — | — | Collapsed the dual `threshold_rel`/`threshold_abs` sentinel to one field per detector. |
| SOLID-05 | P2 | done | — | — | Split `detect/chess/detect.rs` into `neighbors`/`merge`; split the Python and WASM `config.rs` into per-config-type modules. |

## SKILL — reusable tooling  ·  continuous  ·  user-level, project-agnostic

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SKILL-01 | P2 | done | M1 | — | Added the `rust-cpp-bindings` skill beside `rust-python-bindings`/`rust-wasm-bindings`. |
| SKILL-02 | P3 | todo | — | SKILL-01 | Candidate `rust-docs-site` skill — assemble a unified Pages site from the M4 pattern. |
| SKILL-03 | P3 | todo | — | — | Candidate `repo-knowledge-base` skill — restructure a `docs/` folder from the M1 pattern. |

## DOCS — knowledge base

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| DOCS-01 | P1 | done | M1 | — | Restructure `docs/` into a KB (ROADMAP + BACKLOG + algorithm index + design docs). |
| DOCS-02 | P3 | todo | — | — | Keep `design/algorithms-index.md` current as algorithms change (recurring). |
| DOCS-03 | P3 | todo | — | — | Tracing/profiling cookbook examples (pair with book Part VIII). |
| DOCS-04 | P3 | todo | — | — | Worked `no_std` examples for `chess-corners-core`. |
| DOCS-05 | P3 | done | — | — | Regenerated the refiner benchmark plots under `book/src/img/bench/` without the RadonPeak series; fresh data confirms the chapter's refiner claims. |

## Post-1.0 / future (not blocking 1.0)

Carried-over and nice-to-have work, none on the 1.0 critical path. Together with
the scattered `P3` `todo` items above (`PERF-13`, `SOLID-02`, `SKILL-02/03`,
`DOCS-02..05`, `DEBT-06..08`), all deferred until after the 1.0 release.

| ID | Pri | Status | Deps | Task |
|----|-----|--------|------|------|
| ML-01 | P2 | todo | — | Validate ML refiner accuracy on real-world calibration images (synthetic-only now). |
| ML-02 | P2 | todo | API-04 | Integrate ML confidence (`conf_logit`) into the pipeline (currently ignored). |
| ML-03 | P2 | todo | PERF-10 | Optimize ML inference (~23 ms / 77 corners is too slow for real-time). |
| ALGO-01 | P3 | todo | — | Adaptive per-corner refiner selection by local image context. |
| PY-01 | P3 | todo | — | Python batch processing with `PyramidBuffers` reuse across frames. |
| PY-02 | P2 | done | API-01 | `Detector.detect()` returns a structure-of-arrays `Detections` (`.xy`/`.response`/`.angles`/`.sigmas`; `None` axes when orientation is off) instead of a dense `(N,7)` array. |
| TOOL-01 | P3 | done | API-01, PY-02 | Orientation bench reads the Python `Detections` SoA by name; dropped the obsolete `fit_rms`/`contrast` metrics. |
| PERF-11 | P2 | done | API-06, M5 | **Evaluated — keeping nightly `std::simd`.** No stable backend replaces it: `wide` regresses aarch64 ~1.9× (no NEON >128-bit) and `pulp` can't express the pyramid `(a+b+c+d+2)>>2` bit-exactly. Stable scalar/autovec is the supported portable baseline; nightly `simd` stays the optional high-performance path. |
| RADON-01 | P3 | done | — | Removed the no-op `RadonRefiner` config across Rust/Py/WASM/CLI; Radon subpixel stays its Gaussian peak fit. |
| RADON-02 | P3 | done | RADON-01 | Removed `RefinerKind::RadonPeak`, `RadonPeakConfig`, `refine/radon_peak.rs`, the C tag, and all references. |
| RADON-03 | P1 | done | — | Recalibrated the Radon default `threshold_rel` 0.01→0.30 (mid.png plateau midpoint); consolidated to one `RadonDetectorParams::DEFAULT_THRESHOLD_REL` const. |

## Closed

| ID | Resolution |
|----|------------|
| TASK-002 | DiskFit antipodal axis-slot inversion **not reproducible** on 0.11.0; permanent guard added (`crates/chess-corners/tests/orientation_slot_parity.rs`). |

## Cross-workstream dependencies

- **M3 (API freeze) gates M4 and M5** — the demo, rustdoc, and C ABI bind the
  frozen `CornerDescriptor` (no `contrast`/`fit_rms`).
- **M2 → M3** — reshape hot paths before the surface freezes.
- **PERF-09 → SITE-04** — the performance page consumes the perf baseline.
- **SOLID-01 → PERF benches** — share one synthetic-board/blur helper.
