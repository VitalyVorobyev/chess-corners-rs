# Backlog

Task registry for the program in [`ROADMAP.md`](ROADMAP.md). Design detail
lives in the linked `design/` docs; this file is the task list.

**Legend.** ID `<WS>-NN`, `WS ∈ {PERF, API, SITE, CPP, DEBT, DOCS, SWEEP, SOLID, UQ}`
(+ carried-over `ML/ALGO/PY`). Priority `P0`(blocker)–`P3`(nice-to-have).
Status `todo | in-progress | blocked | done | wontfix`. **Milestone** links to
the ROADMAP; **Deps** lists prerequisite IDs.

**Status.** All milestone work (M1–M6) is done; the 1.0 surface is hardened and
coherent. **`1.1.0` shipped to every channel**: `chess-corners`,
`chess-corners-core` and `box-image-pyramid` at `1.1.0` plus
`chess-corners-ml` at `0.13.0` on crates.io, `chess-corners` `1.1.0` on PyPI,
and `@vitavision/chess-corners` `1.1.0` on npm — which also cleared the
`DEBT-11` residual that had left npm stranded at `0.11.2` through the whole
1.0.0 cycle. The release carried the `image` codec-free dependency fix for
\#68 (`DEBT-09`) and the ONNX runtime bump + MSRV `1.91` (`DEBT-10`).
`cargo-semver-checks` is now a genuinely blocking CI gate: flipping it off
advisory exposed that its `package:` filter had always been malformed, so the
job had never actually checked anything. `1.2.0` (facade ROI detection) and
`1.3.0` (detector-config JSON Schema + WASM JSON config path) have since
shipped on every channel, followed by `1.3.1` (maintenance) and `1.4.0`
(saddle-point position covariance, `UQ-00`), the current version
(`chess-corners-ml` unchanged at `0.13.0`). One residual remains open — `CPP-05`, the vcpkg port, which carries
a real SHA512 but still targets `1.1.0` and has never been install-verified on
any platform. Everything under *Post-1.0 / future*
is `P2`/`P3`, off the release-critical path. The pre-release campaign on
`release/v1.0.0-prep` is folded into this
state.

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
| API-08 | P0 | done | M3 | API-01..07 | Advisory `cargo-semver-checks` CI vs `v0.11.2` (`continue-on-error`); flipped to blocking once `1.0.0` became the crates.io baseline (1.1.0 prep). |
| API-09 | P0 | done | M3 | API-08, M4, M5, M6 | **Release act.** Bumped 0.11.2→1.0.0, moved `[Unreleased]`→`docs/changelog/1.0.0.md`, tagged `v1.0.0`, and published to crates.io + PyPI. npm publish did not happen — no `wasm-v1.0.0` tag was pushed, so `@vitavision/chess-corners` stayed at `0.11.2` (carried forward as `DEBT-11`); vcpkg SHA512 finalize also carried forward (`CPP-05`). |
| API-10 | P2 | done | — | — | Facade ROI detection entry point: `Detector::detect_u8_roi`/`detect_roi` + `Roi` re-export, single-scale, ChESS and Radon. |
| API-11 | P2 | done | — | — | Detector-config JSON Schema (`schemars` feature, `cargo xtask emit-schemas`, `schemas/detector_config.json` shipped in the npm package) and WASM `DetectorConfig.fromJson` / `toJson` / `default_detector_config_json`. |

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
| CPP-05 | P3 | in-progress | M5 | CPP-04 | Overlay vcpkg port (`ports/chess-corners/`). Manifest targets `1.1.0` (not yet bumped for `1.2.0`/`1.3.0`) with the real source-tarball SHA512, so it is installable. Still **never install-verified**: cross-platform `vcpkg install` (both linkages) and the registry PR remain open — see `ports/README.md`. |
| CPP-06 | P3 | done | M5 | CPP-03 | Self-contained C++ example + C smoke wired as CTest; `.github/workflows/cpp.yml` matrix (static + shared). |
| CPP-07 | P3 | done | M5 | CPP-03 | Book Part IX "C++ bindings". |

## DEBT — design hardening before the freeze  ·  M6

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| DEBT-01 | P1 | done | M6 | API-03 | Deleted `chess_corners_core::unstable` (root-public fns required `ChessParams`, which lived only there — incoherent); promoted the genuinely-needed types to the crate root, demoted the rest to `pub(crate)`. Visibility-only; detection bit-stable. |
| DEBT-02 | P1 | done | M6 | DEBT-01 | Deleted the facade `chess_corners::low_level` escape hatch; exposed lowering as public `DetectorConfig::chess_params()` / `radon_detector_params()`. `coarse_to_fine_params()` was subsequently withdrawn to `pub(crate)` (multiscale internals aren't part of the stable API) — see `crates/chess-corners/src/config/detection.rs`. |
| DEBT-03 | P2 | done | M6 | — | Retired the `too_many_arguments` `#[allow]` cluster by bundling `(img,w,h)` into `ImageView`; the 3 justified allows kept with comments. |
| DEBT-04 | P3 | done | M6 | — | Replaced the disk-sector argmax `-1.0f32` sentinel with `Option` (make-illegal-states-unrepresentable); bit-exact. |
| DEBT-05 | P3 | done | M6 | — | Split the facade `config.rs` into a cohesive `config/` module; public paths byte-identical. |
| DEBT-06 | P2 | todo | — | — | Parked (post-1.0). py/wasm config duplication is largely inherent to PyO3 vs wasm-bindgen; a shared codegen layer isn't worth its complexity before 1.0. Revisit if a third binding or a schema change forces it. |
| DEBT-07 | P3 | todo | — | — | Parked (post-1.0, likely wontfix). Strategy-dispatch parallel `match` arms — only two variants and new detectors are out of scope, so refactoring dispatch now is YAGNI. |
| DEBT-08 | P3 | todo | — | — | Parked (optional). Splitting `detect/radon/response.rs` by variant can hurt shared-math readability; defer unless the file grows further. |
| DEBT-09 | P2 | done | — | — | Set the `image` workspace dependency to `default-features = false`, dropping bundled codec decoders the crate doesn't need; fixes #68. |
| DEBT-10 | P2 | done | — | — | Bumped the `tract-onnx` ONNX runtime and raised MSRV to `1.91` (`rust-version` in `[workspace.package]`). |
| DEBT-11 | P1 | done | — | — | Published `@vitavision/chess-corners` to npm. `wasm-v1.1.0` closed the gap left by the never-pushed `wasm-v1.0.0`; npm `latest` moved `0.11.2` → `1.1.0`. |

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
| DOCS-06 | P3 | done | — | — | Book Part VIII §7.4 says CenterOfMass has the lowest mean error up to σ = 8; `cv2.cornerSubPix` is lower at every σ ≤ 8 in `opencv_subpix_sweep.json` (0.046–0.086 px). Qualify the claim as "among the built-in refiners" (Codex review, #81). | Outcome: §7.3/§7.4/§7.6, Part V §5.4/§5.6 and README now say "among the built-in refiners"; verified against the regenerated sweep incl. `cv2.cornerSubPix`.
| DOCS-07 | P3 | todo | — | — | AGENTS.md §6 ("default settings must reproduce existing behavior unless the user opts in") conflicts with CLAUDE.md's rule to drop strictly dominated alternatives; #83 made re-centred SaddlePoint the default by decision. Reconcile: a default may change when the old behaviour is dominated, with a CHANGELOG note (Codex review, #83). |

## UQ — accuracy-aware detection  ·  post-1.0  ·  [roadmap](ROADMAP.md#post-10-program-accuracy-aware-detection-uq-)

| ID | Pri | Status | Release | Deps | Task |
|----|-----|--------|---------|------|------|
| UQ-00 | P1 | done | 1.4.0 | — | Saddle-point refiner reports its position covariance per unit noise variance (`RefineResult`, typed). Normal-matrix inverse precomputed per radius, so the covariance adds no per-corner solve. Asserting scatter test (PR #74). |
| UQ-01 | P1 | done | 1.5.0 | UQ-00 | Bias/precision benchmark per refiner. Grid: subpixel phase, blur, Gaussian and signal-dependent noise, cell size, contrast, skew. Metrics: signed bias, precision σ, RMSE, and z-score calibration of the predicted σ about the mean and about the truth. Output `reference/refiner-accuracy.md`. **Gates the 1.5.0 API shape.** Outcome: bias dominates at σₙ 1–3 for all refiners unless blur ≳ 2 px (SaddlePoint σ̂₀² 3–220 with the noise-only covariance); σ_pred is calibrated for scatter (z_prec ≈ 1, incl. signal-dependent noise); SaddlePoint bias is a removable pull toward the window centre — recommend re-centred refit (UQ-07) before exposing covariance. |
| UQ-02 | P1 | done | 1.5.0 | UQ-01 | Re-measure the noise rows in `reference/refiner-comparison.md`, `book/src/img/bench/bench_sweep.json` and book Part VIII §7.4 with the fixed noise helper (they were measured at ~1.3× the stated σ). | Outcome: all noise rows re-measured with the fixed helper (the OpenCV mirror had the same bug and was fixed); CenterOfMass now leads σ≤8, ML only at σ=12.
| UQ-03 | P2 | done | 1.5.0 | UQ-01 | Per-image noise estimate (cheap estimator on one pyramid level, or a caller-supplied value) so covariance comes out in px². Outcome: `estimate_noise_sigma_u8` (core) — product of third differences, gated by orthogonal third-order statistics, per-intensity-band truncated variance; within 5.5% of the realised σ for σ ≥ 1 on every synthetic board it reports on (cells 5–12, blur 0–2.5, axis-aligned and 60° shear), within 6% under signal-dependent noise (root mean variance); declines on full-frame sharp sheared 5 px cells at σ ≤ 4 and over-estimates them at σ 8 (+45%). ~0.04 ms on 0.4–3 MP test images (0.23 ms on a low-noise one). `DetectorConfig::noise_sigma` overrides it. See `reference/refiner-accuracy.md`. |
| UQ-04 | P3 | todo | 1.6.0+ | UQ-01 | Position covariance for Förstner and CenterOfMass. Deferred: both are bias-dominated at realistic noise ([UQ-01](reference/refiner-accuracy.md)), so a noise covariance would overstate their precision. |
| UQ-05 | P2 | in-progress (book Part pending) | 1.5.0 | UQ-03, UQ-07 | SaddlePoint position covariance (px², plus a small constant model-error floor — measured by UQ-07: 0.015 px per axis brings σ̂₀² within 0.05 of the scatter-only value on every cell ≥ 8 px fixture with σ_emp < 0.3 px; 0.03 px at cell 5, low contrast, blur 2.5) on `CornerDescriptor`, scaled by s² when coordinates are mapped back from upscale or pyramid levels (noise variance is per level); Python / WASM / C APIs (`cc_abi_version` bump) and JSON schema; a book Part on uncertainty. Status: descriptor field, `noise_sigma` config, Python / WASM / C (`cc_abi_version` 5) / CLI / schema landed; pyramid needs no rescaling (final refinement is on the base image), upscaled input reports none. Floor 0.002 px (minimises the worst |ln σ̂₀²| on cells 8–12, harness and end to end; 0.015 drove σ̂₀² to 0.17) and noise variance ≥ 1/12: end-to-end σ̂₀² 0.89–1.16 wherever detections match the truth (`reference/refiner-accuracy.md`). Remaining: book Part on uncertainty. |
| UQ-06 | P3 | todo | 1.6.0 | UQ-03 | One noise model for orientation σ across RingFit and DiskFit, re-checked with z-scores. |
| UQ-07 | P1 | done | 1.5.0 | UQ-01 | SaddlePoint refits on a patch re-centred at its previous estimate until it converges (default behaviour). Removes the pull toward the window centre (0.12 → ~0.004 px noise-free). The covariance is derived for the converged estimator (fixed-point sensitivity, correlated resampled pixels) and validated with the UQ-01 harness until σ̂₀² ≈ 1. Outcome: noise-free RMS bias 0.119 → ≤ 0.001 px (sharp, axis-aligned), 0.150 → 0.006 (sheared); σ̂₀² 0.99–1.15 axis-aligned / ≤ 1.36 sheared on cells 8–12 at σₙ 1–4 (was up to 222 / 501); acceptance unchanged on every cell; ~4× the cycles of a single fit. Review follow-up: damped Newton steps on the window offset (no limit cycles; cap hits 29 / 14 / 104 → 0 on the review regimes), and a non-converged corner returns the integer-grid fit without covariance. |
| UQ-08 | P2 | todo | ongoing | UQ-05 | Realistic ground truth: calib-synth / etendue renders, and an end-to-end check that calibration-rs reaches σ̂₀ ≈ 1 with detector covariances. Absorbs `ML-01` and `ML-02`. |
| UQ-09 | P2 | todo | 1.5.0 | — | CenterOfMass returns `Accepted` at the unchanged seed when its window has no positive response (36–100% of corners at cell 5). Report a non-accepted status instead. |
| UQ-10 | P3 | done | 1.5.0 | UQ-01 | Bias harness (`refiner_bias_precision.rs`): remove each phase mean's sampling variance with that phase's own variance, not the pooled σ. Only matters where acceptance or scatter varies by phase (low SNR); the σₙ 1–3 conclusions are unaffected (Codex review, #82). Outcome: no published number moved (largest `rms_bias` change 8e-5 px, no `bias_dominates` flip). |
| UQ-11 | P3 | todo | 1.6.0+ | UQ-07 | Re-centred SaddlePoint is worse than the single fit on 5-px cells with 40–60° shear and blur ≥ 1.5 (60°, blur 2.5: 0.29 vs 0.19 px RMS; noise-free bias 0.21 → 0.26 at 40/200, 0.27 → 0.50 at 90/150). Non-convergence there already falls back to the single fit; the remaining loss is at converged corners whose true position is not a fixed point (weak signal, neighbouring corners in the window). No per-patch guard tested separates them from cell ≥ 8 corners (see `reference/refiner-accuracy.md`); remaining options: pick `radius: 1` from a cell-size estimate, or document the regime. |
| UQ-12 | P1 | todo | 1.5.0 | UQ-07 | Before cutting 1.5.0, re-time on a quiet machine: `refiners` Criterion bench (SaddlePoint per-corner cost; docs quote ~250 ns from cycle counts) and the timing fields of `book/src/img/bench/bench_sweep.json` and `throughput_vs_accuracy` (regenerated at load ≈ 15, which inflates every refiner by 30–40%). |

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
