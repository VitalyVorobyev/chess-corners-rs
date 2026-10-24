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
shipped on every channel, followed by `1.3.1` (maintenance), `1.4.0`
(saddle-point position covariance, `UQ-00`) and `1.5.0` (per-corner position
covariance on the detector output, `UQ-05`), the current version
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
| CPP-07 | P3 | done | M5 | CPP-03 | Book Part X "C++ bindings". |

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
| DOCS-03 | P3 | todo | — | — | Tracing/profiling cookbook examples (pair with book Part IX). |
| DOCS-04 | P3 | todo | — | — | Worked `no_std` examples for `chess-corners-core`. |
| DOCS-05 | P3 | done | — | — | Regenerated the refiner benchmark plots under `book/src/img/bench/` without the RadonPeak series; fresh data confirms the chapter's refiner claims. |
| DOCS-06 | P3 | done | — | — | Book Part IX §9.4 says CenterOfMass has the lowest mean error up to σ = 8; `cv2.cornerSubPix` is lower at every σ ≤ 8 in `opencv_subpix_sweep.json` (0.046–0.086 px). Qualify the claim as "among the built-in refiners" (Codex review, #81). | Outcome: §7.3/§7.4/§7.6, Part V §5.4/§5.6 and README now say "among the built-in refiners"; verified against the regenerated sweep incl. `cv2.cornerSubPix`.
| DOCS-07 | P3 | todo | — | — | AGENTS.md §6 ("default settings must reproduce existing behavior unless the user opts in") conflicts with CLAUDE.md's rule to drop strictly dominated alternatives; #83 made re-centred SaddlePoint the default by decision. Reconcile: a default may change when the old behaviour is dominated, with a CHANGELOG note (Codex review, #83). |

## UQ — accuracy-aware detection  ·  post-1.0  ·  [roadmap](ROADMAP.md#post-10-program-accuracy-aware-detection-uq-)

| ID | Pri | Status | Release | Deps | Task |
|----|-----|--------|---------|------|------|
| UQ-00 | P1 | done | 1.4.0 | — | Saddle-point refiner reports its position covariance per unit noise variance (`RefineResult`, typed). Normal-matrix inverse precomputed per radius, so the covariance adds no per-corner solve. Asserting scatter test (PR #74). |
| UQ-01 | P1 | done | 1.5.0 | UQ-00 | Bias/precision benchmark per refiner. Grid: subpixel phase, blur, Gaussian and signal-dependent noise, cell size, contrast, skew. Metrics: signed bias, precision σ, RMSE, and z-score calibration of the predicted σ about the mean and about the truth. Output `reference/refiner-accuracy.md`. **Gates the 1.5.0 API shape.** Outcome: bias dominates at σₙ 1–3 for all refiners unless blur ≳ 2 px (SaddlePoint σ̂₀² 3–220 with the noise-only covariance); σ_pred is calibrated for scatter (z_prec ≈ 1, incl. signal-dependent noise); SaddlePoint bias is a removable pull toward the window centre — recommend re-centred refit (UQ-07) before exposing covariance. |
| UQ-02 | P1 | done | 1.5.0 | UQ-01 | Re-measure the noise rows in `reference/refiner-comparison.md`, `book/src/img/bench/bench_sweep.json` and book Part IX §9.4 with the fixed noise helper (they were measured at ~1.3× the stated σ). | Outcome: all noise rows re-measured with the fixed helper (the OpenCV mirror had the same bug and was fixed); CenterOfMass now leads σ≤8, ML only at σ=12.
| UQ-03 | P2 | done | 1.5.0 | UQ-01 | Per-image noise estimate (cheap estimator on one pyramid level, or a caller-supplied value) so covariance comes out in px². Outcome: `estimate_noise_sigma_u8` (core) — product of third differences, χ²₉ gate on every other 4×4 orthogonal-polynomial component of total degree ≥ 3 (catches diagonal edges), exact threshold on an in-memory sample of ~8 k windows, per-band fixed points with a structure test (8× image-wide, 3× band median), windows near black/white skipped by their mean. Synthetic boards (cells 5–12, rotations 0–45°, 60° shear, blur 0–2.5): within 4% for σ ≥ 2 and 5% for σ ≥ 1 except near-decline cells (≤ 12%) and dense sharp cells at σ 8 (45° cell 6: +106%). Real test images (no truth): additive within 0.86–1.03 for σₐ 1–4 and stable across gate settings (≤ 5%); 0.055–0.095 ms. `DetectorConfig::noise_sigma` overrides it. See `reference/refiner-accuracy.md`. |
| UQ-04 | P3 | todo | 1.6.0+ | UQ-01 | Position covariance for Förstner and CenterOfMass. Deferred: both are bias-dominated at realistic noise ([UQ-01](reference/refiner-accuracy.md)), so a noise covariance would overstate their precision. |
| UQ-05 | P2 | done | 1.5.0 | UQ-03, UQ-07 | SaddlePoint position covariance (px²) on `CornerDescriptor`, with a `noise_sigma` config, in Python / WASM / C (`cc_abi_version` 5) / CLI / schema. Landed: floor 0.002 px (minimises the worst \|ln σ̂₀²\| on cells 8–12), noise variance ≥ 1/12, end-to-end σ̂₀² 0.90–1.17 where detections match the truth; `None` for upscaled input ([refiner-accuracy](reference/refiner-accuracy.md)). Book Part VIII "Uncertainty" (figures from `tools/render_uncertainty_overlays.py`) documents the model, the calibration evidence, the regimes where it fails, and how to consume it. |
| UQ-06 | P3 | todo | 1.6.0 | UQ-03 | One noise model for orientation σ across RingFit and DiskFit, re-checked with z-scores. |
| UQ-07 | P1 | done | 1.5.0 | UQ-01 | SaddlePoint refits on a patch re-centred at its previous estimate until it converges (default behaviour). Removes the pull toward the window centre (0.12 → ~0.004 px noise-free). The covariance is derived for the converged estimator (fixed-point sensitivity, correlated resampled pixels) and validated with the UQ-01 harness until σ̂₀² ≈ 1. Outcome: noise-free RMS bias 0.119 → ≤ 0.001 px (sharp, axis-aligned), 0.150 → 0.006 (sheared); σ̂₀² 0.99–1.15 axis-aligned / ≤ 1.36 sheared on cells 8–12 at σₙ 1–4 (was up to 222 / 501); acceptance unchanged on every cell; ~4× the cycles of a single fit. Review follow-up: damped Newton steps on the window offset (no limit cycles; cap hits 29 / 14 / 104 → 0 on the review regimes), and a non-converged corner returns the integer-grid fit without covariance. |
| UQ-08 | P2 | todo | ongoing | UQ-05 | Realistic ground truth: calib-synth / etendue renders, and an end-to-end check that calibration-rs reaches σ̂₀ ≈ 1 with detector covariances. Absorbs `ML-01` and `ML-02`. |
| UQ-09 | P2 | done | 1.5.0 | — | CenterOfMass returns `Accepted` at the unchanged seed when its window has no positive response (36–100% of corners at cell 5). Report a non-accepted status instead. Outcome: returns `IllConditioned`; unreachable from the detector (seeds are positive peaks; 0 of 93 M refinements across test images, synthetic cells 5–12, single-scale/pyramid/upscale/ML), detector output bit-identical. |
| UQ-10 | P3 | done | 1.5.0 | UQ-01 | Bias harness (`refiner_bias_precision.rs`): remove each phase mean's sampling variance with that phase's own variance, not the pooled σ. Only matters where acceptance or scatter varies by phase (low SNR); the σₙ 1–3 conclusions are unaffected (Codex review, #82). Outcome: no published number moved (largest `rms_bias` change 8e-5 px, no `bias_dominates` flip). |
| UQ-11 | P3 | todo | 1.6.0+ | UQ-07 | Re-centred SaddlePoint is worse than the single fit on 5-px cells with 40–60° shear and blur ≥ 1.5 (60°, blur 2.5: 0.29 vs 0.19 px RMS; noise-free bias 0.21 → 0.26 at 40/200, 0.27 → 0.50 at 90/150). Non-convergence there already falls back to the single fit; the remaining loss is at converged corners whose true position is not a fixed point (weak signal, neighbouring corners in the window). No per-patch guard tested separates them from cell ≥ 8 corners (see `reference/refiner-accuracy.md`); remaining options: pick `radius: 1` from a cell-size estimate, or document the regime. |
| UQ-12 | P1 | done | 1.5.0 | UQ-07 | Before cutting 1.5.0, re-time on a quiet machine: `refiners` Criterion bench (SaddlePoint per-corner cost; docs quote ~250 ns from cycle counts) and the timing fields of `book/src/img/bench/bench_sweep.json` and `throughput_vs_accuracy` (regenerated at load ≈ 15, which inflates every refiner by 30–40%). | Outcome: re-timed at load ≈ 3.3. Criterion ns/corner: CenterOfMass 18, Förstner 62, SaddlePoint 225 (docs ~20/~60/~250 kept); sweep harness 16/57/184 ns, ML 165 µs, cornerSubPix 2.5 µs (harness tables updated for CenterOfMass and SaddlePoint). |
| UQ-13 | P3 | todo | 1.6.0+ | UQ-03 | `estimate_noise_sigma_u8` over-reads on dense sharp rotated boards at high noise (cell 6, σ 8: +106% at 45°, +24% at 30°), which over-states the covariance (conservative). Find a structure statistic that separates this case, or decline there. |
| UQ-14 | P3 | todo | 1.6.0+ | UQ-09 | CenterOfMass returns `Accepted` at the integer seed when one pixel carries all positive response in its window (up to 100% of accepted truth-seeded corners at cell 5, blur 0). The result is a valid centroid but carries no subpixel information. The detector's `min_cluster_size` makes it rare there; decide whether a minimum-support criterion is worth it. |
| UQ-15 | P1 | done | 1.5.0 | UQ-07 | Re-centred SaddlePoint keeps 0.01–0.02 px noise-free RMS error on boards rotated 2–15° (or 45°) from the pixel grid (cells 8 and 12, blur ≤ 1.5; ≤ 0.004 grid-aligned, 0.002–0.004 at 20–30°), far above the 0.002 px floor: end to end, a board rotated 8° gives σ̂₀² ≈ 10 / 3.6 / 1.6 / 1.1 at σₙ 1 / 2 / 4 / 8 (sharp), 2.6 / 1.4 / 1.1 / 1.05 at blur 1.5 (`tools/render_uncertainty_overlays.py --only calibration`; three renderers agree to 0.0005 px). The `refiner-accuracy` evidence covered only grid-aligned and one-axis-horizontal boards. Find the source (bilinear resampling of oblique edges is the first suspect), then remove the bias or model it in the floor. Documented as a limit in book Part VIII §8.4.1. Outcome: two causes (`reference/refiner-accuracy.md` § Rotated boards). (1) Sampling aliasing of ideally sharp edges: the fixtures blurred *after* pixel sampling, which keeps it at every blur; no estimator on the same pixels removes it (bilinear, Keys, Lagrange, raw-pixel moving-window fits, pre-smoothing, LS weights, radius 1–3 all keep 7–35 mpx at 2–8°), and with the lens blur before pixel integration it is gone from σ_psf 0.5 (8°: 20.5 → 0.5 mpx). (2) With a lens, a 5–9 mpx residual at 30–45° from the bilinear resampler's third moment (→ `UQ-16`). Fixtures moved to `optical_chessboard` (testutil, exact lens-blurred pixel response; also the book renderer); `POSITION_FLOOR_PX` 0.002 → 0.006 by the minimax of \|ln σ̂₀²\| over rotation 0–45° × cells 8/12 × lens σ 0.5–2.5 × σₙ 1–8 + shear: worst 1.40 → 0.56 (σ̂₀² 0.61–1.75; grid-aligned 0.61 at σₙ 1, 0.85 at σₙ 2). Sharp-optics boards stay overconfident (8°: σ̂₀² 6.1 at σₙ 1), documented. |
| UQ-16 | P2 | todo | 1.6.0 | UQ-15 | Re-centre SaddlePoint with a 4-point Lagrange cubic resampler (reproduces cubics, so the resampling kernel's third moment is 0) instead of bilinear, which leaves 5–9 mpx noise-free error at 30–45° under a lens blur of 0.5–1 px. Prototype (16 cached integer-centred fits, derivative-weight Newton slope, covariance with 7×7 tap overlaps; checked against finite differences): end-to-end worst \|ln σ̂₀²\| on the lens-blur grid (σ ≥ 0.8) 1.06 → 0.23 at f = 0.002, every angle ≤ 1.3 mpx at lens σ 0.75–1.5. Costs: 2.5× per corner as prototyped (≈ 600 vs 240 ns; needs a perf pass, e.g. computing the 16 fits from shared separable row sums), predicted σ +4–6%, worse with ideally sharp optics (30°: σ̂₀² 1.1 → 3.2; 60° shear: 1.2 → 2.0 at σₙ 1). Once it lands, re-run the `UQ-15` floor minimax: the floor should shrink toward ~0.002–0.003 px, removing most of the grid-aligned under-confidence. Prototype prototype on branch `proto/uq-16-cubic-resampler` (not for merge). |
| UQ-17 | P3 | todo | 1.6.0 | UQ-15 | `testutil::Lattice::nearest_corner` rounds the two lattice coordinates independently, which on a sheared lattice can pick a corner that is not the Euclidean nearest (60° axes, board coords (0.49, 0.49) → (0, 0) although (1, 0) is closer). Harmless for the current harnesses, which match detections within a fraction of a pixel, but wrong ground truth for loosely localized points: search the neighbouring integer candidates by image-space distance (Codex review, #88). |
| UQ-18 | P1 | done | 1.6.0 | UQ-05 | Per-corner position covariance for the Radon detector. Radon's 3-point peak fit on the blurred, 2× supersampled response is bias-dominated on synthetic boards (cell 8, noise-free: 0.007–0.17 px RMS/axis by blur and rotation, vs ≤ 0.009 for SaddlePoint), so a peak-fit covariance would be dishonest (as `UQ-04`). Opt-in `RadonConfig::refiner: RadonRefiner { PeakFit (default), SaddlePoint(SaddlePointConfig) }` (additive) refines Radon peaks with SaddlePoint on the input image, falling back to the peak fit (no covariance) when re-centring does not converge; public `estimate_noise_sigma_u8` in the facade, Python, WASM, C/C++. The default stays the peak fit until `UQ-22` (real-data audit) decides. Outcome: on camera-formed synthetic boards (cells 4–12, 0/8/45° + 60° shear, lens σ 0.5–2.5, σₙ 0/2/8) Radon+SP equals ChESS+SP wherever both detect, never worse than the peak fit at cells ≥ 5 (geo-mean RMS ratio 0.38), fallback ≤ 0.1% of matched corners there; worse at 4 px cells (60° shear, blur 1.5, σₙ 2: 0.12 → 0.17 px, match 1.00 → 0.81 — the window spans neighbours, `UQ-11`), documented; tightening `max_offset` does not fix it (0.5: match 0.94, but costs accuracy elsewhere). σ̂₀² (estimated σₙ, `fig_calibration` grid, cell 8; `render_uncertainty_overlays.py --detector radon`) 0.55–1.29, identical to ChESS+SP, plus 0.96–1.03 on 60° shear blur 2.5 where ChESS finds nothing; cell 5: 0.77–1.38. Cost +0.5 µs/corner (≤ 3% of a Radon frame on `testimages/`). C ABI 6 (`CC_REFINER_PEAK_FIT`, per-strategy refiner validation). `DenseDetector::refines_on_image` deprecated. |
| UQ-19 | P3 | todo | 2.0.0 | UQ-18 | Lift the subpixel refiner out of `ChessConfig` / `RadonConfig` into one shared stage on `DetectorConfig` (orientation is already one), with per-detector validity in the type (CenterOfMass needs the ChESS response map; Radon seeds are already subpixel, so its fallback is the peak fit). Breaking: `ChessConfig::refiner` / `RadonConfig::refiner` move, bindings and schema follow. Remove `DenseDetector::refines_on_image` (always `true` since 1.6.0) in the same release, after marking it `#[deprecated]` in the 1.6.0 release PR (a minor-level change that needs the version bump). |
| UQ-20 | P2 | todo | 1.6.0+ | — | Radon emits edge-midpoint false peaks on sharp boards rotated ~45° (cell 12, lens σ 0.5, noise 0: ~420 per 200² frame, 6 px from the corners, response ~1/3 of real corners; none at lens σ 2.5; the ChESS preset emits none). Unchanged by `UQ-18` (SaddlePoint rejects them as non-saddles, so they keep the peak fit). Options: raise the default relative threshold, an axial-ray symmetry check, or drop Radon peaks the saddle-point refiner finds not to be saddles. False detections are unrecoverable downstream. |
| UQ-21 | P2 | todo | 1.6.0+ | — | `radon_multiscale` returns far more corners than single-scale Radon on two test images (`small.png` 1091 vs 75, `large.png` 6992 vs 279; `mid.png` 78 vs 77), almost all weak (median response ~10× below matched corners) and without a converged saddle. Suspected cause: `threshold` is relative to each ROI's own maximum on the coarse-to-fine path. Measure, then make the per-ROI threshold relative to the coarse frame. |
| UQ-22 | P1 | todo | 1.6.0+ | UQ-18 | Decide Radon's default refiner (`PeakFit` vs `SaddlePoint`) from real calibration data: the `UQ-18` evidence comes from one synthetic generator (straight edges, Gaussian lens blur, white noise) that may favour re-centred estimators. Compare per-corner reprojection residuals and end-to-end σ̂₀² with both refiners on real captures (`UQ-08`). |
| UQ-23 | P1 | done | 1.6.0 | UQ-18 | Robustness audit of the refiner conclusions beyond one synthetic generator ([design](design/uncertainty-v2.md)). Outcome ([refiner-robustness](reference/refiner-robustness.md)): on an independent supersampled generator (homography, distortion, non-Gaussian/asymmetric PSFs, illumination, Poisson, Bayer, JPEG) SaddlePoint is most accurate in 18/19 conditions (Radon wins under strong shading) and on all 4 real calibration sets (cross-validated). Its bias is 0.02–0.10 px under illumination gradients and asymmetric PSFs, σ̂₀² 1.5–22 there; on real sets the predicted σ sits at the floor and does not track residuals. Re-centred Förstner: wontfix (75% less bias but still 6× SaddlePoint's, ill-conditioned at blur ≥ 1.5). Noise estimator misreads demosaiced images and declines on JPEG. |
| UQ-24 | P1 | todo | 1.6.0 | UQ-18 | `Uncertainty` stage on `DetectorConfig` (`Off` / `Propagated`, default `Propagated`); exact-Jacobian noise covariance for CenterOfMass, Förstner and the Radon peak fit; per-localizer systematic model from window offset and edge sharpness ([design §4.1](design/uncertainty-v2.md)). |
| UQ-25 | P1 | done | 1.6.0 | UQ-18 | Noise estimate on demosaiced (Bayer) and JPEG images. Outcome: no change shipped ([refiner-robustness § Noise estimate](reference/refiner-robustness.md)). The calibrating σ is the noise in SaddlePoint's 5×5 slope band (σ̂₀² 0.85–1.15 when supplied, every noise family); the third-difference operator measures a different band, reads demosaic structure on Bayer (3.3× σ at σ 1, 0.5× at σ 8) and declines on JPEG; the band is not identifiable from 4×4 windows, and a coarse-band estimator reads structure on dense boards and real images. Document: supply `noise_sigma` for demosaiced/JPEG input (`UQ-28`); a coarse-band estimator with a robust noise anchor is open research (`UQ-32`). |
| UQ-26 | P1 | done | 1.6.0 | UQ-23 | Per-corner asymmetry term ([design §4.2](design/uncertainty-v2.md)): fit mislocation + illumination-ramp components to the patch's antisymmetric part. Prototype and gate (design §8). Outcome ([asymmetry-term](reference/asymmetry-term.md)): the design's \((g\cdot d)(S-m)\) 5×5 form fails (shading σ̂₀² 2.1/5.8); a 9×9 fit with a multiplicative ramp \((g\cdot d)S\) passes on the synthetic grid with exceptions (asymmetric 0.94–1.43, symmetric 0.72–0.88, gamma 0.63; shading ±15% at noise 1: 1.99) and is neutral on real held-out calibration (board/model error dominates). Correction: no-go (worse on 3/4 real sets). Multi-radius spread: no-go (blind to shading). Shipping the tier is a user decision. |
| UQ-27 | P1 | todo | 1.6.0 | UQ-24, UQ-26 | Ship the asymmetry tier (and bias correction if gated in) in core, facade and bindings. |
| UQ-28 | P1 | todo | 1.6.0 | UQ-24–27 | Book Part VIII rewrite around the v2 error model with real-data evidence; fix the 1.5.0 statements flagged in `refiner-robustness.md`; Parts IV, V, IX; README; CHANGELOG. |
| UQ-29 | P2 | todo | 1.6.0 | UQ-24 | Demo on lab-ui: covariance for every localizer, tier selector, noise readout; `EllipseSet` item colours (lab-ui #135). |
| UQ-30 | P2 | todo | ongoing | — | Absolute validation data with detector-separable truth: repeated captures (precision), a precision target on a stage or physically based renders. Absorbs `UQ-08`. |
| UQ-31 | P3 | todo | 1.6.0 | — | WASM `rgba_to_gray` truncates instead of rounding (some gray levels low by one; noise estimate ~3% high). Changes `detectRgba` output. |
| UQ-32 | P3 | todo | — | UQ-25 | Noise estimate in SaddlePoint's slope band for correlated noise: a coarse-band estimator calibrates flat and sparse boards (±4%) but needs a noise anchor that is not the quietest windows (dense boards, clipped cells and real images read 5–100× high). |

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
