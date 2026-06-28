# Backlog

Task registry for the program in [`ROADMAP.md`](ROADMAP.md). Design detail
lives in the linked `design/` docs; this file is the task list.

**Legend.** ID `<WS>-NN`, `WS ∈ {PERF, API, SITE, CPP, DOCS, SWEEP, SOLID}`
(+ carried-over `ML/ALGO/PY`). Priority `P0`(blocker)–`P3`(nice-to-have).
Status `todo | in-progress | blocked | done | wontfix`. **Milestone** links to
the ROADMAP; **Deps** lists prerequisite IDs.

## PERF — profiling & optimization  ·  M2 (leads)  ·  [design](design/perf-profiling.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| PERF-01 | P1 | done | M2 | — | Criterion microbench: ChESS ring kernel (scalar + SIMD) — `benches/chess_response.rs`. Baseline: scalar ≈155 / SIMD ≈621 Mpix/s @1024² (3.7–4.0×). |
| PERF-02 | P1 | done | M2 | — | Bench: Radon SAT, i64 vs u32 @1024² — `benches/radon_response.rs`. u32 win +9–16% @up=1, collapses @up=2. |
| PERF-03 | P2 | done | M2 | — | Bench RingFit fit (`descriptor_fit.rs::orientation_fit`): 2.59 µs@corner (robust-grid path; hard-edge worst case). |
| PERF-04 | P2 | done | M2 | — | Bench DiskFit vs RingFit: 121.6 µs@corner = 47× RingFit when the full disk runs (hard-edge worst case; lazy gate never trips → PERF-12). |
| PERF-05 | P2 | done | M2 | — | Bench NMS scaling (`nms_scaling.rs`) r=1/2/4/8 @1024²: 766→1646 µs. Sub-quadratic (64× area = 2.15× time); O(W·H) scan dominates. |
| PERF-06 | P1 | done | M2 | — | Allocation audit: inner loops allocation-free (RadonPeak scratch reused; response kernels use reused buffers). Multiscale loop allocates 2 small `Vec<Corner>`/seed (`multiscale.rs:358,371`) but response-patch dominates — keep; optional `*_into` is PERF-10. |
| PERF-07 | P1 | done | M2 | — | Flamegraph/profiling automation already exists: `tools/profile.sh` (cargo-flamegraph + samply over `profile_target`). Residual: document in book Part VIII (→ DOCS-03). |
| PERF-08 | P1 | todo | M2 | PERF-01..05 | CI bench gate: baseline compare, fail on >2% p95 regression |
| PERF-09 | P2 | todo | M2 | PERF-08 | Capture baseline `metrics.json` (feeds SITE-04) |
| PERF-10 | P2 | done | M2 | PERF-01 | Done (lever a): vectorized ChESS SIMD μₗ/write-back tail — SIMD **4×→~7×** (1074 Mpix/s @1024²), **bit-identical** output (verified, all equivalence/snapshot/accuracy-guard tests green). Remaining optional levers → PERF-13. |
| PERF-12 | P2 | done | M2 | — | Added soft-edge + warped fixtures (`benches/common`). Confirmed soft corner rel_rms≈0.01 → RingFit fast path (2.60→0.73 µs) + DiskFit lazy-gate short-circuit (123→0.73 µs, ~169×); warped (~61° sep) → DiskFit full disk 158 µs. DiskFit is not a flat 47× tax. |
| PERF-13 | P3 | todo | — | PERF-12 | Optional further optimizations (only if profiling on PERF-12 fixtures justifies): Radon angular sampling @upsample=2; NMS O(W·H) scan; rayon 2D tiling for ChESS response; SIMD prefix-sum for Radon SAT. |

## API — v1.0 stabilization  ·  M3  ·  [design](design/api-v1.0.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| API-01 | P0 | done | M3 | — | Dropped `contrast`/`fit_rms` from `CornerDescriptor` + `new()`, Py (9→7 cols), WASM (stride 9→7), CLI JSON, book/README/CHANGELOG. σ math intact; snapshot counts bit-stable; all gates + maturin/pytest(79)/wasm-pack green. Net −59 LOC. Ripple → TOOL-01. |
| API-02 | P1 | done | M3 | — | Lifted `nms_radius`/`min_cluster_size` to a shared `DetectionParams` on `DetectorConfig.detection` (Rust/Py/WASM/CLI/JSON); `with_detection` builders added; snapshot counts unchanged. |
| API-03 | P1 | done | M3 | — | Moved `ChessParams`/`RefinerKind` off the core root into `chess_corners_core::unstable` (facade still re-exports at `low_level`); demoted Radon primitives (`ANGLES`/`DIR_COS`/`DIR_SIN`, `fit_peak_frac`, `box_blur_inplace`, `SatElem`) and the `ring`/`primitives` stage modules to `pub(crate)`. Behavior-identical; snapshot counts unchanged. |
| API-04 | P2 | done | M3 | — | Feature-gated `ChessRefiner::Ml` (already cfg-gated since 0.11.0); removed the `chess_refiner_to_kind` silent-mapping helper. Behavior byte-identical; no user-visible change. |
| API-05 | P1 | done | M3 | — | `.pyi` already complete (`detect`/`config`/`apply_config`/`radon_heatmap`) and factory names already match Rust (`chess`/`chess_multiscale`/`radon`/`radon_multiscale`; the RFC's `single_scale`/`multiscale_preset` "drift" was stale — `single_scale` is the faithful `MultiscaleConfig::SingleScale` ctor). Added a stub-vs-runtime parity guard test (pytest 82→83). |
| API-06 | P1 | done | M3 | — | Sealed `DenseDetector`/`CornerRefiner` (private `Sealed` supertrait; removed the facade's only external impl, a no-op ML-seed refiner provably == a direct `detect_peaks_*` call); added `#[non_exhaustive]` to `CenterOfMassConfig`/`ForstnerConfig`/`SaddlePointConfig`/`RingOffsets`/`ChessBuffers`; documented MSRV (stable ≥ 1.88, `simd` = nightly). All gates + maturin/pytest(82)/wasm-pack green. |
| API-07 | P2 | done | M3 | API-06 | Documented (no behavior change): unknown/future `#[non_exhaustive]` core variants map to each enum's documented default in the core→binding direction (forward-compat); user input is already strict (Python `reject_unknown_keys`; WASM typed getter/setter surface has no free-form keys). WASM discriminants were already explicit (`= 0/1/2`); added a `cargo test` pinning all four `#[wasm_bindgen]` enums (28→29) so reordering is caught. |
| API-08 | P0 | todo | M3 | API-01..07 | Add `cargo-semver-checks` to CI with a 1.0 baseline |
| API-09 | P0 | todo | M3 | API-08 | Tag `1.0.0` with migration notes |

## SITE — GitHub Pages  ·  M4 (dep M3)  ·  [design](design/site-architecture.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SITE-01 | P1 | done | M4 | — | Landing page `.github/pages/index.html` adapted to chess-corners (title, description, 4 cards → /book//api//demo//performance/, GitHub link); dark theme unchanged; zero `calib` strings remain. |
| SITE-02 | P1 | done | M4 | SITE-01,03,04 | Reworked `docs.yml`: nightly + wasm32 + wasm-pack + Bun; build demo; assemble `public/` = landing(/) + api(/api/) + book(/book/) + demo(/demo/) + perf (carried in `.github/pages/`). LFS checkout, `check_doc_versions.py`, `chess_corners` redirect preserved; deploy job unchanged. |
| SITE-03 | P1 | done | M4 | M3 | Vite+React+Bun demo (`demo/`) over `@vitavision/chess-corners` WASM: load image → in-browser ChESS/Radon detect → overlay corners (response colormap) + two orientation axes + σ wedges + response/Radon heatmap; live controls (strategy/multiscale/threshold/refiner/orientation/NMS). `scripts/build-wasm.sh` builds pkg + sets npm name. Playwright-verified (0 console errors); artifacts gitignored. |
| SITE-04 | P2 | done | M4 | — | `scripts/gen-perf-data.sh` + `gen_perf_data.py` + a `perf_overlay` example (4-stage decomposition, p50×60, corner-count faithfulness check) emit real `performance/data.json` + overlay PNGs on small/mid/large; rewrote `performance/index.html` for chess-corners (grep-clean of calib/charuco/puzzle). Example gated `required-features=["image"]`. |
| SITE-05 | P2 | done | M4 | SITE-02 | `scripts/build-site.sh` reproduces the deployed tree locally (cargo doc → mdBook → build-wasm → bun build → assemble `public/`); verified end-to-end (all five sections present). |
| SITE-06 | P3 | done | M4 | SITE-02 | `/book/`-move audit: README "Docs" badge → site root is valid as the new landing hub; no absolute book links in `book/src/` to repoint. Added a "Live demo" badge → `/demo/`. vitavision.dev linkage already present (README header + landing footer). |

## CPP — C++ bindings via vcpkg  ·  M5 (dep M3)  ·  [design](design/cpp-vcpkg-bindings.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| CPP-01 | P2 | done | M5 | M3 | `chess-corners-capi` (publish=false, staticlib+cdylib+rlib): flat `cc_config` + presets, `cc_detect_u8`, `cc_result`/`cc_result_free` (lib-owned alloc), `cc_status`/`cc_status_str`, `cc_abi_version`. Panic-trapped at every boundary; tags are `uint32_t`+`#define` (no discriminant UB). Rust parity test (corner-by-corner vs `Detector::detect_u8`) + C smoke (49 corners on 8×8). |
| CPP-02 | P2 | done | M5 | CPP-01 | `cbindgen.toml` + `generate-ffi-header` bin (with `--check` drift mode) → committed `include/chess_corners.h`. Intra-doc brackets stripped for a clean C header. |
| CPP-03 | P2 | done | M5 | CPP-02 | Header-only `chess_corners.hpp` (C++17): `Axis`/`Corner`/`Config` value types + presets, exception-safe `ResultGuard` RAII, `detect()` → `std::vector<Corner>` throwing `chess_corners::Error` (carries `cc_status`), `check_abi()` guard each call. Compiles clean under `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`. |
| CPP-04 | P2 | done | M5 | CPP-01 | `CMakeLists.txt` + package config: `find_package(chess-corners CONFIG)` → `chess-corners::chess-corners`. Builds the Rust lib, honours static/shared (`BUILD_SHARED_LIBS`/`VCPKG_LIBRARY_LINKAGE`), links `native-static-libs`, fixes macOS dylib install-name, emits relocatable pkg-config `.pc`. Verified both linkages. |
| CPP-05 | P3 | done (draft) | M5 | CPP-04 | Overlay port `ports/chess-corners/` (`vcpkg.json` + `portfile.cmake` + README): `vcpkg_from_github`→cmake configure/install/config_fixup/fixup_pkgconfig/copyright, honours `VCPKG_LIBRARY_LINKAGE`, default features only. JSON validated; structure aligned to the CMake. **Release-finalization (in `ports/README.md`):** real `v1.0.0` tag + SHA512, `vcpkg install` on Linux/macOS/Windows × static/dynamic, then registry PR. Optional feature plumbing (rayon/simd/ml-refiner) is a scoped follow-up. vcpkg not installed here → not install-verified. |
| CPP-06 | P3 | done | M5 | CPP-03 | Self-contained C++ example (synthetic 8×8 → 49 corners) + C smoke wired as CTest; `.github/workflows/cpp.yml` matrix (static+shared): header-drift gate → build → ctest → install → example via `find_package`. (Rust marshalling parity already in CPP-01's `parity.rs`.) |
| CPP-07 | P3 | done | M5 | CPP-03 | Book Part IX "C++ bindings": why C-ABI + thin C++ header, install (find_package/pkg-config/vcpkg), compile-faithful C++ and C usage examples, reentrancy/ABI-guard notes. Contributing → Part X; cross-refs + SUMMARY updated; mdbook builds clean. |

## SWEEP — dev-history/internal reference cleanup  ·  continuous → M3

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SWEEP-01 | P2 | todo | M3 | — | Audit `.rs` doc comments + book + README + CHANGELOG for banned dev-history/internal refs (lineage names, "Phase N", origin notes) |
| SWEEP-02 | P3 | todo | M3 | — | Audit non-doc code comments for outdated/internal references |
| SWEEP-03 | P3 | todo | — | — | Fix `.claude/agents/*` "calib-targets-rs" mis-naming (copied from sibling; should read chess-corners-rs) |

## SOLID — DRY / cohesion cleanup  ·  continuous → M2

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SOLID-01 | P2 | in-progress | M2 | — | Extract shared test utilities: `gaussian_blur` (dup ×5), bilinear patch extraction, synthetic chessboard, `ClassicRefiners` trait. Started: shared `benches/common/synth_chessboard` now used by `radon_response` + `descriptor_fit` (two duplicate generators removed). |
| SOLID-02 | P3 | todo | — | — | Evaluate `Refiner` enum dispatch boilerplate (`refine/mod.rs`) — refactor or accept |
| SOLID-03 | P3 | todo | M2 | SOLID-01 | Merge duplicate synthetic-chessboard generators into one fixture |

## SKILL — reusable tooling  ·  continuous  ·  user-level, project-agnostic

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SKILL-01 | P2 | done | M1 | — | Review the skill set; add `rust-cpp-bindings` (C-ABI/cbindgen + CMake + vcpkg) beside `rust-python-bindings`/`rust-wasm-bindings` |
| SKILL-02 | P3 | todo | — | SKILL-01 | Candidate: `rust-docs-site` skill — assemble a unified GitHub Pages site (landing + book + rustdoc + WASM demo + perf) from the M4 pattern |
| SKILL-03 | P3 | todo | — | — | Candidate: `repo-knowledge-base` skill — restructure a `docs/` folder into ROADMAP + BACKLOG + design + algorithm-index (the M1 pattern) |

## DOCS — knowledge base

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| DOCS-01 | P1 | done | M1 | — | Restructure `docs/` into a KB; add ROADMAP/BACKLOG/algorithm index + design docs |
| DOCS-02 | P3 | todo | — | — | Keep `design/algorithms-index.md` current as algorithms change (recurring) |
| DOCS-03 | P3 | todo | — | — | Tracing/profiling cookbook examples (pair with book Part VIII) |
| DOCS-04 | P3 | todo | — | — | Worked `no_std` examples for `chess-corners-core` |

## Carried-over / future (not yet milestoned)

| ID | Pri | Status | Deps | Task |
|----|-----|--------|------|------|
| ML-01 | P2 | todo | — | Validate ML refiner accuracy on real-world calibration images (currently synthetic-only) |
| ML-02 | P2 | todo | API-04 | Integrate ML confidence (`conf_logit`) into the pipeline (currently ignored) |
| ML-03 | P2 | todo | PERF-10 | Optimize ML inference (~23 ms / 77 corners is too slow for real-time) |
| ALGO-01 | P3 | todo | — | Adaptive per-corner refiner selection by local image context |
| PY-01 | P3 | todo | — | Python batch processing with `PyramidBuffers` reuse across frames |
| TOOL-01 | P3 | todo | API-01 | `tools/orientation_bench/` reads the old Python `(N,9)` layout positionally (`runner.py` row[3..8], `__main__.py` `corners[pi,4]`, `metrics.py`) and breaks after API-01 (now `(N,7)`). It *measured* `fit_rms`/`contrast` as metrics, so this is a semantic rework: drop those fit-residual metrics or source amp/rms via a diagnostics path. Local research tooling, outside the gates — not a 1.0 blocker. |
| PERF-11 | P2 | done | API-06, M5 | **Evaluated — keeping nightly `std::simd`.** A spike ported the ChESS kernel to both `wide` (stable, compile-time) and `pulp` (stable, runtime dispatch) and benchmarked them bit-exact vs scalar. Neither is a viable single replacement: on aarch64/NEON `wide` runs ~571 Mpix/s — a 1.9× regression vs the nightly `std::simd` path (1101) and slower than scalar autovec (745), since `wide` 1.5 has no first-class NEON for >128-bit types; `pulp` cannot express the pyramid `(a+b+c+d+2)>>2` bit-exactly (no runtime-width integer shift), forcing a non-bit-exact pyramid or the multi-backend maintenance we forbid. Decision: keep the nightly `std::simd` feature as the optional high-performance path; the stable scalar/autovec build is the **supported portable baseline** (correct, portable, adequate). No kernel change. |

## Closed

| ID | Resolution |
|----|------------|
| TASK-002 | DiskFit antipodal axis-slot inversion **not reproducible** on 0.11.0 (0 slot swaps vs RingFit among disk-path corners; ≈50/50 for both). Permanent guard added: `crates/chess-corners/tests/orientation_slot_parity.rs`. |

## Cross-workstream dependencies

- **M3 (API freeze) gates M4 and M5** — the demo, rustdoc, and C ABI must
  bind the frozen `CornerDescriptor` (no `contrast`/`fit_rms`).
- **M2 → M3** — reshape hot paths before the surface freezes.
- **PERF-09 → SITE-04** — the performance page consumes the perf baseline.
- **SOLID-01 → PERF benches** — share one synthetic-board/blur helper rather
  than adding a sixth copy when writing `PERF-01..05`.
