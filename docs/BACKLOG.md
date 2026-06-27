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
| PERF-03 | P2 | todo | M2 | — | Bench: RingFit GN per-iteration / convergence — `core/.../orientation/ring_fit/` |
| PERF-04 | P2 | todo | M2 | — | Bench: DiskFit gradient sampling vs RingFit on warped corners |
| PERF-05 | P2 | todo | M2 | — | Bench: NMS scaling with radius {1,2,4,8} on dense maps — `core/.../detect/chess/detect.rs` |
| PERF-06 | P1 | done | M2 | — | Allocation audit: inner loops allocation-free (RadonPeak scratch reused; response kernels use reused buffers). Multiscale loop allocates 2 small `Vec<Corner>`/seed (`multiscale.rs:358,371`) but response-patch dominates — keep; optional `*_into` is PERF-10. |
| PERF-07 | P1 | done | M2 | — | Flamegraph/profiling automation already exists: `tools/profile.sh` (cargo-flamegraph + samply over `profile_target`). Residual: document in book Part VIII (→ DOCS-03). |
| PERF-08 | P1 | todo | M2 | PERF-01..05 | CI bench gate: baseline compare, fail on >2% p95 regression |
| PERF-09 | P2 | todo | M2 | PERF-08 | Capture baseline `metrics.json` (feeds SITE-04) |
| PERF-10 | P2 | todo | M2 | PERF-01,02,07 | Optimize confirmed bottlenecks (evidence): (a) vectorize per-lane μₗ loop + response write-back in ChESS SIMD path (`response.rs:508–524`) — caps gain at ~4×; (b) angular sampling at upsample=2 in Radon (u32-SAT win collapses there). Guarded ≤2% p95 |

## API — v1.0 stabilization  ·  M3  ·  [design](design/api-v1.0.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| API-01 | P0 | todo | M3 | — | Drop `contrast`/`fit_rms` from `CornerDescriptor` + `new()` + Py/WASM/CLI + snapshot |
| API-02 | P1 | todo | M3 | — | Deduplicate `nms_radius`/`min_cluster_size` across Chess/Radon configs |
| API-03 | P1 | todo | M3 | — | Hide `ChessParams`, `RefinerKind` (and other internal leaks) from core root |
| API-04 | P2 | todo | M3 | — | Resolve `ChessRefiner::Ml` silent CoM fallback (honor / post-step / gate out) |
| API-05 | P1 | todo | M3 | — | Ship Python `.pyi` parity (`config()`/`apply_config()`); stub-vs-runtime test |
| API-06 | P1 | todo | M3 | — | Apply `#[non_exhaustive]` + sealed-trait policy; state MSRV |
| API-07 | P2 | todo | M3 | API-06 | Binding unknown-variant handling: document/harden; pin WASM discriminants |
| API-08 | P0 | todo | M3 | API-01..07 | Add `cargo-semver-checks` to CI with a 1.0 baseline |
| API-09 | P0 | todo | M3 | API-08 | Tag `1.0.0` with migration notes |

## SITE — GitHub Pages  ·  M4 (dep M3)  ·  [design](design/site-architecture.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SITE-01 | P1 | todo | M4 | — | Landing page `.github/pages/index.html` (rewrite calib-targets copy → chess-corners, 4 cards) |
| SITE-02 | P1 | todo | M4 | SITE-01,03,04 | Rework `docs.yml` → assemble `public/{,/api,/book,/demo,/performance}` (book → `/book/`) |
| SITE-03 | P1 | todo | M4 | M3 | WASM demo (`demo/`, Vite+React+Bun) + `scripts/build-wasm.sh` |
| SITE-04 | P2 | todo | M4 | PERF-09 | `scripts/gen-perf-data.sh` → `performance/data.json` + overlays + `performance/index.html` |
| SITE-05 | P2 | todo | M4 | SITE-02 | `scripts/build-site.sh` (one-command local build) |
| SITE-06 | P3 | todo | M4 | SITE-02 | Update README/book cross-links for `/book/` move; vitavision.dev linkage |

## CPP — C++ bindings via vcpkg  ·  M5 (dep M3)  ·  [design](design/cpp-vcpkg-bindings.md)

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| CPP-01 | P2 | todo | M5 | M3 | New `chess-corners-capi` crate: `extern "C"` (staticlib+cdylib) over the facade |
| CPP-02 | P2 | todo | M5 | CPP-01 | `cbindgen.toml` + generated `include/chess_corners.h` (committed, CI-checked) |
| CPP-03 | P2 | todo | M5 | CPP-02 | Hand-written C++ header `chess_corners.hpp` (RAII, `std::vector`) |
| CPP-04 | P2 | todo | M5 | CPP-01 | CMake package config (`find_package(chess-corners)`) + pkg-config |
| CPP-05 | P3 | todo | M5 | CPP-04 | vcpkg port (`vcpkg.json` + `portfile.cmake`); overlay first, then registry PR |
| CPP-06 | P3 | todo | M5 | CPP-03 | C/C++ smoke + parity test + CI job + example consumer |
| CPP-07 | P3 | todo | M5 | CPP-03 | Book chapter: C++ usage |

## SWEEP — dev-history/internal reference cleanup  ·  continuous → M3

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SWEEP-01 | P2 | todo | M3 | — | Audit `.rs` doc comments + book + README + CHANGELOG for banned dev-history/internal refs (lineage names, "Phase N", origin notes) |
| SWEEP-02 | P3 | todo | M3 | — | Audit non-doc code comments for outdated/internal references |
| SWEEP-03 | P3 | todo | — | — | Fix `.claude/agents/*` "calib-targets-rs" mis-naming (copied from sibling; should read chess-corners-rs) |

## SOLID — DRY / cohesion cleanup  ·  continuous → M2

| ID | Pri | Status | Milestone | Deps | Task |
|----|-----|--------|-----------|------|------|
| SOLID-01 | P2 | in-progress | M2 | — | Extract shared test utilities: `gaussian_blur` (dup ×5), bilinear patch extraction, synthetic chessboard, `ClassicRefiners` trait. Started: shared `benches/common/synth_chessboard` (de-dups the Radon bench copy). |
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
| PERF-11 | P3 | todo | — | Investigate stable-Rust SIMD as an alternative to nightly `portable_simd` |

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
