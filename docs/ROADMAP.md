# Roadmap

Milestones toward `v1.0.0` and the supporting deliverables. Each milestone
lists its goal, the tasks that satisfy it (IDs from [`BACKLOG.md`](BACKLOG.md)),
its dependencies, and the exit criteria that close it.

**Sequencing:** `M1 → M2 → M3 → (M4 ∥ M5)`. The doc/comment sweep (`SWEEP-*`)
and the SOLID/DRY cleanup (`SOLID-*`) are **continuous**, folded into the
M2/M3 windows.

```
M1 planning ──► M2 perf ──► M3 API v1.0 ──┬──► M4 site
   (done)        (leads)      (freeze)     └──► M5 C++/vcpkg
                          SWEEP-* / SOLID-* run continuously
```

---

## M1 — Planning & knowledge-base backbone  ·  *this session*

**Goal.** A durable, multi-session plan: a clean `docs/` knowledge base, an
algorithm index, and per-workstream design docs.

**Tasks.** `DOCS-01`.

**Exit criteria.**
- `docs/` restructured (history culled; `design/`, `reference/`, `process/`).
- [`README.md`](README.md), [`ROADMAP.md`](ROADMAP.md), [`BACKLOG.md`](BACKLOG.md) created.
- Design docs landed: [`design/algorithms-index.md`](design/algorithms-index.md),
  [`design/api-v1.0.md`](design/api-v1.0.md),
  [`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md),
  [`design/site-architecture.md`](design/site-architecture.md),
  [`design/perf-profiling.md`](design/perf-profiling.md).
- All references to moved/deleted files swept; `mdbook build book` still green.

## M2 — Performance profiling pass  ·  *leads*

**Goal.** Close bench coverage on every atomic hot path, add profiling
automation, and gate regressions in CI. See
[`design/perf-profiling.md`](design/perf-profiling.md).

**Tasks.** `PERF-01..09` (+ `PERF-10` if a bottleneck is confirmed);
fold in `SOLID-01`.

**Exit criteria.**
- Atomic benches for ring kernel, Radon SAT, RingFit solver, DiskFit, NMS.
- Flamegraph automation script under `tools/`.
- Allocation audit complete; "no per-corner allocations" invariant verified.
- CI bench gate live (≤2% p95 drift); baseline `metrics.json` captured.
- A documented verdict on whether/what to optimize (`PERF-10`).

## M3 — API `v1.0.0` freeze & release

**Goal.** Freeze a minimal, clear, semver-stable surface and tag `1.0.0`. See
[`design/api-v1.0.md`](design/api-v1.0.md).

**Tasks.** `API-01..09`; fold in `SWEEP-01`. **Depends on:** M2 (so hot-path
reshaping happens before the surface freezes).

**Exit criteria.**
- `contrast`/`fit_rms` removed from `CornerDescriptor` across all bindings + snapshot.
- `nms_radius`/`min_cluster_size` deduplicated; `ChessParams`/`RefinerKind` hidden.
- `ChessRefiner::Ml` honesty resolved; Python `.pyi` parity shipped.
- `#[non_exhaustive]`/sealed-trait policy applied; `cargo-semver-checks` in CI.
- All gates green; `1.0.0` tagged with migration notes.

## M4 — GitHub Pages site

**Goal.** A unified site: landing → book → API → demo → performance. See
[`design/site-architecture.md`](design/site-architecture.md).

**Tasks.** `SITE-01..06`. **Depends on:** M3 (demo/docs target frozen API);
consumes M2's `PERF-09` baselines.

**Exit criteria.**
- Landing page at `/`; book at `/book/`; rustdoc at `/api/`; WASM demo at
  `/demo/`; performance report at `/performance/`.
- `docs.yml` assembles and deploys all four; `scripts/build-site.sh` reproduces locally.

## M5 — C++ bindings via vcpkg

**Goal.** A vcpkg-installable C/C++ binding. See
[`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md).

**Tasks.** `CPP-01..07`. **Depends on:** M3 (C ABI maps the frozen descriptor).

**Exit criteria.**
- `chess-corners-capi` crate + generated header + C++ convenience header.
- CMake `find_package(chess-corners)` works; vcpkg overlay port installs.
- C/C++ smoke + parity test in CI; example consumer builds.

## Continuous

- **`SWEEP-*`** — remove dev-history/internal references from public surfaces
  (book/README/rustdoc/CHANGELOG) and stale code comments. Prioritized into M3.
- **`SOLID-*`** — DRY/cohesion cleanup (shared test utils, dispatch
  boilerplate, duplicate fixtures). Prioritized into M2.
- **`SKILL-*`** — keep the reusable user-level skill set current with the
  patterns this program produces (C++/vcpkg bindings, docs-site assembly,
  knowledge-base restructure). Project-agnostic, small, reusable.

## Out of scope (for now)

Anything requiring a post-1.0 breaking change; new detector algorithms;
stable-Rust SIMD (tracked in the backlog as a future investigation).
