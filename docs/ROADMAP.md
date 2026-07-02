# Roadmap

Milestones toward `v1.0.0` and the supporting deliverables. Each milestone
lists its goal, the tasks that satisfy it (IDs from [`BACKLOG.md`](BACKLOG.md)),
its dependencies, and the exit criteria that close it.

**Sequencing:** `M1 → M2 → M3 (API surface) → M4 site → M5 C++/vcpkg → M6 design hardening → tag 1.0.0`.
The API *surface* froze in M3, then M6 re-opened it for a final coherence and
tech-debt pass before the freeze becomes a semver-locked contract. The `1.0.0`
**release** (version bump, `cargo-semver-checks`, tag → crates.io/PyPI/npm
publish) is deferred to the end — there is no schedule pressure — so the site,
C++ bindings, and the hardened surface ship together in the first release. The
doc/comment sweep (`SWEEP-*`) and the SOLID/DRY cleanup (`SOLID-*`) are
**continuous**, folded into the M2/M3/M6 windows.

```
M1 ──► M2 ──► M3 API surface ──► M4 site ──► M5 C++/vcpkg ──► M6 hardening ──► tag 1.0.0
(done) (done)  (re-frozen)       (done)      (done)          (done)           (deferred)
                     SWEEP-* / SOLID-* run continuously
```

---

## M1 — Planning & knowledge-base backbone  ·  *done*

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
- CI bench gate live (≤2% drift, same-runner head-vs-base median); baseline snapshot captured (`tools/perf/`, `bench-gate.yml`) — **done**.
- A documented verdict on whether/what to optimize (`PERF-10`).

## M3 — API `v1.0.0` surface freeze

**Goal.** Freeze a minimal, clear, semver-stable surface. See
[`design/api-v1.0.md`](design/api-v1.0.md). **Status:** surface freeze
**done** (API-01..07 landed, all gates green on PR #63), then **re-opened and
re-frozen by M6** — the M3 freeze surfaced an incoherence (root-public functions
required the `unstable`-only `ChessParams`), fixed in M6. The release steps
`API-08` (`cargo-semver-checks` CI) and `API-09` (version bump + tag) are
**deferred to after M4 + M5 + M6** so 1.0.0 ships with the site, C++ bindings,
and the hardened surface.

**Tasks.** `API-01..07` (done); `API-08`/`API-09` (deferred to release); fold
in `SWEEP-01`. **Depends on:** M2 (so hot-path reshaping happens before the
surface freezes).

**Exit criteria.**
- ✅ `contrast`/`fit_rms` removed from `CornerDescriptor` across all bindings + snapshot.
- ✅ `nms_radius`/`min_cluster_size` deduplicated. (`ChessParams`/`RefinerKind`
  were moved to `unstable` here, then promoted to the documented crate root in
  M6 once that proved incoherent — see DEBT-01.)
- ✅ `ChessRefiner::Ml` honesty resolved; Python `.pyi` parity + guard test shipped.
- ✅ `#[non_exhaustive]`/sealed-trait policy applied; MSRV stated; binding discriminants pinned.
- 🟡 `cargo-semver-checks` CI (advisory) + `1.0.0` migration notes **landed**; version bump + tag deferred to the release step (after M4 deploy + M5 vcpkg).

## M4 — GitHub Pages site  ·  *done*

**Goal.** A unified site: landing → book → API → demo → performance. See
[`design/site-architecture.md`](design/site-architecture.md). The existing
`.github/pages/` is a raw copy from the sibling `calib-targets-rs` and is
being adapted to chess-corners (content, demo, perf data), not rebuilt.

**Tasks.** `SITE-01..06`. **Depends on:** M3 surface freeze (demo/docs target
the frozen API); consumes M2's measured perf baselines.

**Exit criteria.**
- Landing page at `/`; book at `/book/`; rustdoc at `/api/`; WASM demo at
  `/demo/`; performance report at `/performance/`.
- `docs.yml` assembles and deploys all four; `scripts/build-site.sh` reproduces locally.

## M5 — C++ bindings via vcpkg  ·  *done*

**Goal.** A vcpkg-installable C/C++ binding. See
[`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md). **Status:**
build work **done** (CPP-01..07); the vcpkg port ships as a verified-locally
draft whose registry finalization (real `v1.0.0` tag + SHA512 + cross-platform
`vcpkg install`) folds into the release phase — see `ports/README.md`.

**Tasks.** `CPP-01..07`. **Depends on:** M3 (C ABI maps the frozen descriptor).

**Exit criteria.**
- ✅ `chess-corners-capi` crate + generated header + C++ convenience header.
- ✅ CMake `find_package(chess-corners)` works (static+shared, verified locally);
  ⏳ vcpkg overlay port install verified at release (needs the tag + vcpkg).
- ✅ C/C++ smoke + Rust marshalling-parity test in CI; example consumer builds.
- ✅ Book Part IX documents C++ usage.

## M6 — Design hardening before the freeze  ·  *done*

**Goal.** A final pass to remove design debt and "agentic slop" *before* the
1.0 surface becomes a semver-locked contract. The M3 freeze was mechanically
correct but left an incoherent surface; this milestone makes it honest. See
`BACKLOG.md` `DEBT-*`.

**Tasks.** `DEBT-01..05`. **Depends on:** M3 (it hardens the M3 surface).

**Exit criteria.**
- ✅ `chess_corners_core::unstable` deleted; its types either promoted to the
  documented crate root (`ChessParams`, `RefinerKind`, `chess_response_u8_patch`,
  two stage fns) or demoted to `pub(crate)` (ring tables, scalar reference,
  `MAX_IMAGE_UPSAMPLE`). Root-public functions are now callable without any
  "no-semver" type. (DEBT-01)
- ✅ Facade `chess_corners::low_level` escape-hatch deleted; config lowering is
  exposed as `DetectorConfig::chess_params()` / `radon_detector_params()` /
  `coarse_to_fine_params()`; hand-composers depend on `chess-corners-core`
  directly. (DEBT-02)
- ✅ `too_many_arguments` allow-cluster on the orientation/detector hot paths
  retired by bundling `(img,w,h)` into the existing `ImageView`; the stale
  allows removed. Zero unjustified `#[allow]` remain in that scope. (DEBT-03)
- ✅ Disk-sector argmax sentinel (`-1.0f32`) replaced by `Option`. (DEBT-04)
- ✅ Facade `config.rs` (1k lines) split into a cohesive `config/` module
  (public paths byte-identical). (DEBT-05)
- ✅ Pure visibility/organization refactors — detection results bit-stable; all
  gates + WASM + Python (86/86) green.

## Continuous

- **`SWEEP-*`** — remove dev-history/internal references from public surfaces
  (book/README/rustdoc/CHANGELOG) and stale code comments. Prioritized into M3.
- **`SOLID-*`** — DRY/cohesion cleanup (shared test utils, dispatch
  boilerplate, duplicate fixtures). Prioritized into M2.
- **`SKILL-*`** — keep the reusable user-level skill set current with the
  patterns this program produces (C++/vcpkg bindings, docs-site assembly,
  knowledge-base restructure). Project-agnostic, small, reusable.

## Out of scope (for now)

Anything requiring a post-1.0 breaking change; new detector algorithms.
Stable-Rust SIMD was evaluated (PERF-11) and rejected: no stable backend
matches the nightly `std::simd` path without regressing aarch64 below even
scalar, so the `simd` feature stays nightly-only as an optional
high-performance path and the stable scalar/autovec build is the supported
portable baseline.
