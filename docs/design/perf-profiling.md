# Design: performance profiling & optimization

**Status:** Implemented (M2). **Workstream:** `PERF-*` (ROADMAP milestone **M2**,
which *leads* the program). **Method:** the four-stage cadence —
**benches → flamegraph → rayon+SIMD → accuracy guard** — with a **≤2% p95 drift**
budget on the accuracy/latency guards.

Optimizing while pre-1.0 lets us reshape hot paths freely before the API and the
published performance page freeze around them. The output (baselines + a CI
bench gate) also feeds the site's performance page
([`site-architecture.md`](site-architecture.md), `SITE-04`).

## Tooling

- **Criterion benches:** `chess-corners-core/benches/{chess_response,radon_response,refiners,descriptor_fit,nms_scaling}.rs`;
  `chess-corners/benches/{upscale,chess_pipeline,radon_pipeline}.rs`;
  `box-image-pyramid/benches/pyramid_perf.rs`. Shared fixtures from
  `benches/common/` (see `SOLID-01`).
- **Flamegraph/profiling:** `tools/profile.sh` wraps `cargo-flamegraph` + `samply`
  over the `profile_target` example (`chess` / `radon` / `refiner` modes).
- **Accuracy guard:** `chess-corners/tests/perf_accuracy_guard.rs` (corner-position
  error unchanged; parallel results sorted by stable keys).
- **CI gate (`PERF-08`):** `tools/perf/{gate.py,gated-benches.json}` +
  `.github/workflows/bench-gate.yml` — same-runner PR-head-vs-merge-base `critcmp`
  over curated low-variance ids, fail on >2% median drift. Baseline snapshot
  (`tools/perf/baseline-metrics.json`, `PERF-09`) is a reference, distinct from
  the live head-vs-base check.

## Atomic hot paths (bench IDs from [`algorithms-index.md`](algorithms-index.md))

| Task | Hot path | Source |
|------|----------|--------|
| `PERF-01` | ChESS ring kernel (R1) | `core/src/detect/chess/response.rs` |
| `PERF-02` | Radon SAT build (R2) | `core/src/detect/radon/response.rs` |
| `PERF-03` | RingFit GN solver (O1) | `core/src/orientation/ring_fit/` |
| `PERF-04` | DiskFit gradient sampling (O2) | `core/src/orientation/disk_sector/` |
| `PERF-05` | NMS scaling (D1) | `core/src/detect/chess/detect.rs` |

## Measured baseline (synthetic 8-bit boards, criterion median, single thread)

**PERF-01 — ChESS response kernel (scalar vs SIMD, after PERF-10):**

| size | scalar | SIMD | speedup |
|------|--------|------|---------|
| 256² | 162 Mpix/s | 978 Mpix/s | 6.03× |
| 512² | 158 Mpix/s | 1041 Mpix/s | 6.61× |
| 1024² | 155 Mpix/s | 1074 Mpix/s | 6.95× |

**PERF-02 — Radon SAT response (i64 vs u32 SAT), per base-input pixel:**

| size / upsample | i64 | u32 | u32 win |
|-----------------|-----|-----|---------|
| 1024², up=1 | 121 Mpix/s (8.64 ms) | 132 Mpix/s (7.93 ms) | +8.9% |
| 1024², up=2 | 22.4 Mpix/s (46.9 ms) | 25.5 Mpix/s (41.1 ms) | +14.0% |

**PERF-03/04 — orientation fit, single corner (`fit_axes_at_point`):**

| fixture | RingFit | DiskFit | note |
|---------|---------|---------|------|
| hard-edge 90° corner | 2.60 µs | 123 µs | worst case — both forced to the slow path |
| **soft-edge 90° corner (typical)** | **0.73 µs** | **0.73 µs** | RingFit fast-seed path; DiskFit lazy gate short-circuits to RingFit |
| warped corner (~61° sep) | — | 158 µs | DiskFit full disk (its intended case) |

**PERF-05 — NMS scaling at 1024² (radius sweep, one shared response map):**

| NMS radius | 1 | 2 | 4 | 8 |
|------------|---|---|---|---|
| time | 766 µs | 836 µs | 995 µs | 1646 µs |

## Findings

- The ChESS kernel is **compute-bound, not memory-bound** up to 1024²
  (throughput roughly size-independent).
- **PERF-10 (done):** SIMD originally delivered ~4× because the per-lane μₗ loop
  + response write-back were unvectorized. Replacing that tail with five
  contiguous cross-loads + a single vector store lifted SIMD to ~6–7×
  (978–1074 Mpix/s), **bit-identical** (the 5-sample sum ≤ 1275 casts to f32
  exactly; no FMA).
- **PERF-11 (evaluated — keeping nightly `std::simd`):** a spike ported the
  kernel to `wide` and `pulp` (both stable), bit-exact vs scalar on
  aarch64/NEON. `wide` regressed to 571 Mpix/s — 1.9× below the nightly path
  (1101) and below scalar autovec (745) — because `wide` 1.5 has no first-class
  NEON for >128-bit types; `pulp` cannot express the pyramid `(a+b+c+d+2)>>2`
  bit-exactly (no runtime-width integer shift). x86 portable builds were SSE2
  for both. **Decision: nightly `simd` stays the optional high-performance path;
  the stable scalar/autovec build is the supported portable baseline.**
- The u32-SAT win (~9–16%) **collapses toward noise at upsample=2**, proving SAT
  construction is *not* the upsample=2 bottleneck — per-output-pixel angular
  sampling is. Aim SAT SIMD prefix-sum at upsample=1; target angular sampling
  for upsample=2.
- NMS is **sub-quadratic**, not O(n²): a 64× window-area increase (r=1→8) costs
  only 2.15×. The O(W·H) threshold/max scan dominates the ~766 µs floor
  (`is_local_max` early-exits on slopes). Any NMS win must target the full scan.
- **DiskFit is not a flat 47× tax** (`PERF-12`): the full disk (~47× RingFit) is
  paid only on genuinely warped corners. On typical soft, near-90° corners the
  lazy gate short-circuits to RingFit (≈ 0.73 µs), so default-path overhead is
  negligible.

## Allocation audit (`PERF-06`)

Hot paths are allocation-free per corner (response kernels write into reused
detector buffers). The multiscale coarse-to-fine loop allocates two small
`Vec<Corner>` per seed (`multiscale.rs`), but per-seed `compute_response_patch`
dominates that cost. **Verdict: keep as-is;** adding `*_into` buffer-returning
variants is a low-value follow-up (`PERF-13`), taken only if a flamegraph later
flags those allocations.

## Accuracy/perf guard (≤2% p95 drift)

Every optimization must pass `perf_accuracy_guard.rs` (corner-position error
unchanged; determinism) and a p95 latency guard: regressions beyond 2% p95 fail.

---

Task list: [`../BACKLOG.md`](../BACKLOG.md) `PERF-*`. Book chapter: Part VIII
(Benchmarks and performance).
