# Design: performance profiling & optimization

**Status:** Draft / design. **Workstream:** `PERF-*` (ROADMAP milestone **M2**
— this milestone *leads* the program). **Method:** the four-stage cadence —
**benches → flamegraph → rayon+SIMD → accuracy guard**, with a **≤2% p95
drift** budget on the accuracy/latency guards.

## Why perf leads

Optimizing while the code is still pre-1.0 lets us reshape hot paths freely
before the API and the published performance page freeze around them. The
output (baselines + a CI bench gate) also feeds the site's performance page
([`site-architecture.md`](site-architecture.md), `SITE-04`).

## Existing tooling (reuse, don't reinvent)

- **Criterion benches:** `chess-corners-core/benches/{refiners,radon_response,descriptor_fit}.rs`;
  `chess-corners/benches/{upscale,chess_pipeline,radon_pipeline}.rs`;
  `box-image-pyramid/benches/pyramid_perf.rs`.
- **Python harness:** `tools/perf_bench.py` (feature combos `{simd, rayon,
  par_pyramid}` × `{multi, single, radon}` → JSON trace metrics).
- **Accuracy guard:** `chess-corners/tests/perf_accuracy_guard.rs` (median
  error tracking); `refiner_benchmark.rs` (per-refiner throughput/latency).
- **Feature flags:** `rayon`, `simd`, `par_pyramid`, `radon-sat-u32`.

## Stage 1 — close the bench coverage gaps

End-to-end pipelines are benched, but several **atomic** hot paths
(see [`algorithms-index.md`](algorithms-index.md) IDs) are not isolated:

| Task | Hot path (ID) | Source | What to measure |
|------|---------------|--------|-----------------|
| `PERF-01` | ChESS ring kernel (R1) | `core/src/detect/chess/response.rs` | per-pixel 16-sample gather + SR/DR/MR accumulation, scalar vs SIMD |
| `PERF-02` | Radon SAT build (R2) | `core/src/detect/radon/response.rs` | SAT construction on 1k²/2k², i64 vs u32 element |
| `PERF-03` | RingFit GN solver (O1) | `core/src/orientation/ring_fit/{solver,robust}.rs` | per-iteration cost + convergence path, nominal vs outlier samples |
| `PERF-04` | DiskFit gradient sampling (O2) | `core/src/orientation/disk_sector/` | disk accumulation vs RingFit on warped corners |
| `PERF-05` | NMS scaling (D1) | `core/src/detect/chess/detect.rs` | throughput vs NMS radius {1,2,4,8} on dense response maps |

## Stage 2 — flamegraph automation (`PERF-07`)

There is **no** flamegraph/`perf record` automation today. Add a script under
`tools/` (cargo-flamegraph wrapper) that profiles representative configs
(single-scale ChESS, multiscale ChESS, Radon) on a standard image and emits
SVGs to a known location, documented in the perf chapter (book Part VIII).

## Stage 3 — allocation audit + targeted optimization

- `PERF-06` — **Allocation audit.** Verify the "no per-corner allocations in
  hot paths" invariant (`AGENTS.md`). Suspects: `core/src/refine/radon_peak.rs`
  (localized Radon per seed) and multiscale ROI refinement
  (`chess-corners/src/multiscale.rs`). Measure alloc count vs corner count on
  256²/512²/1024² boards; reuse scratch buffers where it leaks.
- `PERF-10` — **Optimize confirmed bottlenecks** surfaced by Stages 1–2.
  Candidate levers (only after profiling justifies them):
  - rayon 2D tiling for R1 (current parallelism is row-by-row, not cache-tiled);
  - SIMD horizontal prefix-sum for the R2 SAT (currently scalar);
  - batched refinement across corners (long-standing backlog item).

## Stage 4 — accuracy/perf guards (≤2% p95 drift)

Every optimization in Stage 3 must pass:
- `perf_accuracy_guard.rs` — corner-position error unchanged (determinism +
  numerical equivalence; parallel results sorted by stable keys).
- A **p95 latency guard**: regressions beyond **2%** p95 fail.

## CI bench gate (`PERF-08`)

Wire criterion into CI as a tracked gate: store a baseline, compare PR runs,
fail on >2% p95 regression on the core benches. Depends on `PERF-01..05`
existing.

## Baselines for the site (`PERF-09`)

Capture a baseline `metrics.json` (per-stage timings + corner counts across
`testimages/`) that the performance page consumes. Single source of truth
shared by `tools/perf_bench.py`, the CI gate, and `SITE-04`.

## Recommended order

1. `PERF-01`, `PERF-02` (biggest hot paths) → `PERF-07` flamegraph.
2. `PERF-03`, `PERF-04`, `PERF-05` (remaining gaps) + `PERF-06` alloc audit.
3. `PERF-08` CI gate + `PERF-09` baselines.
4. `PERF-10` optimize only what the data flags, guarded by Stage 4.

Fold `SOLID-01` (shared test utilities — `gaussian_blur` is duplicated 5×)
into this window so new benches reuse one synthetic-board + blur helper
instead of adding a sixth copy.

---

Task list: [`../BACKLOG.md`](../BACKLOG.md) `PERF-*`. Book chapter: Part VIII
(Benchmarks and performance).
