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
- **Flamegraph/profiling:** `tools/profile.sh` (cargo-flamegraph + samply over
  the `profile_target` example) — see Stage 2.
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

## Measured baseline (PERF-01–05)

First baselines on synthetic 8-bit chessboards (criterion median, single
thread; `crates/chess-corners-core/benches/{chess_response,radon_response}.rs`,
shared board from `benches/common/`). Reproduce with
`cargo bench -p chess-corners-core --bench chess_response` (add
`--features simd`) and `--bench radon_response` (add `--features radon-sat-u32`).

**PERF-01 — ChESS response kernel (scalar vs SIMD, after PERF-10):**

| size | scalar | SIMD | speedup |
|------|--------|------|---------|
| 256² | 162 Mpix/s | 978 Mpix/s | 6.03× |
| 512² | 158 Mpix/s | 1041 Mpix/s | 6.61× |
| 1024² | 155 Mpix/s | 1074 Mpix/s | 6.95× |

(Pre-PERF-10 SIMD was 599–621 Mpix/s = 3.7–4.0×; vectorizing the per-lane
μₗ/write-back tail lifted it to ~6–7× with **bit-identical** output.)

**PERF-02 — Radon SAT response (i64 vs u32 SAT), throughput per base-input pixel:**

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

> **Hard- vs soft-edge (resolved by PERF-12).** The hard 40/215 board gives
> `rel_rms ≈ 0.47` even at a perfect corner (the tanh model can't fit a step),
> forcing RingFit's robust grid and blocking DiskFit's lazy gate — the
> worst-case row above. On the **soft-edge** board (`rel_rms ≈ 0.01`, measured)
> RingFit hits its fast 2nd-harmonic seed path and **DiskFit's lazy gate
> short-circuits to RingFit** (123 µs → 0.73 µs, ~169×), confirming the gate
> works as designed. DiskFit pays the full disk only on genuinely warped
> corners (~61° sep → 158 µs). So DiskFit is **not** a flat 47× tax — on
> typical corners it costs the same as RingFit.

**Findings (these drive PERF-10):**

- Throughput is roughly **size-independent up to 1024²** (scalar ≈155–162,
  SIMD ≈978–1074 Mpix/s after PERF-10), so the ChESS kernel is compute-bound,
  not yet memory-bound, at these sizes.
- **PERF-10 (done):** SIMD originally delivered only ~3.7–4.0× because the
  per-lane local-mean (μₗ) loop + response write-back were unvectorized.
  Replacing that tail with five contiguous cross-loads + a single vector store
  lifted SIMD to **~6–7×** (978–1074 Mpix/s; kernel ~1.65× faster) with
  **bit-identical** output (the 5-sample sum ≤1275 casts to f32 exactly; no FMA).
- The u32-SAT win (~9–16%) **collapses toward noise at upsample=2**, which
  proves SAT construction is *not* the upsample=2 bottleneck; the
  per-output-pixel angular sampling is. Aim "SAT SIMD prefix-sum" work at
  upsample=1, and target angular sampling for upsample=2.
- NMS is **sub-quadratic**, not the feared O(n²): a 64× window-area increase
  (r=1→8) costs only 2.15× time. The O(W·H) threshold/max scan dominates the
  ~766 µs floor (`is_local_max` early-exits on slopes, so the `(2r+1)²` window
  is paid only at sparse maxima). A PERF-10 NMS win must target the full scan,
  not the window.
- DiskFit costs ~47× RingFit *only when it runs the full disk* (warped/large-
  skew corners). PERF-12 confirmed the lazy gate short-circuits to RingFit on
  typical soft, near-90° corners (DiskFit ≈ RingFit ≈ 0.73 µs), so default-path
  DiskFit overhead is negligible — the full disk is paid only where the
  geometry needs it.

## Stage 2 — flamegraph automation (`PERF-07` — already in place)

Profiling automation already exists: `tools/profile.sh` wraps
`cargo-flamegraph` and `samply` over the dedicated `profile_target` example
(`crates/chess-corners/examples/profile_target.rs`), with `chess` / `radon` /
`refiner` modes and timestamped SVG output under `testdata/out/profiles/`.
Use it to confirm the PERF-10 hotspots:

```sh
tools/profile.sh chess   testimages/large.png   # ChESS multiscale
tools/profile.sh radon   testimages/large.png   # whole-image Radon
tools/profile.sh refiner saddle testimages/large.png
```

The only residual PERF-07 work is documentation — surface this in book
Part VIII (tracked as DOCS-03).

## Stage 3 — allocation audit + targeted optimization

- `PERF-06` — **Allocation audit (done — invariant holds).** The genuine hot
  paths are allocation-free per corner: `RadonPeakRefiner` sizes
  `resp`/`blur_scratch` once in `new()` and the `refine()` loop only writes
  into them (`refine/radon_peak.rs:119–136,236–251` — no `vec!`/`collect`);
  the ChESS/Radon response kernels write into reused detector buffers. The
  multiscale coarse-to-fine loop (`multiscale.rs:345–383`) *does* allocate
  **two small `Vec<Corner>` per seed** — `detect_corners` (`:358`) and
  `refine_peaks_on_image` (`:371`) return owned vectors — but the per-seed
  `compute_response_patch` dominates that cost (PERF-01/02: response is
  ms-scale; the vecs hold ~1 corner), and `refined`, the `Refiner`, and the
  detector buffers are reused across seeds. Verdict: keep as-is; adding
  `*_into` buffer-returning variants is a low-value PERF-10 follow-up, taken
  only if a flamegraph later flags those allocations.
- `PERF-10` — **Optimize confirmed bottlenecks** surfaced by Stages 1–2.
  - **Done:** ChESS SIMD μₗ/write-back tail vectorized — 4×→~7×, bit-identical
    (`detect/chess/response.rs`).
  - Remaining candidate levers (PERF-13, only if profiling justifies): Radon
    angular sampling at upsample=2; NMS O(W·H) scan; rayon 2D tiling for R1;
    SIMD prefix-sum for the R2 SAT; batched refinement.

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
