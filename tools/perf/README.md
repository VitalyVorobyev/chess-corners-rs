# Benchmark-regression gate (`tools/perf/`)

A CI gate that reports drift against a budget (default **2%**) for a curated
subset of the workspace's Criterion microbenches, plus a committed reference
snapshot of those benches.

> **Advisory on shared CI runners.** GitHub-hosted runners exhibit ±13%
> run-to-run variance on identical code (observed: `radon_response/640x480_up1`
> at −13.65%, `chess_response_kernel/256x256` at +2.37% on a same-runner no-change
> run), which precludes a hard 2% block. The gate currently reports drift without
> blocking the PR. To make it blocking, move the CI job to a dedicated or
> self-hosted runner and drop `continue-on-error` in
> `.github/workflows/bench-gate.yml`.

| File | Role |
| --- | --- |
| `gated-benches.json` | Single source of truth: which benches are gated, their crate/feature/filter, the budget, and the bench-time settings. |
| `gate.py` | Driver with three subcommands: `run`, `check`, `snapshot`. |
| `baseline-metrics.json` | Machine-stamped reference snapshot (see *Snapshot* below). |

The live gate runs in [`.github/workflows/bench-gate.yml`](../../.github/workflows/bench-gate.yml).

## Why same-runner head-vs-merge-base

An absolute, committed benchmark number compared against a fresh run is
hopelessly flaky at a 2% budget: GitHub-hosted runners vary by CPU model, noisy
neighbours and frequency scaling, routinely 10–30% run-to-run. So the gate
measures **both** sides in one CI job on one runner, back-to-back:

```
gate.py run pr      # PR head (the merge commit)  -> Criterion baseline "pr"
# <check out the PR's merge-base with main>
gate.py run base    # merge-base                  -> Criterion baseline "base"
gate.py check base pr
```

Measuring `base` and `pr` on the same physical CPU cancels the machine-model and
most steady-state-frequency variance. What remains is sampling variance plus
slow thermal drift between the two runs; the gated subset is chosen so that
residual stays under the budget.

The gate runs on **nightly with `--features simd`** — the workspace's supported
high-performance path (the SIMD kernels gate on nightly `portable_simd`). The
pyramid crate has no `simd` feature and is benched single-threaded for
stability.

## What is gated (and what is not)

The subset was picked from measured same-runner *no-op* deltas (two back-to-back
runs with no code change): every gated id moved < 1.5%, so a 2% gate does not
false-fail on it.

**Gated** (11 ids): `chess_response_kernel/{256,512,1024}`,
`nms_scaling/radius_{4,8}`, `descriptor_fit/corners_{64,256,1024}`,
`radon_response/{640x480,1280x720}_up1`,
`pyramid_builders/box_image_pyramid/reuse_buffers/640x480`.

**Excluded — too noisy for a 2% budget** (sub-millisecond / allocation-bound,
scheduling jitter exceeds 2% even same-runner):

- `nms_scaling/radius_{1,2}` — < 0.4 ms, +3..10% no-op jitter.
- `pyramid .../fresh_buffers/*` — per-iteration allocation, +11% no-op jitter.
- `radon_response/*_up2` — 2× internal upsample, ~8% sample dispersion.
- `orientation_fit/*` — single-corner, ns…µs scale.
- `chess_response/*` (the `radon_response` target's secondary group) —
  duplicate of `chess_response_kernel` and noisy on small frames.

**Excluded — too coarse/variable** (whole targets): `refiners`, `upscale`,
`chess_pipeline`, `radon_pipeline` (end-to-end), and the third-party pyramid
comparison builders (`fast_image_resize` / `resize` / `image`) which are not our
code.

## How the budget is enforced

`check` prints the human-readable `critcmp base pr` table into the log, but the
pass/fail is computed from `critcmp --export <baseline>` JSON, which carries the
**full-precision** Criterion median (`criterion_estimates_v1.median.point_estimate`)
keyed by `full_id`. The printed table rounds its ratio to two decimals — too
coarse to decide a 2% budget — so the export (the same data critcmp itself
consumes) is used for the actual decision.

The compared statistic is the **median** point estimate (what critcmp reports).
A true p95 would require parsing each `target/criterion/<id>/new/sample.json`;
the median is the pragmatic, standard mechanism.

A gated id is enforced only when present in **both** baselines. An id missing
from `base` (added by the PR) or from `pr` (renamed/removed by the PR) is
reported and skipped, so the gate does not false-fail when a parallel task
refactors the bench fixtures. If nothing comparable is found, the gate **fails
closed**.

Override the budget per-run with `REGRESSION_THRESHOLD_PCT`; override bench time
with `PERF_WARMUP` / `PERF_MEAS` (seconds).

## Run it locally

```bash
cargo install critcmp --locked          # one-time
python3 tools/perf/gate.py run base      # baseline A
python3 tools/perf/gate.py run pr        # baseline B (after your change)
python3 tools/perf/gate.py check base pr # compare; non-zero exit on regression
```

(To reproduce the CI semantics, run `base` from a clean merge-base checkout and
`pr` from your branch.)

## Snapshot: `baseline-metrics.json`

A machine-stamped reference of the gated medians + throughput, for human
tracking and the perf page (SITE-04). It is **distinct** from
`.github/pages/performance/data.json` and the **live gate does not read it** —
the gate is always head-vs-base. Regenerate after a deliberate, measured perf
change:

```bash
python3 tools/perf/gate.py run pr
python3 tools/perf/gate.py snapshot pr tools/perf/baseline-metrics.json
```

The committed numbers are host-specific (the file records the CPU, rustc and git
SHA); treat them as a reference point, not an absolute contract.
