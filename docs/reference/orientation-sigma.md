# Orientation σ: what `AxisEstimate::sigma` can be

Evidence study for `UQ-06`: one definition of the per-axis orientation σ for
RingFit and DiskFit, and whether it can be a calibrated 1σ in 2.0.

**Verdict.** No candidate is a calibrated 1σ on either generator, for either
method. The angle error is dominated by deterministic (noise-free) error of
the fits, up to 2° on clean, large cells and 4° on 6-px cells, which no
per-corner noise or residual term predicts. Under the pre-registered rule the
field is a **score**: rename it in 2.0. The definition to ship is the
**residual-scaled least-squares standard error**,
\\(\sqrt{[(J^\top J)^{-1}]_{\theta\theta}\,\mathrm{SSR}/(N-4)}\\), for both
methods. For RingFit that is today's value without the lookup table. For
DiskFit it replaces the heuristic and fixes its ranking: pooled Spearman
against |error| goes from 0.34 / 0.46 / −0.13 to 0.42 / 0.49 / 0.55
(optical / synth.py / real). It needs no noise estimate, so it stays
independent of `Uncertainty`.

## Pre-registered rule

Written before any result was computed (`z = wrap_π(θ̂ − θ) / σ̂` per axis;
constants fitted on `optical_chessboard` only and judged on `synth.py`):

- **R1, calibrated 1σ.** On both generators and for both methods: pooled
  P(|z|<1) in [0.55, 0.80], P(|z|<2) ≥ 0.88, robust std(z) (1.4826·median|z|)
  in [0.75, 1.33], and at least 75% of condition cells with P(|z|<1) in
  [0.45, 0.90].
- **R2, ranking.** Pooled Spearman(σ̂, |err|) at most 0.05 below the best
  candidate on either generator (in-cell mean likewise).
- **R3, cost.** At most +10% of the method's per-corner fit cost.
- **Decision.** If a candidate passes R1–R3, ship it as a calibrated `sigma`
  (prefer one without the noise estimate unless the noise-propagated one is
  ≥ 15 pp better in cells or ≥ 0.05 better in Spearman). If none passes R1,
  the field is a score: rename it, and ship the definition with the best
  minimum-over-generators pooled Spearman; ties (within 0.02) go to one
  formula for both methods, then to the cheaper, then to no noise estimate.

## Method

Harness: `crates/chess-corners/examples/orientation_sigma_study.rs`. For each
true corner the default ChESS detector finds within 1 px (12 px from the
border), it runs `fit_axes_at_point` for both methods **at the detected
position** (no top-80 DiskFit cap) and records the undirected angle error
against the true axis pair, the shipped σ, and the ingredients of each
candidate at the reported axes: \\(J\\) over the 16 bilinear ring samples
(RingFit, \\(\beta = 4\\)) or the support-disk pixels (DiskFit, the edge width
whose heuristic reproduces the shipped σ), with parameters
\\((\mu, A, \theta_1, \theta_2)\\).

| Generator | Grid | Corners |
|---|---|---|
| `optical_chessboard` (affine, exact optics) | cell 6/10/20 px × Gaussian PSF σ 0.5/1/1.5/2.5 × rotation 0/15/30/45° × axis angle 90/60/40° × noise σₙ 0/1/2/4/8 (2 seeds) | 86 814 |
| `tools/refiner_robustness/synth.py` (12× supersampled) | same cells, blurs, rolls, noises × {fronto, perspective (tilt 40°/25°, f = 1.2·width)}; plus ramp / wave illumination and vignetting at cell 10, blur 1 and 2.5 | 78 046 |
| private stereo ChArUco set (54 images, cells 70–100 px) | truth: per corner, the Jacobian of a homography fitted to the ChArUco corners within two cells (fit residual median 0.15 px) | 13 084 |

Candidates (DiskFit's 10° minimum on its disagreement fallback is kept on
all of them):

| | Definition | Needs σₙ |
|---|---|---|
| `cur` | shipped: RingFit GN covariance × residual × lookup table (1.0–1.25); DiskFit `(1.5°|3° + min(8·rel_rms, 6°))·0.55` | no |
| `b` | \\(\sqrt{c_{\theta\theta}\,s^2}\\), \\(c = (J^\top J)^{-1}\\), \\(s^2 = \mathrm{SSR}/(N-4)\\) | no |
| `a` | \\(\sqrt{[(J^\top J)^{-1} J^\top D J (J^\top J)^{-1}]_{\theta\theta}}\,\sigma_{pix}\\), \\(D\\) = bilinear weight energy per ring sample (identity on the disk), \\(\sigma_{pix}^2 = \sigma_n^2 + 1/12\\) (oracle) | yes |
| `a_est` | `a` with the detector's noise estimate (declines on 15% of the synthetic images) | yes |
| `X+f` | \\(\sqrt{X^2 + f^2}\\), floor fitted on optical to pooled P(\|z\|<1) = 0.683: `b+f` ring 2.20°, disk 1.70°; `a+f` ring 2.77°, disk 2.27° | — |

Harness checks. The shipped σ is reproduced to 1% from the recomputed
ingredients for 94% of RingFit rows (the rest are mis-converged fits, median
axis separation 36°, where re-solving μ and A at the reported angles differs
from the Gauss–Newton state; dividing the shipped σ by the lookup factor
instead gives the same Spearman) and 99.98% of DiskFit rows. Where the error
is pure noise (cell 20, square, rotation 0°/45°, blur 1–1.5), `a` is exactly
calibrated: std(z) 0.98–0.99 and P(|z|<1) 0.67–0.69 at σₙ 2, 4, 8; `b`
reaches that at σₙ 8 (std 0.98), and `cur` stays conservative (0.54–0.78).

## Results

Pooled over all conditions (R1 fails for every row):

| Gen. | Method | σ̂ | robust std(z) | P(\|z\|<1) | P(\|z\|<2) | cells in band | Spearman pooled | Spearman in-cell | median σ̂ |
|---|---|---|---|---|---|---|---|---|---|
| optical | RingFit | cur | 1.31 | 0.53 | 0.76 | 21% | 0.367 | 0.082 | 1.66° |
| optical | RingFit | b | 1.65 | 0.47 | 0.68 | 24% | 0.372 | 0.092 | 1.34° |
| optical | RingFit | b+f | 0.83 | 0.68 | 0.86 | 22% | 0.372 | 0.092 | 2.57° |
| optical | RingFit | a | 4.83 | 0.25 | 0.40 | 27% | 0.333 | 0.129 | 0.37° |
| optical | RingFit | a+f | 0.81 | 0.68 | 0.84 | 20% | 0.333 | 0.129 | 2.79° |
| optical | DiskFit | cur | 0.92 | 0.62 | 0.82 | 19% | 0.344 | 0.093 | 1.74° |
| optical | DiskFit | b | 1.88 | 0.44 | 0.63 | 23% | 0.417 | 0.102 | 0.95° |
| optical | DiskFit | b+f | 0.70 | 0.68 | 0.87 | 17% | 0.417 | 0.102 | 1.95° |
| optical | DiskFit | a+f | 0.65 | 0.68 | 0.87 | 16% | 0.353 | 0.130 | 2.28° |
| synth.py | RingFit | cur | 0.90 | 0.67 | 0.86 | 49% | 0.427 | 0.181 | 1.90° |
| synth.py | RingFit | b | 1.10 | 0.60 | 0.81 | 46% | 0.427 | 0.183 | 1.54° |
| synth.py | RingFit | b+f | 0.55 | 0.79 | 0.93 | 31% | 0.427 | 0.183 | 2.68° |
| synth.py | RingFit | a+f | 0.55 | 0.76 | 0.91 | 34% | 0.302 | 0.210 | 2.79° |
| synth.py | DiskFit | cur | 0.76 | 0.70 | 0.85 | 50% | 0.458 | 0.193 | 1.59° |
| synth.py | DiskFit | b | 1.28 | 0.54 | 0.74 | 48% | 0.486 | 0.225 | 1.22° |
| synth.py | DiskFit | b+f | 0.52 | 0.77 | 0.91 | 37% | 0.486 | 0.225 | 2.10° |
| synth.py | DiskFit | a+f | 0.49 | 0.75 | 0.90 | 39% | 0.378 | 0.198 | 2.29° |
| real | RingFit | cur | 2.96 | 0.20 | 0.50 | — | 0.035 | 0.069 | 1.36° |
| real | RingFit | b | 3.67 | 0.15 | 0.39 | — | 0.035 | 0.068 | 1.09° |
| real | RingFit | a_est | 42.1 | 0.01 | 0.03 | — | −0.170 | 0.206 | 0.11° |
| real | DiskFit | cur | 1.96 | 0.43 | 0.61 | — | −0.130 | −0.091 | 1.25° |
| real | DiskFit | b | 4.16 | 0.17 | 0.35 | — | 0.549 | 0.529 | 0.64° |
| real | DiskFit | a_est | 38.1 | 0.02 | 0.03 | — | 0.490 | 0.581 | 0.08° |

`a_est` tracks `a` on the synthetic sets (Spearman 0.31–0.42) and a single
scale `cur·k` (ring k = 1.62, disk 1.22) calibrates the pool but not the
cells (23% / 22% in band), like the floors. On real data "in-cell" is per
image, the truth carries its own error (the homography fit), so z there is an
upper bound and only the ranking is meaningful.

### Why nothing calibrates

The error does not shrink with the noise. P(|z|<1) is flat across σₙ for
`cur`, `b` and the floored candidates (optical RingFit `cur`: 0.53, 0.53, 0.53, 0.53, 0.55 at σₙ
0–8). It moves with geometry instead: by rotation 0.58 / 0.59 / 0.34 / 0.62
(0/15/30/45°), by axis angle 0.30 / 0.37 / 0.77 (40/60/90°). Noise-free
RingFit on 20-px square cells, maximum |error|:

| blur | 0° | 15° | 30° | 45° |
|---|---|---|---|---|
| 0.5 | 0.00° | 1.99° | 2.14° | 0.00° |
| 1.0 | 0.01° | 0.27° | 1.13° | 0.02° |
| 1.5 | 0.03° | 0.77° | 0.67° | 0.07° |
| 2.5 | 0.04° | 1.27° | 0.94° | 0.13° |

The 16-sample ring and the fixed-sharpness model localise an edge between
samples only through the samples that straddle it, so the angle is biased
by an amount set by the board rotation, the blur and the cell size (with
6-px cells the ring also crosses the neighbouring cells: up to 4.5° median
error at 15–30° rotation).
Both generators show it. A noise-propagated σ (`a`) describes the noise part
exactly (above) but that part is small; a residual term (`b`) sees part of
the model mismatch, which is why it ranks better pooled; neither knows the
sign or size of the bias, so no per-corner value covers it cell by cell.
This is the same situation as the bias-dominated position refiners (`UQ-04`).

### Gating, as calib-targets-rs uses it

Gate `σ̂ ≤ 15°` (the chessboard admission rule), gross error = |err| > 5°:

| Gen. | Method | σ̂ | rejected | gross among kept | gross among rejected | AUC(σ̂ → gross) | gate rejecting as many as `cur` at 15° |
|---|---|---|---|---|---|---|---|
| optical | RingFit | cur | 1.97% | 18.2% | 92% | 0.706 | 15.0° |
| optical | RingFit | b | 1.07% | 18.8% | 100% | 0.709 | 8.2° |
| optical | RingFit | a+f | 0.58% | 19.2% | 100% | 0.687 | 4.8° |
| optical | DiskFit | cur | 0.90% | 13.9% | 85% | 0.640 | 15.0° |
| optical | DiskFit | b | 0.50% | 14.1% | 100% | 0.730 | 10.0° |
| synth.py | RingFit | cur | 0.17% | 11.0% | 96% | 0.782 | 14.9° |
| synth.py | RingFit | b | 0.14% | 11.0% | 98% | 0.782 | 12.2° |
| synth.py | DiskFit | cur | 0.17% | 11.0% | 95% | 0.781 | 14.9° |
| synth.py | DiskFit | b | 0.14% | 11.0% | 98% | 0.803 | 12.2° |
| real | RingFit | cur / b / a_est | 0% | 11.7% | — | 0.567 / 0.567 / 0.612 | — |
| real | DiskFit | cur / b / a_est | 0% | 8.9% | — | 0.540 / 0.739 / 0.780 | — |

`b` rejects only gross errors and detects them at least as well as `cur`
(AUC); at the same 15° it rejects about half as many corners, and a gate of
8–12° rejects the same share. No real corner reaches 15° under any
definition.

### Real data

DiskFit's own fit was accepted on 39% of the real corners, with 0.68° median
error where RingFit has 2.69° on the same corners; on the other 61% it
returns the RingFit result (3.16° median error). The shipped DiskFit σ puts
the two on different scales (median 1.30° on the accurate disk fits, 1.14°
on the ring fallbacks) and therefore anti-ranks (Spearman −0.13); `b` puts both on one scale (0.55). RingFit's
3° real-data error is not predicted by any candidate (Spearman ≈ 0.03).

### Cost

Per corner, release, one thread, noisy optical boards (`orientation_sigma_study
cost`; the machine was loaded, so ±30%): RingFit 1.2–2.0 µs; DiskFit 1.2 µs
when the lazy gate skips the disk, 135–215 µs when it runs. `b` adds nothing
to RingFit (the solver already holds \\((J^\top J)^{-1}\\) and the residual;
it drops the table lookup). For DiskFit it is one Jacobian pass over the
~200 disk pixels at the final angles and width, 2.7–3.8 µs in f64 in the
harness: under 2% of the disk fit, and only on accepted disk fits. `a` costs
the same plus the frame noise estimate (< 0.1 ms per frame).

## Decision

By the rule: R1 fails for every candidate on both generators, so the field
is a score. Minimum-over-generators pooled Spearman: RingFit `b` 0.372 vs
`cur` 0.367 (tie), `a` 0.302; DiskFit `b` 0.417 vs `cur` 0.344, `a` 0.353.
`b` is one formula for both methods, wins DiskFit by 0.07, costs < 2%, and
needs no noise estimate. **Ship `b`, as a score, named `fit_std_error`.** A floor
does not change the ranking and calibrates only the pool, so none is added.

2.0 API consequence (breaking; renamed directly in 2.0 with a migration
note, no deprecation in 1.x):

- Rename `AxisEstimate::sigma` to `fit_std_error` and document it as "the
  residual-based least-squares standard error of the fit for `angle`,
  radians; ranks and gates axes, not a calibrated error bound: the ring/disk
  discretisation bias (up to ~2° on clean corners) is not in it". Same rename
  for `AxisFitResult::sigma_theta{1,2}`, `AxisEstimate::new`, Python
  `Detections.sigmas`, WASM and the C ABI field.
- RingFit: drop the lookup table (`ring_fit/uncertainty.rs`). Values are divided by
  1.0–1.25.
- DiskFit: replace `sigma_for_fit` by the disk Jacobian standard error; keep
  the 10° minimum on the disagreement fallback; per-axis values now differ.
- Values stay radians, `π` stays the no-information value, and the field is
  still computed whenever orientation is on, independent of `Uncertainty`.

Shipped in 2.0 as above. Since then the harness's `cur` column (CSV
`fit_std_error`) reads the shipped value, which equals `b`; the tables
here were measured on the 1.x values. The ring fit used to keep each
standard error on its pre-canonicalisation axis, so on the 1–5% of
corners where canonicalisation swaps the axes the two values were
exchanged; 2.0 fixes that.

Rejected: `a` (noise-propagated) would tie the axes to the noise estimate
and the `Uncertainty` tier (calib-targets-rs switches the tier off when it
has no noise level, so its axes would lose σ), is 4–5× smaller than the
error, and ranks worse pooled. Keeping `cur` keeps DiskFit's anti-ranking on
real data.

## Downstream: calib-targets-rs

| Where | Value | Use | Under `b` |
|---|---|---|---|
| `calib-targets-chessboard` `pipeline/mod.rs` `axis_admission_sigma`, used in `pipeline/{mod,inputs,recover}.rs` | `max(σ₀, σ₁) ≤ axis_align_tol_rad` = 15° (0.262 rad) | corners above it keep their position but lose their axes | keep 15° (admits ~1% more on hard synthetic boards, all of them non-gross) or ~10° (×0.65) to keep today's rejection; decide on their regression set, where this gate is load-bearing |
| `projective-grid` `TopologicalParams::max_axis_sigma_rad` (`topological/axis.rs`) | 0.6 rad (34°) | backstop: axis uninformative at or above it | no change (only degenerate fits reach it) |
| `projective-grid` `cluster/mod.rs` `vote_weight` | `strength / (1 + σ)` | axis-histogram vote | no change: σ ≈ 0.01–0.05 rad moves weights < 5% |
| `projective-grid` `cluster/mod.rs` `effective_tol_rad` | `tol + min(k·min(σ, 10°), 3°)`, `cluster_sigma_k = 0` in the chessboard | sigma-widened cluster tolerance | disabled; if enabled, `k` scales with the same ×0.65 |
| `calib-targets-core` `corner.rs` | `σ = π` no information, `σ = 0` for a bare angle | sentinels | no change |

Under `a`: ×0.3 on the admission gate (4.8° to match), and no σ at all when
the tier is off. Under a rename only (`cur` kept): no threshold change. The
calib-targets adapter (`calib-targets-chessboard/src/corner.rs`
`adapt_chess_corner`) maps the field one-to-one, so the rename is one line
there.

## Side finding

On the real set RingFit's axes are off by 2.9° median (p90 5.2°), DiskFit's
own fit by 0.68°, and DiskFit's lazy gate returns the RingFit result for 61%
of the corners. Separate accuracy task (`UQ-47`), not a σ question.

## Reproduce

```bash
cargo run --release -p chess-corners --example orientation_sigma_study -- optical OUT/optical.csv
python tools/orientation_sigma/gen_synth.py OUT/synth
cargo run --release -p chess-corners --example orientation_sigma_study -- files OUT/synth/list.tsv OUT/synth.csv
python tools/orientation_sigma/real.py <stereo_charuco root> OUT/real      # private data, aggregates only
cargo run --release -p chess-corners --example orientation_sigma_study -- files OUT/real/list.tsv OUT/real.csv
python tools/orientation_sigma/analyze.py OUT/optical.csv OUT/synth.csv OUT/real.csv --breakdown
cargo run --release -p chess-corners --example orientation_sigma_study -- cost
```

`OUT` must be outside the repository or gitignored.
