# Position covariance for every localizer (UQ-24)

Internal evidence for the `Uncertainty::Propagated` tier
([design](../design/uncertainty-v2.md) §3, §4.1, §6, §7 waivers).
Aggregates only.

Tools: `crates/chess-corners-core/src/uncertainty_eval.rs` (Monte Carlo,
σ̂₀², calibration dumps; `--ignored` tests), `tools/refiner_robustness/`
`export_uq24.py` (independent-generator scenes), `fit_systematic.py`
(synthetic tables), `real_validation/fit_floor.py` (real-data floor, on the
`UQ-30` harness), `run_real_weighted.py` (per-camera weighted calibration).
Local outputs: `bench_results/uq24/`, `tools/refiner_robustness/out/`.

## Model

`Σ = max(σₙ², 1/12)·C·(growth) + Σ_sys`, `C = Σₚ gₚgₚᵀ`, `gₚ = ∂x̂/∂Iₚ`.

- **Jacobians.** CenterOfMass: through the ChESS response (`SR − DR −
  16|μₙ−μₗ|`, signs of each `|·|`) of every positive pixel of the window and
  the centroid ratio; the response map carries its ring so the refiner knows
  the response function. Förstner: `du = A⁻¹ Σ w (dg gᵀ + g dgᵀ)(p − u)`.
  Radon peak fit: 3-point fit → 3×3 box blur → `(S_max − S_min)²` with
  arg-max/min rays fixed → ray sums → 2× bilinear upsample; the 8-bit
  rounding of the upsampled samples enters as 1/12 per working sample in
  `Σ_sys`, and the noise term grows by `1 + 3·var(D)/D²` per axis (`D` the
  fit's curvature). SaddlePoint unchanged. Central-difference checks: CoM
  1e-4 (float re-implementation), Radon 2e-5 at upsample 1 / 2.3e-2 at 2
  (the reference does not round), Förstner 2e-3 (±1 gray level),
  SaddlePoint 1e-3. One convergence predicate for SaddlePoint under both
  tiers (regular window sensitivity); tested for bit-identical positions.
- **Systematic term** (`SystematicModel`, one table per localizer next to its
  estimator): observed window offset `û`, edge width `w`, gain `g(w)` and
  residual `ρ(w)` per knot, Bayesian pull estimate with a uniform prior over
  the grid cell, plus a real-data floor `f²I`.
- **Edge width** (`PixelTile::edge_width`, shared tile with the CoM and
  Förstner Jacobians): on a 5×5-binomial-smoothed radius-5 disk, contrast
  (means within a fifth of the range of the extremes) over √2π × mean of the
  eight largest gradients. Camera-formed boards: 1.12 / 1.15 / 1.31 / 1.40 /
  1.50 px at lens σ 0.3 / 0.5 / 1.1 / 1.5 / 2.5 (std 0.03–0.15 over cells,
  rotations). At a contrast of 60 and σₙ ≤ 8: within 6% of noise-free
  (≤ 15% at σₙ 8, blur 2.5).

## Monte Carlo noise term (whitened std, 1 = calibrated)

Lens-blur renderer, cells 8/12, 0/8/45°, blur 0.5/1.5/2.5, σₙ 1/2/8, 64
draws; independent generator: 4 scenes, 32 draws. 66 cells per localizer.

| localizer | range | outside 0.9–1.1 |
|---|---|---|
| SaddlePoint, Radon+SP | 0.98–1.07 | none |
| CenterOfMass | 0.80–1.01 | 9, all σₙ 8 (conservative) |
| Förstner | 0.72–1.01 | 7, σₙ 8 at blur ≥ 2 (conservative) |
| Radon peak fit | 0.99–2.54 | 17: blur ≥ 1.5 at 45° (1.4–2.5 at σₙ ≤ 2), 1.1–1.25 at 8°/30° blur 2–2.5 |

Radon's failures are peak-pixel switching on a flat-topped response
(waiver in the design doc); the second-order growth term moves the σₙ 8
cells (1.53 → 1.38, 1.31 → < 1.1) but not the low-noise ones, and the
response curvature does not predict the systematic error (|corr| ≤ 0.1).

## Systematic tables (synthetic, both generators pooled)

Noise-free corners, cells 6–12, rotations 0–45°, 60° shear, lens σ 0.3–3,
~70 k per localizer; the generators agree on the gains.

- Förstner: gain 0.78 → 0 (w ≈ 1.5) → −0.22; the linear pull explains most
  of its bias at w ≤ 1.24 (RMS 0.07–0.12 → 0.015–0.025 px).
- CenterOfMass: no pull (`g = 1`); RMS 0.04–0.09 px, strongly
  rotation-dependent (8°: 0.17 px at blur 0.3; 0/45°: 0.02–0.05), which
  width cannot see.
- Radon peak fit: gain 0.86–0.95 (w ≤ 1.35), 0.64–0.76 (w ≥ 1.4); largest
  at 45° and under shear.
- SaddlePoint: `ρ` 0.004–0.011 px by width.

Correction vs covariance: removing the modelled pull would lower Förstner's
noise-free RMS 3–8× at w ≤ 1.24 (still ≥ 2× SaddlePoint's error), nothing
useful for CoM or Radon; all report it as covariance, positions unchanged.

## Real-data floor and weighting

`UQ-30` harness (stereo ChArUco set, rig calibration), local remainder (lens
field, per-corner target term and per-view smooth pattern removed). Floor =
value making the local-remainder σ̂₀² 1, max over the two corner sets:

| localizer | floor all / with blurred corners | shipped | σ̂₀² all / blurred | Spearman(σ, \|e\|) all / blurred |
|---|---|---|---|---|
| CenterOfMass | 0.074 / 0.074 | 0.074 | 1.00 / 1.00 | 0.13 / 0.23 |
| Förstner | 0.096 / 0.110 | 0.110 | 0.86 / 1.00 | 0.56 / 0.72 |
| SaddlePoint (both seeds) | 0.038 / 0.053 | 0.053 | 0.53 / 0.98 | 0.19 / 0.36 |
| Radon peak fit | 0.045 / — | 0.045 | 0.99 / — | 0.15 / — |

Weighting (Σ-whitened vs uniform rig calibration; held-out CV RMS and
cross-fold remainder):

| set | localizer | CV u / w | x-fold remainder u / w |
|---|---|---|---|
| all | CoM | .3788 / .3788 | .1787 / .1785 |
| all | Förstner | .4252 / .4267 | .2701 / .2719 |
| all | SaddlePoint | .3542 / .3542 | .1061 / .1062 |
| all | Radon | .3595 / .3597 | .1239 / .1246 |
| all | Radon+SP | .3537 / .3537 | .1043 / .1044 |
| blurred | CoM | .4213 / .4217 | .1781 / .1782 |
| blurred | Förstner | .5778 / .5901 | .4464 / .4519 |
| blurred | SaddlePoint | .4088 / .4093 | .1348 / .1371 |

1.5.0's SaddlePoint covariance: .4186 / .1485 on the blurred set. Weighting
never helps here; its cost falls monotonically with the floor (SaddlePoint
x-fold +2.7% at 0.038, +1.7% at 0.053, +0.5% at 0.106) because much of a
corner's error (blur bias) repeats per corner id across views.

## σ̂₀² on synthetic boards (σₙ supplied; 189 cells per localizer)

Both generators, cells 8/12, 0/8/45°, blur 0.5/1.5/2.5, σₙ 1/2/8, plus the
audit's PSF / illumination conditions. Median / 90th pct / max: SaddlePoint
0.30 / 0.94 / 2.2 (> 1.6 only under ±15% shading), CoM 0.41 / 0.93 / 1.7
(only 8°, blur 0.5), Förstner 0.59 / 1.38 / 3.5 (45° blur 1.5, coma,
perspective), Radon peak 0.80 / 2.09 / 9.3 (29 cells: 45° blur 1.5–2.5 up
to ~4, coma / decentred defocus up to 9). Estimated σₙ gives the same
medians. Harness: `sigma0_sweep` (core localizers, 0.75 px match radius);
re-measured with the shipped model in `UQ-36` (identical). Book Part VIII
quotes different medians (SP 0.14, CoM 0.37, Förstner 0.58, Radon 0.64)
from another harness, `tools/render_uncertainty_overlays.py` (bindings,
2 px match radius, 95–98 cells: lens-blur 8 px boards incl. 60° shear, the
`synth.py` Gaussian grid, and 8 other-optics conditions at σₙ 2 only);
re-run with the shipped model it reproduces them exactly. Both are right
for their mix: here 81 of the 189 cells are the audit's other-optics
conditions at three noise levels, whose medians are higher (SP 0.47, CoM
0.53, Förstner 0.84, Radon 1.02); on the Gaussian grids alone this harness
gives SP 0.14–0.15, CoM 0.29–0.32, Förstner 0.49–0.51, Radon 0.59–0.60.

## Systematic-model ablation and held-out floor (UQ-36, UQ-37, UQ-38)

### Pre-registered criteria

Written and committed before any number of this section was computed.

**Variants** (per localizer: CenterOfMass, Förstner, SaddlePoint, Radon
peak fit, Radon + SaddlePoint), all with the same noise term
`max(σₙ², 1/12)·C·(growth)` and, for the Radon peak fit, the same 8-bit
rounding term of the upsampled samples:

- **(a)** `Propagated` as shipped (knot tables over edge width, window
  offset, shipped floor).
- **(b)** noise term + isotropic floor `f_b²I`; no knots, no edge width.
- **(c)** noise term + the window-offset model with one constant
  `(g, ρ)` per localizer (all edge widths pooled in the synthetic
  calibration of `fit_systematic.py`; no edge width) + floor `f_c²I`.

`f_b`, `f_c` (and a refitted `f_a`, to check the shipped value reproduces)
are refitted by the shipped rule: the value making the local-remainder
σ̂₀² 1 on the `UQ-30` decomposition of the stereo ChArUco set, max over the
**all** and **chess** corner sets. Per-corner records come from a test-only
dump through the core localizers on the same images (positions must match
the bindings' detections).

**Gate for adopting a simpler variant for a localizer** (all required):

1. *Ranking.* Spearman(√(tr Σ/2), |e|) of the variant ≥ that of (a) − 0.02
   on the local **and** the cross-fold remainder, on both corner sets.
2. *Weighting.* Σ-whitened rig calibration with the variant: held-out
   2-fold CV residual RMS and cross-fold remainder RMS each ≤ 1.005 × the
   value with (a), on both sets.
3. *Synthetic.* On the existing σ̂₀² grid (`sigma0_sweep`: lens-blur
   renderer and `synth.py` `s0_grid_*` + `s0_audit_*`, σₙ 1/2/8, σ
   supplied, 0.75 px match radius; 189 cells), with the variant's real-data
   floor: worst |ln σ̂₀²| over the cells ≤ that of (a) + 0.2, and no cell
   with σ̂₀² > 2 under the variant unless it is > 2 under (a) too.

(b) is preferred over (c) when both pass. `edge_width` and the knot tables
are deletable only if every localizer passes with (b) or (c); otherwise the
report names the localizers that need them.

**Held-out floor (UQ-37).** For (a), (b), (c): the floor is fitted on a
training part (its own decomposition) and σ̂₀² is evaluated on the held-out
part's local remainder with the smooth and per-corner target terms
estimated on the training part only (the per-view smooth pattern is fitted
per held-out view; for a held-out camera, its smooth lens field is fitted
on that camera, the target term comes from the other). Splits: leave one
camera out (2), the 2-fold view split of the harness, leave one commanded
Rx level out (5). The floor is called transferable if every held-out
σ̂₀² lies within [0.67, 1.5] (|ln σ̂₀²| ≤ 0.4).

**End-to-end (UQ-38, evidence only).** Per variant and uniform: the rig
calibration's held-out CV RMS, cross-fold remainder RMS, fold-to-fold
|Δfx| per camera, and σ̂₀² = Σ rᵀΣ⁻¹r / (2N − p) of the weighted fit
(in-sample) and of the CV residuals. Weighting counts as a benefit if a
weighted calibration lowers the held-out CV RMS or the cross-fold
remainder by ≥ 0.5% against uniform for a localizer on either set.

**Synthetic medians (UQ-41).** The σ̂₀² medians of (a) are re-measured on
the grid above and the harness behind each published median named.

### Results

Tools: `uncertainty_eval.rs` `dump_ablation_real` / `dump_ablation_synth`
(per corner: noise term, shipped systematic term, window offset, edge
width; `--ignored`), `real_validation/ablation.py` (re-scoring, floors,
held-out floor, weighted calibrations), `calibrate.py` (fold intrinsics).
The core dump matches the bindings' detections on 99.93–100% of the
observations, and Σ(a) equals the bindings' covariance to 1e-7 relative
(one CoM corner of 13 433: 3%). The uniform and (a)-weighted rig
calibrations reproduce the tables above. Python reproduces
`SystematicModel::covariance` to 1e-6 px² on every row except the Radon
peak fit's rounding term, which (b) and (c) keep. Constant models for (c),
all widths pooled: CoM g 1, ρ 0.069; Förstner g 0.39, ρ 0.116; Radon peak
g 0.82, ρ 0.050; SaddlePoint g 1, ρ 0.007.

**Floors** (px per axis, local-remainder σ̂₀² = 1, max of **all** /
**chess**): (a) CoM 0.074, Förstner 0.110, SP 0.0525, Radon 0.045 — the
shipped values reproduce; (b) CoM 0.105, Förstner 0.277, SP 0.053, Radon
0.057; (c) CoM 0.078, Förstner 0.157, SP 0.052, Radon 0.006.

**Gate.** ✗ marks a failed condition.

| localizer | set | Spearman local a / b / c | Spearman x-fold a / b / c | weighted CV a / b / c | weighted x-fold rem a / b / c |
|---|---|---|---|---|---|
| CoM | all | .129 / .016 ✗ / −.070 ✗ | .038 / .105 / −.042 ✗ | .3788 / .3788 / .3788 | .1785 / .1787 / .1787 |
| CoM | chess | .225 / −.088 ✗ / −.088 ✗ | .103 / .041 ✗ / −.013 ✗ | .4217 / .4213 / .4213 | .1782 / .1783 / .1784 |
| Förstner | all | .564 / .059 ✗ / .512 ✗ | .553 / .124 ✗ / .440 ✗ | .4267 / .4252 / .4251 | .2719 / .2701 / .2701 |
| Förstner | chess | .718 / .509 ✗ / .521 ✗ | .701 / .506 ✗ / .494 ✗ | .5901 / .5778 / .5780 | .4519 / .4465 / .4469 |
| SaddlePoint | all | .187 / .128 ✗ / .128 ✗ | .278 / .224 ✗ / .224 ✗ | .3542 / .3542 / .3542 | .1062 / .1062 / .1062 |
| SaddlePoint | chess | .359 / .356 / .356 | .349 / .345 / .345 | .4093 / .4093 / .4093 | .1371 / .1371 / .1371 |
| Radon peak | all | .149 / .178 / .214 | .223 / .274 / .288 | .3597 / .3595 / .3595 | .1246 / .1241 / .1241 |
| Radon + SP | all | .192 / .132 ✗ / .132 ✗ | .281 / .227 ✗ / .227 ✗ | .3537 / .3537 / .3537 | .1044 / .1044 / .1044 |

| localizer | synthetic median a / b / c | worst \|ln σ̂₀²\| a / b / c | cells > 2 a / b / c (new) |
|---|---|---|---|
| CoM | 0.41 / 0.38 / 0.38 | 3.37 / 3.77 ✗ / 3.75 ✗ | 0 / 0 / 0 |
| Förstner | 0.59 / 0.52 / 0.56 | 1.74 / 2.85 ✗ / 2.84 ✗ | 3 / 3 (3 ✗) / 3 (3 ✗) |
| SaddlePoint | 0.30 / 0.31 / 0.31 | 4.08 / 4.07 / 4.07 | 2 / 2 / 2 |
| Radon peak | 0.80 / 0.90 / 0.91 | 4.01 / 4.09 / 4.09 | 20 / 30 (10 ✗) / 30 (10 ✗) |
| Radon + SP | 0.30 / 0.31 / 0.31 | 4.08 / 4.07 / 4.07 | 2 / 2 / 2 |

Synthetic (a) uses the shipped floors. CoM's |ln| failure is on the
conservative side (its smallest σ̂₀² falls). Förstner's new cells are cell
12, blur 2.5 at σₙ 1–2 (0.45–0.68 → 2.1–2.6): the gain that turns negative
at wide edges is what its knots model. Radon's are 45° blur 1.5 on both
generators and mild coma (1.2–1.9 → 2.0–2.7).

**Verdict: keep the shipped model.** Every localizer fails at least one
pre-registered condition with both (b) and (c), so `edge_width` and the
knot tables stay. The edge width is not idle on real images: it is the
only source of CoM's ranking (local Spearman .13 / .23 with it, .02 / −.09
from the noise term alone) and most of Förstner's (.56 / .72 against
.06 / .51). For SaddlePoint it moves σ by ≤ 2% (ρ 0.004–0.011 under a
0.053 floor), yet that still carries .06 of Spearman on the **all** set;
weighting cannot tell (b) from (a) there. The Radon peak fit is the one
localizer whose real-data ranking and weighting improve without the knots
(Spearman +.03 to +.06; x-fold remainder .1241 vs .1246), but dropping them
adds 10 synthetic cells above 2. Weighting with (b) or (c) is never worse
than with (a) by more than 0.1%; for Förstner it is better (CV .4252 vs
.4267, .5778 vs .5901).

**Edge width of the real corners** (SaddlePoint windows, p5 / p25 / p50 /
p75 / p95, px): **all** 1.15 / 1.18 / 1.21 / 1.26 / 1.35 on both cameras,
none outside 1.09–1.68; **chess** 1.15–1.16 / 1.19 / 1.25 / 1.40–1.42 /
1.69–1.70, 5.8–6.3% above 1.68 (held at the last knot), none below.

### Held-out floor (UQ-37)

Floor fitted on the training part; σ̂₀² of the held-out part's local
remainder, smooth and target terms from the training part. Model (a); (b)
and (c) are within ±0.13 of it except Förstner (below).

| localizer | set | floors fitted px | cam1→cam2 / cam2→cam1 | 2-fold split | leave one Rx level out (−14 / −9 / 0 / +9 / +14) |
|---|---|---|---|---|---|
| CoM | all | .046–.093 | 0.89 / **3.07** | 1.16 / 1.14 | 1.00 / 1.01 / 1.05 / 1.14 / 1.38 |
| CoM | chess | .051–.091 | 0.87 / **2.78** | 1.00 / 1.24 | 1.02 / 1.26 / 1.08 / 1.07 / 1.29 |
| Förstner | all | .080–.109 | 1.23 / **2.20** | 1.18 / 1.14 | 0.84 / 0.82 / 1.04 / 1.28 / **1.60** |
| Förstner | chess | .097–.121 | 1.24 / **1.92** | 1.14 / 1.13 | 1.01 / 1.04 / 1.11 / 1.16 / 1.35 |
| SaddlePoint | all | .038–.044 | **1.85** / **5.51** | 1.16 / 0.97 | 1.28 / 0.88 / 1.16 / 1.30 / 1.44 |
| SaddlePoint | chess | .046–.059 | 1.28 / **3.79** | 0.89 / 1.27 | 1.42 / **1.73** / 1.47 / 0.70 / 0.77 |
| Radon peak | all | .044–.050 | 1.50 / **3.40** | 1.17 / 1.08 | 1.29 / 1.06 / 1.11 / 1.15 / 1.24 |
| Radon + SP | all | .038–.044 | **1.86** / **5.51** | 1.16 / 0.97 | 1.29 / 0.88 / 1.16 / 1.30 / 1.43 |

Not transferable by the pre-registered rule (bold: outside 0.67–1.5).
Across views of one camera the floor mostly holds (0.70–1.73; 54 of 56
cells inside the band). Across cameras it does not, and not because the
floor moves (the floors fitted on either camera differ by ≤ 0.02 px): the
held-out camera's remainder is larger when the per-corner target term
comes from the other camera. The held-out camera alone would need
SaddlePoint 0.061 / 0.093 px against 0.04 in-sample, CoM 0.084 / 0.13,
Förstner 0.13–0.18, Radon 0.067 / 0.098. Part of what the target term
absorbs in-sample is repeatable per camera — localizer error tied to how
that camera images each corner — so the shipped floors, fitted after it is
removed, are low by this margin wherever that term is not known (any user
calibration). With (b), Förstner's larger floor (0.28) moves the Rx cells
to 0.64–1.56; with (c) to 0.67–1.51.

#### Cross-camera floor (UQ-44): pre-registered

Written and committed before any number below it was computed.

**Idea.** The target's manufacturing error is common to both cameras; a
camera's repeatable per-corner localizer error is not. Estimating the
per-corner target term from the *other* camera keeps the latter in the
remainder.

**Remainders** (model (a), shipped knots; per corner set **all** /
**chess**; 5×4 lens basis):

- *in-sample* `e_in`: the shipped local remainder (lens field, per-id
  target term, per-view cubic, all fitted on both cameras).
- *cross-camera* `e_x,B` for held-out camera B: target term fitted on
  camera A only, B's lens field fitted on B after removing it, per-view
  cubic per view of B; dof = the per-view cubic's share (the `UQ-37`
  camera split of `ablation.py::held_out`).

**A flaw in the plain cross-camera floor, stated beforehand.** Write the
remainder at corner `i` of camera c as `n + b_c,i` (random, and repeatable
per camera; the target term `t_i` and any error common to both cameras
`a_i` cancel). Then `e_in ≈ n` (the in-sample term absorbs `b`), but
`e_x,B ≈ n + b_B − b_A`: it carries **both** cameras' repeatable error, so a
floor whitening it double-counts `b` when the cameras are alike. The
estimand — what one camera's corner error is when the target is known —
is `n + b_B`. With exchangeable cameras its variance is the midpoint of
the two remainders' variances; the leakage of the random error into a
per-id term (3 parameters from ~21 × 2 values: ~7% of `n`'s variance,
taken from `e_in`, added to `e_x`) cancels there to first order. `a_i`
(per-corner error common to both cameras, e.g. pattern bias) is not
identifiable from target error with one board; §4 of
[real-data validation](real-data-validation.md) bounds the pattern part at
≤ 1% of the target term.

**Floors.**

- `f_x` (upper bound): σ̂₀² = 1 on `e_x` pooled over both directions.
- `f_m` (**candidate**): ½[σ̂₀²_in(f) + σ̂₀²_x(f)] = 1, σ̂₀²_x pooled over
  both directions.
- Per corner set; a table's floor is the max over **all** / **chess** and,
  for SaddlePoint, over the ChESS and Radon seeds (the shipped rule). The
  Radon + SaddlePoint value is reported against SaddlePoint's.

**Adopt `f_m` for every table only if all of the following hold;
otherwise keep the shipped floors and document the limit.**

1. *Precision.* View bootstrap (the 27 poses resampled with replacement,
   both cameras together, every term refitted; 200 replicates; table
   floor by the max rule per replicate): the 90% interval's half-width ≤
   25% of `f_m`.
2. *Direction.* With the pooled `f_m`, each held-out camera's midpoint
   ½[σ̂₀²_in,B + σ̂₀²_x,B] lies in 0.67–1.5 on both sets; and with the
   pooled `f_x`, each direction's σ̂₀²_x,B lies in 0.67–1.5 (the two
   cameras must agree on the excess for the midpoint to mean anything).
3. *Ranking and weighting.* With the new floors, the `UQ-36` gate against
   the shipped Σ: Spearman(σ, |e|) ≥ shipped − 0.02 on the local and the
   cross-fold remainder, Σ-whitened rig calibration's held-out CV RMS and
   cross-fold remainder ≤ 1.005 × the shipped-Σ value, both sets.
4. *Not lens structure.* The excess `f_x² − f_in²` must keep ≥ 50% of its
   size when every lens field uses a 10×8 basis (~200 px knots: too coarse
   for the 90 px corner pitch, fine enough for lens structure the 5×4
   basis misses). Otherwise the excess is a field over the image, not a
   corner error, and a per-corner floor should not carry it.

Reported, not gating: the difference of the two cameras' target terms
against its noise expectation and across localizers (how much of `b` is
shared by every localizer), the per-view cubic's RMS on `e_x` vs `e_in`,
and the `sigma0_sweep` medians with the new floors.

#### Cross-camera floor (UQ-44): results

Tool: `real_validation/crossfloor.py` (reuses `ablation.py`'s records and
`analyze.py`'s decomposition). `f_in` reproduces the shipped floors.

| localizer | set | shipped | f_in | f_x | f_m | σ̂₀²_x(f_x) cam1 / cam2 | midpoint(f_m) cam1 / cam2 |
|---|---|---|---|---|---|---|---|
| CoM | all | .074 | .074 | .112 | .095 | 1.29 / 0.69 | 1.29 / 0.69 |
| CoM | chess | | .074 | .105 | .091 | 1.27 / 0.71 | 1.28 / 0.71 |
| Förstner | all | .110 | .096 | .140 | .119 | 1.10 / 0.89 | 1.12 / 0.87 |
| Förstner | chess | | .110 | .163 | .139 | 1.09 / 0.90 | 1.10 / 0.90 |
| SaddlePoint | all | .053 | .038 | .079 | .062 | **1.38 / 0.59** | **1.35 / 0.63** |
| SaddlePoint | chess | | .053 | .083 | .070 | 1.31 / 0.68 | 1.28 / 0.71 |
| Radon peak | all | .045 | .045 | .085 | .068 | 1.29 / 0.69 | 1.25 / 0.73 |
| Radon + SP | all | (SP) | .038 | .079 | .062 | **1.38 / 0.59** | **1.35 / 0.62** |

Table floors (max rule): `f_x` CoM .112, Förstner .163, SP .083, Radon
.084; `f_m` CoM .095, Förstner .139, SP .070, Radon .068. Radon + SP's
values match SaddlePoint's to 0.0002, so one SaddlePoint floor serves both.

| condition | result |
|---|---|
| 1. precision (90% half-width / point) | `f_m` 5–10%, `f_x` 7–15% ✓. The percentile intervals sit above the point estimates (SP `f_m` .068–.076, Förstner .146–.174): resampling whole views repeats them, so each per-id term averages fewer independent views |
| 2. direction | ✗ SaddlePoint and Radon + SP on **all**: 1.38 / 0.59 (`f_x`), 1.35 / 0.63 (midpoint) |
| 3. ranking and weighting | ✓ Spearman unchanged (≤ 0.001; a common floor per table does not reorder σ); CV RMS ship → new .3542 → .3542 (SP all), .4093 → .4090 (SP chess), .4267 → .4262 / .5901 → .5883 (Förstner), others within ±.0002; cross-fold remainder never higher by > .0001 |
| 4. not lens structure | ✓ a 10×8 lens basis keeps 82–103% of the excess `f_x² − f_in²` |

**Verdict: keep the shipped floors.** Condition 2 fails for the
SaddlePoint table. Other findings:

- *The cameras differ, not the method.* Each camera's own in-sample floor
  differs: SaddlePoint .042 / .033, CoM .093 / .045, Förstner .109 / .080
  (cam1 / cam2, **all**). Camera 1's corners are noisier for every
  localizer, so the direction asymmetry is real camera dependence. One
  rig cannot fix a constant floor more tightly than its two cameras
  disagree.
- *Leakage is larger than assumed.* The 3-D per-id term fitted on one
  camera carries that camera's random error into the other at 0.027 /
  0.032 px per axis (SaddlePoint), 1.5× what it absorbs on its own camera
  (0.018 / 0.020). The out-of-plane component is poorly determined from
  one camera, and the 24° convergence magnifies it in the other camera. So
  the midpoint does not cancel the leakage, and `f_m` is biased upward.
  With an in-plane-only target term, leakage and absorption agree
  (0.0145 / 0.0144), but the target term then misses out-of-plane error:
  SP `f_m` .058 (**all**) / .065 (**chess**), CoM .090 / .086, Förstner
  .109 / .120, Radon .064. The directions are unchanged (SP 1.36 / 0.62).
- *The camera-specific term exists and is common to all localizers.*
  The two cameras' in-plane target terms differ by 1.53 µm RMS
  (SaddlePoint) against 0.98 µm expected from noise, an excess of ~1.2 µm
  (~0.055 px per axis summed over both cameras). Radon's difference
  explains 0.82 of SaddlePoint's, CoM's 0.69 and Förstner's 0.42. The
  excess is mostly an imaging property of each camera at each corner,
  shared by every localizer, and not lens structure (condition 4). A
  user's calibration sees it. Its size is about that of the in-sample
  floor, so the shipped floors are low by roughly √2 on optics like
  these. That factor cannot be pinned down from two cameras.
- The per-view cubic takes .096 px on `e_x` vs .082 on `e_in`
  (SaddlePoint **all**): it absorbs a little of the cross-camera excess
  as well, so the excess above is a lower bound.
- Synthetic `sigma0_sweep` medians (re-scored from the `UQ-36` dump),
  shipped → `f_m` → `f_x`: CoM 0.41 → 0.32 → 0.26, Förstner 0.59 → 0.49 →
  0.45, SP 0.30 → 0.18 → 0.13, Radon 0.80 → 0.64 → 0.50; cells > 2: SP
  2 → 0, Radon 20 → 17 → 12, Förstner 3 → 2 → 0.

A second real dataset with different optics is what would settle the
floor (`UQ-45`). Until then the shipped values stay, documented as
in-sample on one rig.

### End-to-end calibration (UQ-38)

Rig calibration per localizer (fx ≈ 21 800 px), uniform (u) vs Σ-weighted.
σ̂₀² = Σ rᵀΣ⁻¹r / (2N − p) of the weighted fit (in brackets: uniform, px²
per axis).

| localizer | set | CV RMS u / a / b | x-fold rem u / a / b | σ̂₀² fit a / b (u) | fold \|Δfx\| cam1/cam2 px, u · a · b |
|---|---|---|---|---|---|
| CoM | all | .3788 / .3788 / .3788 | .1787 / .1785 / .1787 | 6.6 / 6.5 (.072) | 80/75 · 82/72 · 80/75 |
| CoM | chess | .4213 / .4217 / .4213 | .1781 / .1782 / .1783 | 9.0 / 7.9 (.089) | 54/44 · 54/40 · 52/43 |
| Förstner | all | .4252 / .4267 / .4252 | .2701 / .2719 / .2701 | 3.6 / 1.2 (.091) | 94/89 · 58/98 · 94/90 |
| Förstner | chess | .5778 / .5901 / .5778 | .4464 / .4519 / .4465 | 3.2 / 2.1 (.167) | 41/122 · 46/58 · 36/119 |
| SaddlePoint | all | .3542 / .3542 / .3542 | .1061 / .1062 / .1062 | 21.8 / 22.2 (.063) | 91/55 · 91/55 · 91/55 |
| SaddlePoint | chess | .4088 / .4093 / .4093 | .1348 / .1371 / .1371 | 27.2 / 27.7 (.084) | 29/46 · 35/43 · 35/43 |
| Radon peak | all | .3595 / .3597 / .3595 | .1239 / .1246 / .1241 | 19.4 / 19.5 (.065) | 87/60 · 76/60 · 85/60 |
| Radon + SP | all | .3537 / .3537 / .3537 | .1043 / .1044 / .1044 | 21.8 / 22.2 (.063) | 90/55 · 89/55 · 90/55 |

(c) is within 0.0005 of (b) in CV and cross-fold remainder. No weighting
lowers the held-out CV RMS or the cross-fold remainder by 0.5% anywhere
(best: CoM cross-fold −0.1%); the shipped Σ makes Förstner worse (CV
+0.35% / +2.1%) and SaddlePoint **chess** (cross-fold +1.7%). The weighted
fit's σ̂₀² is 3–28, not 1: the shipped per-corner Σ (0.05–0.11 px) carries
4–31% of the residual variance of a full calibration, the rest being
lens-model, target and per-view error. Fold-to-fold |Δfx| (0.13–0.56% of
fx) moves with weighting as much as it differs between folds; no
consistent gain. By the pre-registered rule weighting brings no benefit on
this set.

## Cost

Per corner, 512² board (Off → Propagated): CoM 19 → 526 ns, Förstner 57 →
486 ns, SaddlePoint 207 → 569 ns. The shared edge width is ~240 ns of each,
~90 ns of it the selection of the eight largest gradients, whose summation
order (and so the width's last bit) follows `select_nth_unstable`; the CoM
Jacobian is ~220 ns. Radon peak covariance 2.6 µs per peak, only for peaks
that keep the peak fit. Per frame, the noise estimate costs 47–60 µs on the
test images. Every refined corner is reported (≥ 99%: merge and the frame
filters drop almost none), so computing covariances later would save
nothing.

Conditions: Criterion medians (`cargo bench -p chess-corners --bench
chess_pipeline -- chess_pipeline_real`, `--bench radon_pipeline`,
`--bench noise_estimate`, `-p chess-corners-core --bench refiners`), release
build, Apple M4 Pro (12 cores), rustc 1.101.0-nightly (2026-10-02), 1-minute
load 1.1 to 1.8 before each run, one bench at a time. The `rayon` rows
are re-measured after `PERF-18` parallelised the refinement: interleaved
A/B against the previous `main`, 3 s Criterion runs per binary in 6
(ChESS) or 3 (Radon) rounds, the median of the round medians, rayon's
default 12 threads, 1-minute load 2 to 6 (Radon) and 7 to 18 (ChESS:
other jobs on the machine plus the benches' own threads). The multiscale
`rayon` row and its "before" are from a later 3-round A/B (load 4 to 10).
The default is `Uncertainty::Off`; the table is the cost of opting in to
`Propagated`. Noise estimate per frame
(single-threaded call): 60 / 47 / 56 µs on small / mid / large. Per corner
(Off to Propagated, 1024² board): CoM 18 to 526 ns, Förstner 57 to 478 ns,
SaddlePoint 222 to 569 ns.

Frame time, `testimages/` small / mid / large, median:

| config | Off | Propagated | cost of opting in |
|---|---|---|---|
| ChESS single scale | 3.87 / 4.22 / 23.2 ms | 4.14 / 4.31 / 23.6 ms | +6.9 / +2.1 / +1.5% |
| ChESS multiscale | 0.80 / 0.82 / 5.77 ms | 0.90 / 0.91 / 6.04 ms | +13.5 / +10.7 / +4.5% |
| Radon single scale | 13.2 / 28.1 / 148 ms | 13.4 / 28.4 / 149 ms | +1.7 / +0.9 / +0.7% |
| ChESS single, `rayon` | 0.86 / 0.92 / 4.63 ms | 1.04 / 1.02 / 4.87 ms | +20 / +10 / +5.2% |
| ChESS multiscale, `rayon` | 0.27 / 0.28 / 1.40 ms | 0.39 / 0.37 / 1.52 ms | +47 / +35 / +8.4% |
| Radon single, `rayon` | 4.47 / 14.7 / 78.2 ms | 4.70 / 14.9 / 78.5 ms | +5.3 / +1.4 / +0.4% |

Before `PERF-18` (refinement sequential under `rayon`; same A/B runs):
ChESS single 0.85 / 0.91 / 4.60 to 1.12 / 1.01 / 5.05 ms (+32 / +11 / +9.7%),
multiscale 0.73 / 0.62 / 4.31 to 0.85 / 0.69 / 4.59 ms (+16 / +11 / +6.7%),
Radon 4.44 / 14.7 / 78.9 to 4.78 / 15.0 / 79.1 ms (+7.5 / +2.0 / +0.4%).
Multiscale frames got 2 to 3× faster, Off and Propagated alike, because the
per-seed ROI pass (patch response, refinement, covariance) runs in parallel;
the opt-in cost there stays about 0.1 ms (small 116 to 127 µs, mid 69 to
96 µs, large 287 to 117 µs), so its share of the shorter frame grew.

Dense board (synthetic anti-aliased board, 2048×1536, 48 px cells, 1376
corners, ChESS single scale, CenterOfMass): 22.70 to 23.45 ms single-threaded
(+3.3%, 546 ns per corner for the covariance and the noise estimate), 4.54
to 4.77 ms with `rayon` (+5.1%; 4.62 to 5.30 ms, +15%, while the covariance
ran sequentially). Under `rayon` a refiner splits a corner set into jobs of
about 20 µs of work (≈ 38 corners at the Propagated cost, ≈ 1000 CoM
corners at the Off cost, so Off stays on the calling thread on these
frames); the per-frame noise estimate stays sequential and is now most of
the remaining opt-in cost on small and multiscale frames. WASM (node, `testimages/mid.png`, median
of 40 runs): single scale 5.09 to 5.27 ms (+3.5%), multiscale 1.18 to 1.36 ms
(+15%).

The cost is the noise estimate plus ~0.5 µs per corner, so it is largest
on small, sparse frames: on small multiscale the noise estimate alone is
~7% of the frame single-threaded and ~20% with `rayon`. Off vs 1.5.0, p95, default build: ChESS single −4.5 /
−0.7 / −0.9%, multiscale −1.7 / −0.6 / −1.2%; Radon −0.8 / −1.7 / 0.0%.
The Radon kernel moves by a few percent with code layout alone.

## Limits

One real set for the floor (narrow-field stereo ChArUco), and it does not
transfer across that set's two cameras (`UQ-37` above; `UQ-44`: the
cameras' own floors differ by up to 2×, so one rig cannot fix it); the floor is a
constant, while the blurred corners there need more (SaddlePoint 0.053 vs
0.038). The width observable cannot see rotation, so CoM and Radon tables
are averages over rotations.
