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
that keep the peak fit. Per frame, the noise estimate costs 43–55 µs on the
test images. Every refined corner is reported (≥ 99%: merge and the frame
filters drop almost none), so computing covariances later would save
nothing.

Frame time, `testimages/` small / mid / large, default build, mean (the p95
cost is within about a point of it):

| config | Off | Propagated | default-on cost |
|---|---|---|---|
| ChESS single scale | 3.77 / 4.15 / 22.80 ms | 4.02 / 4.23 / 23.22 ms | +6.7 / +2.1 / +1.8% |
| ChESS multiscale | 0.79 / 0.80 / 5.66 ms | 0.89 / 0.89 / 5.93 ms | +13.4 / +11.1 / +4.8% |
| Radon single scale | 12.69 / 28.46 / 146.5 ms | 12.95 / 27.75 / 147.4 ms | +2.0 / −2.5 / +0.6% |

The cost is the noise estimate plus ~0.5 µs per corner, so it is largest
on small, sparse frames: on small multiscale the noise estimate alone is
~7% of the frame. Off vs 1.5.0, p95, default build: ChESS single −4.5 /
−0.7 / −0.9%, multiscale −1.7 / −0.6 / −1.2%; Radon −0.8 / −1.7 / 0.0%.
The Radon kernel moves by a few percent with code layout alone (the
negative mid cost above).

## Limits

One real set for the floor (narrow-field stereo ChArUco), and it does not
transfer across that set's two cameras (`UQ-37` above); the floor is a
constant, while the blurred corners there need more (SaddlePoint 0.053 vs
0.038). The width observable cannot see rotation, so CoM and Radon tables
are averages over rotations.
