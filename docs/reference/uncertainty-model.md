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
medians.

## Cost

Per corner, 512² board (Off → Propagated): CoM 19 → 675 ns, Förstner 58 →
569 ns, SaddlePoint 228 → 625 ns; the shared edge width is ~350 ns of each.
Radon peak covariance 2.7 µs per peak, only for peaks that keep the peak
fit. (Before this pass: CoM 580, Förstner 352, SaddlePoint 382 ns with a
non-robust width; the tile removed per-pixel clamped sampling from the CoM
Jacobian, the robust width added the rest.)

Frame time, `testimages/` small / mid / large, default build, mean:

| config | Off | Propagated | default-on cost |
|---|---|---|---|
| ChESS single scale | 4.00 / 4.26 / 23.47 ms | 4.38 / 4.41 / 24.12 ms | +9.4 / +3.5 / +2.8% |
| ChESS multiscale | 0.81 / 0.83 / 5.85 ms | 0.99 / 0.97 / 6.25 ms | +21.6 / +16.7 / +6.8% |
| Radon single scale | 13.59 / 29.09 / 151.3 ms | 13.93 / 29.30 / 152.0 ms | +2.5 / +0.7 / +0.4% |

Off vs 1.5.0, p95, `codegen-units = 1` (both trees): ChESS +0.8 / +0.8 /
+0.2%; Radon −2.0 / −4.4 / −2.0%. In the default build the Radon kernel
alone moves by ±2.6% with code layout (it reverses with one codegen unit).

## Limits

One real set for the floor (narrow-field mono ChArUco); the floor is a
constant, while the blurred corners there need more (SaddlePoint 0.053 vs
0.038). The width observable cannot see rotation, so CoM and Radon tables
are averages over rotations.
