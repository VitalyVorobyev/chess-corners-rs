# Position covariance for every localizer (UQ-24)

Internal evidence for the `Uncertainty::Propagated` tier
([design](../design/uncertainty-v2.md) §3, §4.1, §6). Aggregates only.

Tools: `crates/chess-corners-core/src/uncertainty_eval.rs` (Monte Carlo,
σ̂₀², calibration dumps; `--ignored` tests), `tools/refiner_robustness/`
`export_uq24.py` (independent-generator scenes), `fit_systematic.py`
(synthetic tables), `real_validation/fit_floor.py` (real-data floor, on the
`UQ-30` harness), `run_real_weighted.py` (per-camera weighted calibration).
Local outputs: `bench_results/uq24/`, `tools/refiner_robustness/out/`.

## Model

`Σ = max(σₙ², 1/12)·C + Σ_sys`, `C = Σₚ gₚgₚᵀ` with `gₚ = ∂x̂/∂Iₚ`.

- **Jacobians.** CenterOfMass: through the ChESS response (`SR − DR −
  16|μₙ−μₗ|`, signs of each `|·|`) of every positive pixel of the window and
  the centroid ratio; the response map carries its ring so the refiner knows
  the response function. Förstner: `du = A⁻¹ Σ w (dg gᵀ + g dgᵀ)(p − u)`.
  Radon peak fit: 3-point fit gradient → 3×3 box blur → `(S_max − S_min)²`
  with arg-max/min rays fixed → ray sums → 2× bilinear upsample (its 8-bit
  rounding treated as identity). SaddlePoint unchanged. Central-difference
  checks: CoM 1e-4 and Radon 1e-3 (float re-implementations), Förstner 2e-3
  (±1 gray level on the real refiner), SaddlePoint 1e-3.
- **Systematic term** (`SystematicModel`, one table per localizer next to its
  estimator): observed window offset `û`, edge width `w` (intensity range over
  the radius-5 disk / (√2π · max gradient); tracks lens blur up to σ ≈ 2.5),
  gain `g(w)` and residual `ρ(w)` per knot, Bayesian pull estimate with a
  uniform prior over the grid cell, plus a real-data floor `f²I`.

## Monte Carlo noise term (whitened std, 1 = calibrated)

Lens-blur renderer, cells 8/12, 0/8/45°, blur 0.5/1.5/2.5, σₙ 1/2/8, 64
draws; independent generator: 4 scenes, 32 draws.

| localizer | where 0.9–1.1 | outside |
|---|---|---|
| SaddlePoint, Radon+SP | everywhere (0.98–1.07) | — |
| CenterOfMass, Förstner | σₙ ≤ 2 everywhere; σₙ 8 at blur ≤ 1.5 | σₙ 8, blur 2.5: 0.72–0.88 (conservative, low SNR) |
| Radon peak fit | blur 0.5 everywhere; blur 1.5 at 0/8° | blur 1.5 at 45°: 1.5; blur 2.5: 1.1–1.25 (0/8°), 2.0–2.6 (45°); independent generator 30°, σ 2: 1.24–1.41 |

The Radon failures are not linearisation error (the Jacobian is exact):
the blurred, rotated response is flat-topped and noise moves the peak pixel
by up to ~1 px. CoM/Förstner are grouped by fit window (noise can move the
seed peak, a systematic-model effect).

## Systematic tables (synthetic, both generators pooled)

Fitted per edge-width knot on noise-free corners (cells 6–12, rotations
0–45°, 60° shear, lens σ 0.3–3, ~70 k corners per localizer; the two
generators agree, e.g. Förstner gain 0.77/0.78, 0.65/0.64, 0.27/0.16,
−0.13/−0.16 at w 0.7, 1.0, 1.5, 1.9).

- Förstner: gain 0.78 → 0 (w ≈ 1.75) → −0.28; the linear pull explains
  most of its bias at w ≤ 1.3 (RMS 0.07–0.16 → 0.008–0.05 px).
- CenterOfMass: no pull (gain fits 0.89–1.07, non-monotone); its 0.04–0.09
  px error depends on the board's rotation (8°: 0.17 px, 0/45°: 0.02–0.05 at
  blur 0.3), which width cannot see; `g = 1`, `ρ` = RMS by width.
- Radon peak fit: gain 0.91–1.0 (w ≤ 1), 0.81–0.84 (1.15–1.5), 0.54 (2.1);
  largest at 45° and under shear (anisotropic `G`, off-diagonals ±0.5).
- SaddlePoint: `ρ` 0.002–0.012 px by width (replaces the fixed 0.006).

Window-offset G is strongly orientation-dependent (Förstner at blur 1.5:
0.28 at 0°, ≈0 at 45°); a per-corner measured gain (estimate at windows
shifted ±1 px) cuts Förstner's residual 3–10× at w ≤ 1.1 but not at heavy
blur, and does nothing for CoM / Radon. Not adopted (outside the design's
observables).

**Correction vs covariance.** Removing the modelled pull would lower the
noise-free RMS of Förstner 2.5–9× at w ≤ 1.3 (still 2–10× SaddlePoint's
error) and nothing useful elsewhere (CoM 0%, Radon ≤ 8%). All localizers
report it as covariance; positions unchanged.

## Real-data floor

`UQ-30` harness (stereo ChArUco set, rig calibration): the localizer's
*local remainder* (lens field, per-corner target term and the per-view
smooth pattern common to all localizers removed). Floor = value that makes
`σ̂₀²` of the local remainder 1 with the synthetic model's Σ:

| localizer | floor needed (all / with blurred corners) | shipped |
|---|---|---|
| CenterOfMass | 0.073 / 0.071 | 0.073 |
| Förstner | 0.123 / 0.151 | 0.150 |
| SaddlePoint (both seeds) | 0.038 / 0.053 | 0.053 |
| Radon peak fit | 0.050 / — | 0.050 |

With the shipped floors: local-remainder σ̂₀² 0.53–1.00 (all), 0.98–1.00
(with blurred corners); Spearman(predicted σ, |e|) on the local remainder
0.15–0.17 (SP, CoM, Radon), 0.60–0.73 (Förstner).

**Weighting** (rig calibration, Σ-whitened vs uniform; held-out CV RMS and
cross-fold remainder):

| set | localizer | CV uniform / weighted | x-fold remainder uniform / weighted |
|---|---|---|---|
| all | CoM | .3788 / .3789 | .1787 / .1785 |
| all | Förstner | .4252 / .4256 | .2701 / .2706 |
| all | SaddlePoint | .3542 / .3542 | .1061 / .1062 |
| all | Radon | .3595 / .3601 | .1239 / .1255 |
| all | Radon+SP | .3537 / .3537 | .1043 / .1044 |
| with blurred | CoM | .4213 / .4215 | .1781 / .1782 |
| with blurred | Förstner | .5778 / .5861 | .4464 / .4507 |
| with blurred | SaddlePoint | .4088 / .4093 | .1348 / .1371 |

1.5.0's SaddlePoint covariance on the same set: .4088 → .4186 and .1348 →
.1485. Weighting never helps here and the residual cost shrinks
monotonically with the floor (SaddlePoint x-fold +2.7% at floor 0.038, +1.7%
at 0.053, +0.9% at 0.08, +0.5% at 0.106): much of a corner's error
(blur bias) repeats per corner id across views, so down-weighting blurred
(peripheral) corners costs lens-model geometry. Per-camera free
calibrations on four real sets (`run_real_weighted.py`): held-out RMS
weighted − uniform within ±0.0004 px for every localizer.

## σ̂₀² on synthetic boards (σₙ supplied)

Both generators, cells 8/12, 0/8/45°, blur 0.5/1.5/2.5, σₙ 1/2/8, plus the
audit's PSF / illumination conditions (189 cells per localizer). With the
real-data floor the covariance is conservative on ideal corners: median /
90th percentile / max — SaddlePoint 0.30 / 0.94 / 2.3, CoM 0.41 / 0.99 /
1.8, Förstner 0.57 / 1.3 / 2.0, Radon peak 0.75 / 2.0 / 7.4. Above 1.6:
SaddlePoint only under ±15 % shading; CoM only at 8°, blur 0.5; Förstner
only under coma; Radon peak in 32 of 189 cells — 45° at blur 1.5–2.5 and
coma / decentred defocus. Estimating σₙ instead of supplying it gives the
same medians. The asymmetry cases are UQ-26's.

## Cost

Per corner (512² board, 3 362 seeds; Off → Propagated): CoM 18 → 580 ns,
Förstner 63 → 352 ns, SaddlePoint 208 → 382 ns; Radon peak covariance
1.9 µs per peak. Frame time, ChESS single scale (`testimages/` small / mid
/ large): 1.5.0 3.98 / 4.22 / 23.29 ms; Off 3.98 / 4.23 / 23.23 ms;
Propagated 4.35 / 4.36 / 23.82 ms (+9.2 / +3.3 / +2.3%). Multiscale
Propagated +21 / +16 / +7% over 1.5.0. Radon single scale: Off +3.4–3.6%
over 1.5.0 in the default build, but the untouched response kernel alone
moves by the same amount (+2.6%) and reverses (−1.5%) with
`codegen-units = 1`: code layout, not work; Propagated +0.8–1.7% over Off.

## Limits

One real set for the floor (narrow-field mono ChArUco); the floor is a
constant, while the blurred corners there need more (SaddlePoint 0.053 vs
0.038). The width observable cannot see rotation, so CoM and Radon tables
are averages over rotations.
