# Refiner accuracy — bias vs precision

Splits each refiner's position error into **bias** (the error of the
noise-free estimate) and **precision** (the scatter noise adds around it),
and checks the saddle-point `covariance_per_noise_variance` against both.
This is the `UQ-01` evidence that sets the 1.5.0 uncertainty API (see
[`ROADMAP.md`](../ROADMAP.md#post-10-program-accuracy-aware-detection-uq-)).
[Results](#results) and [Findings](#findings) measure the single
integer-grid SaddlePoint fit; [After re-centring](#after-re-centring)
measures the re-centred refit that replaced it (`UQ-07`).

## Reproduce

```sh
cargo test -p chess-corners-core --release --test refiner_bias_precision \
    -- --ignored --nocapture
```

Prints the tables below and writes `out/uq01/refiner_accuracy.json`
(gitignored; per-cell metrics, per-phase bias vectors and linear-gain fits).
About 4 s wall / 45 s CPU on a 12-core M4 Pro. Deterministic: fixed seeds,
and results do not depend on the thread split.

## Method

- **Fixture**: `area_chessboard` (testutil), a chessboard rendered by exact
  pixel-area integration, so the corner sits exactly on the lattice. 41×41
  image, corner at `(20, 20) + (kx, ky)/5`, `kx, ky ∈ 0..5` (25 subpixel
  phases). Then `gaussian_blur`, then noise; `u8` after each step.
- **Grid**: refiners CenterOfMass, Förstner, SaddlePoint (default configs;
  CenterOfMass reads the ChESS response of the same noisy image; ML out of
  scope) × geometry {axis-aligned, sheared to a 60° axis angle} × cell
  {5, 8, 12} px × contrast {40/200, 90/150} × blur σ {0, 0.8, 1.5, 2.5} ×
  noise {white σₙ 0, 1, 2, 4, 8; signal-dependent `sd1`, `sd4`}. Nothing
  pruned.
- **Signal-dependent noise**: variance `a + b·I` with `b = 8/255` and `a = 1`
  (`sd1`: 1 GL² at black, 9 at white; 2.3 / 7.3 at 40 / 200) or 4× that
  (`sd4`). White-noise variance is σₙ² + 1/12 (8-bit rounding).
- **Draws**: 200 per (condition, phase), 1 at noise 0. Seed = truth rounded
  to the nearest pixel.
- **Metrics** (per refiner × condition, pooled over phases): RMS bias =
  RMS over phases of the per-phase mean error, with each phase's own draw
  noise σ²/n subtracted; σ = pooled within-phase std (radial `sqrt(σx² + σy²)` in the
  tables); acceptance rate. SaddlePoint only, with `Σ = v · cov` and `v` the
  true noise variance (for signal-dependent noise, the mean of `a + b·I` over
  the 5×5 fit patch): `z_prec = (e − phase bias)/σ_pred`,
  `z_total = e/σ_pred`, and **σ̂₀² = mean(eᵀΣ⁻¹e)/2** — the a-posteriori
  variance factor a calibration that whitens with `Σ` would see (1 when `Σ`
  is right).

## Results

**Noise-free RMS bias (px)**, axis-aligned, 40/200. SaddlePoint's bias is
the same at 90/150 within 0.015 px (CenterOfMass and Förstner differ by up
to 0.1 px at cell 5, blur 2.5) and does not change with noise (0.119 at
every white σₙ, cell 8, blur 0).

| refiner | cell | blur 0 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|---|
| CenterOfMass | 5 | 0.400 | 0.358 | 0.348 | 0.306 |
| CenterOfMass | 8 | 0.085 | 0.065 | 0.061 | 0.065 |
| CenterOfMass | 12 | 0.085 | 0.063 | 0.030 | 0.015 |
| Förstner | 5 | 0.056 | 0.107 | 0.140 | 0.137 |
| Förstner | 8–12 | 0.056 | 0.111 | 0.276 | 0.44–0.50 |
| SaddlePoint | 5 | 0.119 | 0.090 | 0.054 | 0.062 |
| SaddlePoint | 8–12 | 0.119 | 0.090 | 0.044 | 0.022–0.024 |

**RMS bias / σ (px), white noise**, axis-aligned, cell 8, 40/200.

| refiner | blur | σₙ=1 | σₙ=2 | σₙ=4 | σₙ=8 |
|---|---|---|---|---|---|
| CenterOfMass | 0 | 0.083 / 0.011 | 0.081 / 0.022 | 0.077 / 0.044 | 0.072 / 0.089 |
| CenterOfMass | 1.5 | 0.061 / 0.014 | 0.059 / 0.026 | 0.053 / 0.050 | 0.039 / 0.095 |
| Förstner | 0 | 0.056 / 0.010 | 0.057 / 0.020 | 0.057 / 0.041 | 0.060 / 0.082 |
| Förstner | 1.5 | 0.276 / 0.041 | 0.277 / 0.079 | 0.281 / 0.156 | 0.289 / 0.290 |
| SaddlePoint | 0 | 0.119 / 0.008 | 0.119 / 0.015 | 0.119 / 0.030 | 0.119 / 0.060 |
| SaddlePoint | 0.8 | 0.090 / 0.010 | 0.090 / 0.019 | 0.089 / 0.038 | 0.089 / 0.075 |
| SaddlePoint | 1.5 | 0.044 / 0.016 | 0.044 / 0.033 | 0.046 / 0.064 | 0.043 / 0.130 |
| SaddlePoint | 2.5 | 0.024 / 0.037 | 0.023 / 0.072 | 0.020 / 0.144 | 0.000 / 0.314 |

**SaddlePoint σ̂₀² about the truth** (median over phases of |bias|/σ_pred in
parentheses), axis-aligned, cell 8.

| contrast | blur | white 1 | white 2 | white 4 | white 8 | sd1 | sd4 |
|---|---|---|---|---|---|---|---|
| 40/200 | 0 | 222 (17.4) | 59 (8.9) | 16 (4.5) | 4.9 (2.2) | 51 (8.2) | 13 (4.1) |
| 40/200 | 0.8 | 81 (10.0) | 22 (5.2) | 6.5 (2.6) | 2.4 (1.3) | 19 (4.8) | 5.7 (2.4) |
| 40/200 | 1.5 | 7.9 (2.8) | 3.0 (1.5) | 1.6 (0.7) | 1.2 (0.3) | 2.6 (1.3) | 1.5 (0.6) |
| 40/200 | 2.5 | 1.5 (0.6) | 1.2 (0.3) | 1.0 (0.2) | 0.9 (0.1) | 1.1 (0.3) | 1.0 (0.1) |
| 90/150 | 0 | 32 (6.4) | 9.4 (3.3) | 3.2 (1.6) | 1.6 (0.8) | 8.0 (3.0) | 2.8 (1.5) |
| 90/150 | 0.8 | 12 (3.7) | 4.0 (1.9) | 1.8 (0.9) | 1.2 (0.4) | 3.5 (1.7) | 1.7 (0.8) |
| 90/150 | 1.5 | 1.7 (0.9) | 1.2 (0.4) | 1.1 (0.2) | 0.9 (0.1) | 1.2 (0.4) | 1.0 (0.2) |

**SaddlePoint scatter calibration** over all 24 fixtures (both geometries,
all cells, contrasts, blurs).

| noise | z_prec std min / median / max | σ_pred/σ_emp max | accept min |
|---|---|---|---|
| white 1 | 0.68 / 1.00 / 1.02 | 1.88 | 0.70 |
| white 2 | 0.64 / 0.99 / 1.03 | 2.12 | 0.58 |
| white 4 | 0.59 / 0.99 / 1.01 | 2.26 | 0.51 |
| white 8 | 0.53 / 0.96 / 1.02 | 2.55 | 0.44 |
| sd1 | 0.64 / 0.99 / 1.12 | 2.20 | 0.56 |
| sd4 | 0.59 / 0.98 / 1.12 | 2.32 | 0.50 |

Wherever σ_emp < 0.3 px, z_prec std is 0.92–1.03 (408 axis values),
except the sheared board under signal-dependent noise: there the axes split
(x 0.88–0.98, y 1.01–1.12, widest at 40/200, blur ≤ 0.8) while σ̂₀² prec
stays 0.96–0.97. Every z_prec below 0.9 elsewhere is a low-SNR cell
(σ_emp ≥ 0.34 px; cell 5 or blur ≥ 1.5 at 90/150 or σₙ = 8).

**Bias structure.** The per-phase noise-free bias fitted as a linear
function of the corner's offset from the fit-window centre,
`b = (G − I)·u`, `u = truth − round(truth)`. Cell 8, 40/200; RMS bias →
RMS left after removing the linear part (G).

| refiner | geometry | blur 0 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|---|
| SaddlePoint | axis | 0.119 → 0.012 (0.70·I) | 0.090 → 0.012 (0.78·I) | 0.044 → 0.007 (0.89·I) | 0.024 → 0.007 (0.94·I) |
| SaddlePoint | skew60 | 0.150 → 0.015 | 0.104 → 0.013 | 0.053 → 0.010 | 0.026 → 0.015 |
| Förstner | axis | 0.056 → 0.018 (0.87·I) | 0.111 → 0.011 (0.72·I) | 0.276 → 0.011 (0.31·I) | 0.442 → 0.016 (−0.10·I) |
| Förstner | skew60 | 0.118 → 0.030 | 0.169 → 0.014 | 0.328 → 0.010 | 0.500 → 0.028 |

CenterOfMass does not fit this model (cell 8: 0.085 → 0.048 at blur 0,
0.061 → 0.060 at blur 1.5).

## Findings

1. **Bias dominates at realistic noise (σₙ 1–3) for every refiner, unless
   the image is heavily blurred.** SaddlePoint, cell 8, 40/200, σₙ = 2: RMS
   bias / σ = 8× (blur 0), 4.7× (0.8), 1.3× (1.5), 0.3× (2.5); a calibration
   whitening with the noise-only covariance would see σ̂₀² = 59, 22, 3.0,
   1.2 (σ under-stated by 7.7×, 4.7×, 1.7×, 1.1×). At 90/150 the factors
   shrink (σ̂₀² 9.4, 4.0, 1.2) because σ grows, not because bias falls.
   Förstner's bias is 2.4–6.8× its scatter at σₙ 1–2 at every blur, and
   grows with blur (0.056 → 0.44 px). CenterOfMass: 1.1–7.7× (cell 8).
2. **σ_pred is calibrated for scatter, including under signal-dependent
   noise.** z_prec std 0.92–1.03 wherever σ_emp < 0.3 px, white or
   signal-dependent (mean patch variance), median 1.00. Two deviations: at
   low SNR (σ_emp ≳ 0.35 px) σ_pred over-predicts by up to 2.5× —
   conservative, from the 1.5 px offset gate truncating the tails and the
   first-order linearisation breaking down; and on the sheared
   high-contrast board the patch-mean variance mis-splits the axes by up to
   ±12% while the 2-D σ̂₀² stays ≈ 0.97. Propagating per-pixel variance
   through the fit would remove the split.
3. **SaddlePoint bias is systematic and almost entirely modelable**: a
   pull toward the fit-window centre, `û ≈ G·u`. For axis-aligned corners
   `G = g·I` with `g` set by edge sharpness — 0.70 (blur 0), 0.78, 0.89,
   0.94 (blur 2.5) — and not by contrast or cell size (5–12 px, except
   cell 5 at blur 2.5); sheared corners give an anisotropic `G`. The linear
   model leaves 0.007–0.015 px. It is zero at
   phase 0 by point symmetry, so it vanishes when the window is centred on
   the corner. Förstner shows the same structure but its gain collapses
   with blur (0.87 → −0.10: at blur 2.5 it overshoots past the seed), a
   mismatch between its straight-edge-line model and the hyperbolic
   gradient field near a blurred X-junction. CenterOfMass bias is not a
   function of the window offset.
4. **Re-centring removes SaddlePoint's bias** — implemented and measured
   in [After re-centring](#after-re-centring).
5. **CenterOfMass reports `Accepted` without refining when its window has
   no positive response.** At cell 5 (ChESS ring radius = cell) it returns
   the seed itself for 36–100% of corners, all marked `Accepted`, so its
   100% acceptance hides a non-refinement. Cell ≥ 8 is unaffected (only
   the exact phase-0 corner lands on the seed, legitimately). Reported, not
   fixed here.
6. **The renderer is not the source of the clean-image floor.** The
   8×-supersampled `aa_chessboard` and the exact-area board give the same
   SaddlePoint noise-free RMS bias within 0.005 px at blur 0 (0.124 vs
   0.119; 0.059 vs 0.044 at blur 1.5); the ~0.11 px figure in
   [`refiner-comparison.md`](refiner-comparison.md) is refiner bias.

## After re-centring

The SaddlePoint refiner now refits on the patch resampled bilinearly
around its previous estimate until a step is shorter than 0.001 px (at
most 16 steps). Its covariance is that of the converged estimate: the
fixed point `F(o, I) = 0` of the fitted offset `F` at window offset `o`
gives `∂ô/∂I = −M⁻¹ ∂F/∂I`, `M = ∂F/∂o`, with `∂F/∂I` carried through
the bilinear weights (correlated resampled pixels). A unit test checks
it against finite differences of the full iterated estimate.

Cell 8, axis-aligned, white noise; before → after. σ_pred is within 2%
of σ_emp in every row except 90/150, blur 2.5, σₙ = 4 (+7%).

| contrast | blur | noise-free RMS bias | σ_emp at σₙ = 2 | z_prec std (σₙ 1–4) | σ̂₀² σₙ = 1 / 2 / 4 |
|---|---|---|---|---|---|
| 40/200 | 0 | 0.119 → 0.000 | 0.015 → 0.019 | 1.00–1.01 | 222 / 59 / 16 → 1.00 / 1.01 / 1.02 |
| 40/200 | 0.8 | 0.090 → 0.004 | 0.019 → 0.021 | 1.00–1.02 | 81 / 22 / 6.5 → 1.11 / 1.04 / 1.00 |
| 40/200 | 1.5 | 0.044 → 0.004 | 0.033 → 0.030 | 1.01–1.02 | 7.9 / 3.0 / 1.6 → 1.06 / 1.04 / 1.02 |
| 40/200 | 2.5 | 0.024 → 0.003 | 0.072 → 0.061 | 1.00–1.01 | 1.5 / 1.2 / 1.0 → 1.03 / 1.04 / 1.02 |
| 90/150 | 0 | 0.119 → 0.001 | 0.040 → 0.049 | 1.00–1.01 | 32 / 9.4 / 3.2 → 1.01 / 1.01 / 1.00 |
| 90/150 | 0.8 | 0.088 → 0.007 | 0.051 → 0.055 | 1.00–1.02 | 12 / 4.0 / 1.8 → 1.04 / 1.01 / 1.05 |
| 90/150 | 1.5 | 0.036 → 0.014 | 0.087 → 0.079 | 1.00–1.01 | 1.7 / 1.2 / 1.1 → 1.13 / 1.04 / 1.04 |
| 90/150 | 2.5 | 0.031 → 0.013 | 0.195 → 0.161 | 0.99–1.01 | 1.1 / 1.0 / 0.9 → 1.04 / 1.03 / 1.14 |

- **Bias.** Noise-free RMS bias at 40/200, cells 8–12: ≤ 0.004 px
  axis-aligned and ≤ 0.010 px sheared at blur ≤ 1.5 (was 0.044–0.150).
  The 0.013–0.016 px left at 90/150 with blur is 8-bit rounding of the
  low-contrast edge profile. Cell 5 keeps 0.03 px at blur 2.5 (axis)
  and gets worse when sheared (0.21 → 0.26 px): the window then
  overlaps the neighbouring corners.
- **Scatter.** σ grows by up to ~1/g (25% sharp, 0.019 vs 0.015 px)
  where the single fit shrank the offset by `g`, and falls up to 25% at
  blur ≥ 1.5, where the resampled window averages pixels.
- **Calibration.** Wherever σ_emp < 0.3 px, z_prec std is 0.97–1.06
  except the known axis split on the sheared board under
  signal-dependent noise (0.85–1.10). At low SNR (σ_emp ≥ 0.3 px)
  σ_pred now under-predicts slightly (σ̂₀² prec up to 1.5) instead of
  over-predicting up to 2.5× in σ, and a near-singular `M` gives an
  occasional very large σ_pred.
- **Model-error floor.** The smallest constant `f` (px per axis) for
  which `Σ + f²I` brings σ̂₀² about the truth within 0.05 of the
  scatter-only value, over fixtures with σ_emp < 0.3 px: 0.015 px covers
  every cell 8–12 fixture (median 0); cell 5 needs up to 0.03 px
  (low contrast, blur 2.5). This is the floor `UQ-05` adds.
- **Acceptance** is decided by the first (integer-grid) fit exactly as
  before; a refit that fails or leaves the `max_offset` box ends the
  iteration at the previous estimate. Acceptance is identical on every
  cell of the grid.

## Recommendation for 1.5.0

**Reduce SaddlePoint bias first, then ship its noise covariance with a
small constant model-error floor; do not ship a noise-only covariance as
the corner uncertainty as-is.** With the current refiner, a calibration
that whitens with the noise covariance would be overconfident by up to
15× in σ at realistic noise and blur ≤ 1.5 (1.1× only at low contrast and
blur 1.5) — worse than uniform weighting,
because the weights would track noise level while the error tracks blur.
A per-corner model-error term is not needed: the bias is deterministic and
removable (findings 3–4), and a learned or blur-indexed bias term would
need a blur estimate the detector does not have. `UQ-07` (re-centred
refit, covariance of the converged estimate) has brought σ̂₀² to ≈ 1
([After re-centring](#after-re-centring)); `UQ-05` exposes that
covariance plus a constant floor of 0.015 px per axis. Förstner and CenterOfMass
covariances (`UQ-04`) are of little use while their bias is 1–7× their
scatter; document SaddlePoint as the refiner for calibration-grade work
and defer them.

**Limits.** Synthetic, straight-edged corners with a box-plus-Gaussian
PSF; uncorrelated noise added after blur (no demosaicing or compression);
seeds at the rounded truth rather than the detector's NMS peak.
