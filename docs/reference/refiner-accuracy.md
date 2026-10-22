# Refiner accuracy — bias vs precision

Splits each refiner's position error into **bias** (the error of the
noise-free estimate) and **precision** (the scatter noise adds around it),
and checks the saddle-point `covariance_per_noise_variance` against both.
This is the `UQ-01` evidence that sets the 1.5.0 uncertainty API (see
[`ROADMAP.md`](../ROADMAP.md#post-10-program-accuracy-aware-detection-uq-)).
[Results](#results) and [Findings](#findings) measure the single
integer-grid SaddlePoint fit; [After re-centring](#after-re-centring)
measures the re-centred refit that replaced it (`UQ-07`).
[Rotated boards and image formation](#rotated-boards-and-image-formation)
(`UQ-15`) moves the fixtures to lens blur before pixel integration and
sets the floor to 0.006 px; the tables before it used blur after
sampling and the 0.002 px floor.

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
  phases). Then `gaussian_blur`, then noise; `u8` after each step. The
  harness now renders `optical_chessboard` (lens blur, then pixel
  integration) instead; see [Rotated boards and image
  formation](#rotated-boards-and-image-formation).
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
the same at 90/150 within 0.015 px (Förstner differs by up to 0.1 px at
cell 5, blur 2.5) and does not change with noise (0.119 at every white σₙ,
cell 8, blur 0). CenterOfMass at cell 5 is over its accepted corners only
(36% at blur 0, 84% at blur 0.8–2.5; see finding 5).

| refiner | cell | blur 0 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|---|
| CenterOfMass | 5 | 0.298 | 0.302 | 0.288 | 0.225 |
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
5. **CenterOfMass cannot refine when its window has no positive
   response**, and now reports `IllConditioned` there (it returned the
   seed marked `Accepted` before). Seeded at the true corner, this rejects
   0–75% of corners at cell 5 axis-aligned (ChESS ring radius = cell),
   44–100% at cell 5 sheared, and 27–66% at cell 8 sheared with blur 2.5
   (≤ 0.2% at other cell-8 conditions);
   cell 12 and axis-aligned cell 8 are unaffected. The detector never
   reaches this case: its seeds are response peaks above a threshold
   clamped to ≥ 0 (0 of 93 M refinements over the test images and
   synthetic boards, cells 5–12, single-scale, pyramid, upscale and ML
   paths). Accepted results can still sit exactly on the seed at cell 5
   (up to 100% at blur 0) when a single pixel carries the positive
   response.
6. **The renderer is not the source of the clean-image floor.** The
   8×-supersampled `aa_chessboard` and the exact-area board give the same
   SaddlePoint noise-free RMS bias within 0.005 px at blur 0 (0.124 vs
   0.119; 0.059 vs 0.044 at blur 1.5); the ~0.11 px figure in
   [`refiner-comparison.md`](refiner-comparison.md) is refiner bias.

## After re-centring

The SaddlePoint refiner now moves its fit window, refitting on the patch
resampled bilinearly around each new position, until the fitted saddle
is within 0.001 px of the window centre. Each move is a Newton step on
the window offset (the sensitivity `M` below comes free from the cached
integer-patch fits), halved until it brings the saddle closer to the
window centre, so the iteration cannot cycle; at most 16 steps. When it
does not converge, the refiner returns the integer-grid fit with no
covariance. Its covariance is that of the converged estimate: the
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
  and gets worse when sheared (0.21 → 0.26 px at 40/200, 0.27 → 0.50 at
  90/150; see below).
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
  (low contrast, blur 2.5). That worst-case floor over-states the error
  everywhere else; `UQ-05` picks 0.002 px instead
  ([below](#noise-estimate-and-end-to-end-covariance)).
- **Acceptance** is decided by the first (integer-grid) fit exactly as
  before and is identical on every cell of the grid. Re-centring
  converges for ≥ 99.98% of accepted corners on every cell 8–12 fixture
  with σ_emp < 0.3 px; at low SNR down to 42% (those corners get the
  integer-grid fit and no covariance).
- **Convergence**, area-rendered boards, 400 phases each (corners that
  did not reach the tolerance / of those, hit the step cap; before →
  after): 90°, cell 8, blur 2.5, σₙ 16: 36 / 29 → 6 / 0; 60°, cell 8,
  blur 2.5, σₙ 8: 15 / 14 → 1 / 0; 40°, cell 5, clean: 104 / 104 → 0 /
  0. The plain step `o += u` contracts only by `1 − g`, and oscillated
  (factor ≈ −0.85) or crept (≈ 0.8) where the gain `g` was small.
- **Small sheared cells.** With 5 px cells, a 40–60° shear and blur
  ≥ 1.5, the converged position can be worse than the single fit
  (60°, blur 2.5: RMS 0.289 vs 0.192 px; 40°, blur 1.5: 0.415 vs
  0.379, of which 285/400 do not converge and fall back). Little of the
  5-px pattern survives the blur, the window holds parts of the
  neighbouring corners, and the true corner is no longer a fixed point:
  the saddle fitted on a window centred on it is 0.1–0.25 px off, with a
  cubic resampler as much as with the bilinear one. No per-patch
  criterion tested separates these corners from cell ≥ 8 ones without
  hurting the latter (drift from the single fit, drift relative to the
  single-fit offset, gain eigenvalues, curvature change, estimated
  interpolation error): each either misses the 60° case or makes cell
  ≥ 8 rows worse (5–20% RMS at low SNR, or 0.010 → 0.026 px on sharp
  40° cells). `radius: 1` helps there (40°, blur
  1.5: 0.104 px) but is worse on blurred cell-8 corners.

## Noise estimate and end-to-end covariance

`UQ-03` / `UQ-05`: the detector reports `CornerDescriptor::position_covariance`
= `max(σₙ², 1/12)·C + f²I` (`f = 0.002` px), with σₙ from
`DetectorConfig::noise_sigma` or `estimate_noise_sigma_u8`.

```sh
cargo test -p chess-corners-core --release --test noise_estimate -- --ignored --nocapture
cargo test -p chess-corners --release --test position_covariance -- --ignored --nocapture
```

**Estimator** (`core/src/noise.rs`): a 4×4 window expanded in the
orthogonal polynomials on four points (`s, l, e, d`, degrees 0–3); the
operator is `d ⊗ d` (cancels anything at most quadratic along an axis;
output std 20σ). The gate uses every other component of total degree ≥ 3 —
`(3,0) (0,3) (3,1) (1,3) (2,1) (1,2) (2,2) (3,2) (2,3)`, each scaled to
the operator's white-noise variance, so `q/(400σ²)` is χ²₉ — and keeps
`q < 20·400σ²` (never tighter than the 8-bit rounding variance). An edge
along the window diagonal projects only onto `(i, i)` components, so
`(2,2)` is what catches it (the earlier 4-term gate read +2 000–3 700% on
45° sharp 6 px cells). About 8 k windows are sampled once on a staggered
grid and kept in memory (gate statistic, squared operator output, by
64-level intensity band), so the threshold is applied exactly. The noise
level is a fixed point (the gate it sets yields it) reached from the
quietest sixteenth of the windows; each band reaches its own fixed point
from its own quietest windows and is dropped as structure if its variance
exceeds 8× the image-wide one or 3× the median over bands; the result is
the admitted-count weighted mean of 4σ-truncated band variances. Windows
whose mean is within 12 levels of 0 or 255 are skipped (by the mean, the
`(0,0)` component, which does not bias the operator).

Normalising each gate component per image (its own noise variance from the
image) was tried and dropped: on the test images the components are
within 1.0–1.6× of their white-noise ratios, and on boards with no flat
area every self-normalisation tried absorbed structure as "noise texture"
(+100–4 000% on 30–45° boards at σₙ ≤ 1). The earlier instability on real
images came from the binned gate threshold (a factor-2 bin jump between
G 18 and 20 moved `large.png` from 0.74 to 1.16), from too few windows,
and from bright bands with legitimately higher noise being dropped.

Worst relative error against the realised σ (std of `noisy − clean`,
rounding included), 256² boards, 40/200, mean of 3 seeds, cells 5/8/12
(exact-area axis / 60° shear) or 6/8/12 (supersampled, rotated), blur
0/0.8/1.5/2.5; `(n)` = cells where the estimator declined:

| geometry | σₙ 0.5 | 1 | 2 | 4 | 8 | sd a=1 | sd a=4 |
|---|---|---|---|---|---|---|---|
| axis | +15% (3) | +4.8% (2) | +1.4% (2) | +3.8% | +2.2% | +3.5% (1) | +2.4% (1) |
| skew60 | +12% (5) | +4.4% (5) | +1.7% (3) | +0.6% (2) | +1.3% (1) | +14% (2) | +8.7% (2) |
| rot 0° | +22% (3) | +12% (1) | +3.8% | -2.1% | +0.4% | +12% | +5.2% |
| rot 10° | +10% (4) | +4.0% (2) | +3.0% (1) | +2.5% | +1.8% | +12% (1) | +4.6% (1) |
| rot 22.5° | +12% (5) | +4.4% (3) | +2.6% (1) | +4.0% | +1.8% | +6.5% (1) | +6.6% (1) |
| rot 30° | +12% (5) | +4.1% (3) | +2.4% (2) | +3.3% | +24% | +7.8% (2) | +5.1% (2) |
| rot 45° | +11% (6) | +4.6% (4) | +2.9% (3) | +3.1% (1) | +106% | +9.8% (2) | +4.8% (2) |

Errors above 10% at σₙ ≥ 1 are cells next to a decline (few structure-free
windows; an image whose first ~8 k-window sample is short is sampled once
more, 4× as densely) or sharp 6–12 px cells at σₙ 8, where every window
holds an edge comparable to the noise (45° cell 6: +106%). σₙ 0.5 errors
are blur ≥ 1.5. Rounding is
included: σₙ 0.5 / 1 / 2 realise 0.572 / 1.044 / 2.030 (`√(σₙ² + 1/12)`
0.577 / 1.041 / 2.021).

**Real images: additivity.** No ground truth, so each `testimages/` image
gets white noise σₐ added (3 seeds); adding σₐ should raise σ̂² by
`σₐ² + 1/12` whatever the image's own noise. Ratio
`(σ̂²(img + n) − σ̂²(img)) / (σₐ² + 1/12)`, before → after:

| image | σ̂(img) before → after | σₐ 1 | σₐ 2 | σₐ 4 |
|---|---|---|---|---|
| small 720×540 | 1.34 → 1.52 | 1.36 → 0.96 | 1.07 → 0.98 | 0.99 → 0.99 |
| mid 1024×576 | 0.10 → 0.23 | 1.07 → 1.03 | 1.03 → 1.02 | 1.01 → 1.01 |
| large 2048×1536 | 1.16 → 1.55 | 0.44 → 0.86 | 1.29 → 0.95 | 1.08 → 0.98 |

Stability of σ̂(img) across gate settings G 16 / 20 / 24: small 1.53 /
1.52 / 1.52, mid 0.226 / 0.230 / 0.237, large 1.53 / 1.55 / 1.55 (before,
G 16–20: large 0.74–1.16, small 1.28–1.34); across 4 k / 8 k sampled
windows within 1%. Real images are signal-dependent: `large.png`'s band
variances run 0.7 → ~6 gray level² from dark to bright.

Cost (min of 20, M4 Pro): 0.095 / 0.074 / 0.055 ms on small / mid /
large, against a SaddlePoint multiscale detect of 0.54 / 0.83 / 4.2 ms.
Nothing is estimated unless a refined corner carries a covariance.

**Floor choice** (superseded by the `UQ-15` [floor](#rotated-boards-and-image-formation),
which adds rotated boards and lens blur). Worst |ln σ̂₀²| per candidate `f` over cells 8–12, both
contrasts and geometries, blur 0–2.5, white σₙ 1–8, converged corners: the
`UQ-01` harness with the true noise variance, and the facade end to end
(below) with the estimated one. The worst over every cell is a low-SNR
cell no floor fixes (harness: sheared 90/150 blur 2.5 σₙ 8, 1.39; end to
end: axis 90/150 blur 2.5 σₙ 8, 0.53, where 41% of detections miss the
corner by > 0.5 px), so the comparison uses cells with σ_emp < 0.3 px
(harness) / ≤ 1% missed detections (end to end):

| f (px) | harness | end to end |
|---|---|---|
| 0 | 0.278 (1.32, skew60 40/200 blur 0 σₙ 1) | 0.238 (1.27, skew60 blur 0 σₙ 1) |
| 0.002 | 0.192 (1.21, same) | **0.152** (1.17, same) |
| 0.003 | **0.182** (0.83, axis 40/200 blur 0 σₙ 1) | 0.203 (0.82, axis blur 0 σₙ 1) |
| 0.005 | 0.441 (0.64, axis) | 0.461 (0.63, axis) |
| 0.0075 | 0.808 (0.45) | 0.826 (0.44) |
| 0.01 | 1.165 (0.31) | 1.182 (0.31) |
| 0.015 | 1.784 (0.17) | 1.799 (0.17) |

`f = 0.002` px minimises the larger of the two (0.192); end-to-end σ̂₀² is
0.90–1.17 on every row whose detections match the truth. The pixel-noise
variance is clamped at `1/12` (8-bit rounding) so a smoothed or compressed
image whose measured σ is below 0.29 (`testimages/mid.png`: 0.20) does not
get an overconfident covariance; the estimate already includes rounding
on the synthetic boards (σₙ 0.5 / 1 / 2: realised 0.572 / 1.044 / 2.030 vs
`√(σₙ² + 1/12)` 0.577 / 1.041 / 2.021; estimates 0.546 / 1.032 / 2.005), so
the clamp changes no synthetic number above.

**End to end at f = 0.002** (facade, SaddlePoint, `area_chessboard` 200²,
cell 8, 25 boards per row, ~13 k matched corners): σ̂₀² =
mean(eᵀΣ⁻¹e)/2 about the truth, estimated σₙ (supplying the realised σ
changes it by ≤ 0.11).

| board | blur | σₙ 1 | σₙ 2 | σₙ 4 | σₙ 8 |
|---|---|---|---|---|---|
| axis 40/200 | 0 | 0.90 | 0.95 | 0.97 | 0.99 |
| axis 40/200 | 0.8 | 1.02 | 0.98 | 0.99 | 1.00 |
| axis 40/200 | 1.5 | 0.91† | 0.98 | 1.00 | 1.01 |
| axis 40/200 | 2.5 | 0.94 | 0.98 | 1.00 | 1.02 |
| axis 90/150 | 0 | 0.97 | 0.98 | 1.00 | 1.02 |
| axis 90/150 | 0.8 | 1.02 | 1.00 | 1.01 | 1.02 |
| axis 90/150 | 1.5 | 1.07 | 1.02 | 1.03 | 0.93* |
| axis 90/150 | 2.5 | 0.95 | 1.00 | 0.92* | 0.53* |
| skew60 40/200 | 0 | 1.17 | 1.04 | 1.00 | 0.99 |
| skew60 40/200 | 0.8 | — | — | 0.99 | 1.00 |
| skew60 40/200 | 1.5 | 0.94‡ | 1.01 | 1.01 | 1.02 |

\* low SNR: 5–41% of detections > 0.5 px off a lattice corner (excluded).
† σₙ off by more than 5%. ‡ the estimator declined on some boards (no
covariance there; at sheared blur 0.8, σₙ 1 on all 25). Sheared blur 2.5
yields too few detections at the default threshold to report.

## Rotated boards and image formation

`UQ-15`. The re-centred fit kept 0.01–0.02 px noise-free RMS error on
sharp boards rotated 2–15° and at 45° (0.0002 px at 0°, 0.002–0.003 px at
20–30°), so end-to-end σ̂₀² reached ≈ 10 at σₙ 1 on an 8° board with the
0.002 px floor. Measurements: SaddlePoint, cell 8 (cell 12 identical),
f32 renders unless stated, 25–64 subpixel phases, seed at the rounded
truth.

**Root cause: two effects.**

1. **Sampling aliasing under an ideally sharp lens.** The error is a
   function of the subpixel phase with zero mean (mean over 64 phases
   0.0000 px at every angle; at 8° the x error follows the *y* phase,
   amplitude ≈ 35 mpx). It is not the quadratic model: the same fit and
   fixed-point iteration fed exact pixel-area values at window + offset,
   instead of bilinear samples, gives 0.00 mpx at every angle (the window
   is point-symmetric about the corner). It is not 8-bit rounding (f32 and
   u8 renders agree within 0.5 mpx) or the neighbouring corners (cells 8
   and 12 agree). No estimator on the same pixels removes it (noise-free
   RMS mpx at 0/2/5/8/15/30/45°, sharp):

   | estimator | 0° | 2° | 5° | 8° | 15° | 30° | 45° |
   |---|---|---|---|---|---|---|---|
   | re-centred, bilinear (shipped) | 0.1 | 10.5 | 18.3 | 20.8 | 8.8 | 2.7 | 9.8 |
   | `radius: 1` / `3` | 0.1 | 12.3 / 9.3 | 26.0 / 14.7 | 33.0 / 12.8 | 32.6 / 0.8 | 8.0 / 1.0 | 19.8 / 6.7 |
   | pre-smoothing [1 2 1] / [1 4 6 4 1] | 0.1 / 2.9 | 11.9 / 11.7 | 20.2 / 18.8 | 21.1 / 18.3 | 7.4 / 5.8 | 3.9 / 3.5 | 5.6 / 4.5 |
   | Gaussian LS weights σ 1.0 / 1.5 | 20.1 / 9.6 | 21.8 / 13.5 | 25.9 / 19.9 | 28.0 / 22.6 | 21.0 / 13.9 | 4.7 / 4.5 | 8.7 / 7.2 |
   | Keys cubic resampling | 0.0 | 10.0 | 17.8 | 20.7 | 9.1 | 4.7 | 11.5 |
   | Lagrange cubic resampling | 0.0 | 7.2 | 17.6 | 20.9 | 9.8 | 8.4 | 4.4 |
   | moving-window fit on raw pixels, best weight (Hann, reach 3.5) | 7.7 | 12.2 | 19.0 | 21.8 | 14.1 | 2.2 | 3.8 |

   The earlier fixtures (`area_chessboard`, then `gaussian_blur`) blur
   *after* pixel sampling, which keeps this aliasing at every blur level.
   A lens blur *before* pixel integration removes it from σ_psf ≈ 0.5 px
   (shipped refiner, 25 phases):

   | formation | 0° | 2° | 5° | 8° | 15° | 30° | 45° |
   |---|---|---|---|---|---|---|---|
   | sharp | 0.1 | 7.5 | 17.8 | 20.5 | 8.6 | 2.6 | 9.9 |
   | blur after sampling, σ 1.5 | 3.8 | 9.1 | 15.6 | 13.3 | 3.8 | 2.6 | 3.3 |
   | lens σ 0.25 | 0.2 | 2.3 | 4.8 | 5.6 | 1.8 | 2.3 | 9.7 |
   | lens σ 0.5 | 0.1 | 0.2 | 0.3 | 0.5 | 1.3 | 5.4 | 8.9 |
   | lens σ 0.75 | 0.3 | 0.5 | 1.0 | 1.5 | 3.0 | 6.9 | 8.4 |
   | lens σ 1.0 | 1.2 | 1.3 | 1.8 | 2.4 | 4.2 | 7.1 | 7.8 |
   | lens σ 1.5 | 1.9 | 2.0 | 2.3 | 2.7 | 3.8 | 5.5 | 5.9 |
   | lens σ 2.5 | 1.8 | 1.8 | 1.9 | 2.1 | 2.5 | 3.5 | 3.9 |

2. **Under a lens, a 5–9 mpx residual at 30–45° from the bilinear
   resampler.** The 2-tap kernel at fraction `t` has third moment
   `t(1−t)(1−2t)` about the sample point; it breaks point symmetry where
   the oblique corner has mixed third derivatives. Keys cubic has the same
   third moment (0.094 at `t = ¼`) and does not help; a 4-point Lagrange
   cubic (reproduces cubics, third moment 0) brings every angle to
   ≤ 2.5 mpx at lens σ 0.5 and ≤ 1.3 mpx at 0.75–1.5. Prototyped in the
   refiner: 2.5× the cost, predicted σ +4–6%, worse with sharp optics at
   30° and on the 60° shear. Deferred to `UQ-16`.

**Fixture.** `optical_chessboard` / `Lattice` (testutil): Gaussian lens
PSF, then exact integration over each pixel (closed form across rows,
Gauss–Legendre down them; checked against a separable closed form, a brute
force and point symmetry). The `UQ-01` harness, the facade covariance
tests and `tools/render_uncertainty_overlays.py` use it (the Python
renderer agrees with testutil within 0.02 gray levels beyond rounding).
Re-running the `UQ-01` harness on it: SaddlePoint noise-free RMS bias,
axis, cell 8, 40/200 is 0.000 / 0.002 / 0.006 / 0.007 px at lens σ 0 /
0.8 / 1.5 / 2.5 (was 0.000 / 0.004 / 0.004 / 0.003 with blur after
sampling); CenterOfMass and Förstner change by ≤ 0.03 px.

**Floor.** `position_covariance_calibration` (facade, estimated σₙ, 200²
boards, 25 per row): rotation {0, 2, 5, 8, 15, 30, 45}° × cells {8, 12} ×
lens σ {0.5, 0.8, 1.5, 2.5} × σₙ {1, 2, 4, 8}, plus 60° shear (cells 8,
12) and axis 90/150; rows with ≥ 1 000 matched corners and ≤ 1 %
unmatched (the noise estimator declines on 45° and sheared boards at
σₙ 1–2 with sharp-ish blur). Worst |ln σ̂₀²|, with σ̂₀² [min, max]:

| f (px) | all rows | grid-aligned (rot 0°, shear, 90/150) | rotated |
|---|---|---|---|
| 0 | 1.58 (4.86: 45°, c12, lens 0.5, σₙ 1) | 0.35 [0.94, 1.42] | [0.77, 4.86] |
| 0.002 | 1.40 (4.06, same) | 0.29 [0.92, 1.33] | [0.77, 4.06] |
| 0.003 | 1.21 (3.36) | 0.21 [0.85, 1.24] | [0.76, 3.36] |
| 0.004 | 1.00 (2.72) | 0.26 [0.77, 1.12] | [0.75, 2.72] |
| 0.005 | 0.78 (2.18) | 0.37 [0.69, 1.04] | [0.70, 2.18] |
| **0.006** | **0.56** (1.75, same) | 0.50 [0.61, 1.04] | [0.61, 1.75] |
| 0.007 | 0.65 (0.52: 15°, c12, lens 0.5, σₙ 1) | 0.64 [0.53, 1.04] | [0.52, 1.43] |
| 0.008 | 0.80 (0.45) | 0.78 [0.46, 1.04] | [0.45, 1.17] |
| 0.01 | 1.10 (0.33) | 1.06 [0.35, 1.04] | [0.33, 1.09] |

`POSITION_FLOOR_PX = 0.006`. σ̂₀² at f = 0.006, cell 12, σₙ 1 / 2 / 4 / 8
(cell 8 within ±0.1 where the estimator answers, except 30°, lens 0.5,
σₙ 4: 0.73, the noise estimate reading 19% high):

| board | lens 0.5 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|
| 0° | 0.62 / 0.86 / 0.97 / 1.00 | 0.63 / 0.87 / 0.97 / 1.00 | 0.84 / 0.96 / 1.00 / 1.02 | 0.96 / 1.00 / 1.02 / 1.04 |
| 8° | 0.62 / 0.86 / 0.97 / 1.01 | 0.68 / 0.88 / 0.97 / 1.01 | 0.83 / 0.95 / 1.00 / 1.03 | 0.94 / 0.99 / 1.02 / 1.06 |
| 15° | 0.61 / 0.86 / 0.97 / 1.01 | 0.76 / 0.92 / 0.99 / 1.02 | 0.87 / 0.97 / 1.01 / 1.03 | 0.95 / 1.00 / 1.03 / 1.06 |
| 30° | 0.90 / 0.95 / 0.93 / 0.98 | 1.20 / 1.09 / 1.03 / 1.01 | 0.95 / 0.99 / 1.01 / 1.02 | 0.95 / 0.99 / 1.02 / 1.08 |
| 45° | 1.75 / 1.38 / 1.15 / 1.03 | 1.49 / 1.23 / 1.09 / 1.03 | 0.96 / 0.99 / 1.02 / 1.03 | 0.95 / 1.00 / 1.03 / 1.09 |
| shear 60° | 0.74 / 0.88 / 0.96 / 0.99 | 0.89 / 0.95 / 0.98 / 1.00 | 0.87 / 0.96 / 1.00 / 1.02 | 0.95 / 0.99 / 1.03 / 1.05 |

Costs of the floor: grid-aligned boards at σₙ 1 read 0.61–0.62 (σ
overstated 1.27×), 0.83–0.87 at σₙ 2. With ideally sharp optics (outside
the grid) an 8° board still reads 6.1 / 3.0 / 1.6 / 1.2 at σₙ 1 / 2 / 4 /
8 (was 10.2 / 3.6 / 1.7 / 1.2 at 0.002), 45° 2.1 / 1.6 / 1.2 / 1.0, and
grid-aligned 0.59 / 0.84 / 0.96 / 1.00.

Small sheared cells (cell 5, 40–60° shear, lens σ 1.5–2.5): no
detections end to end at the default threshold; at the refiner level
(36 phases) the re-centred position is still worse than the single fit
except at 50–60°/1.5 (e.g. 60°, lens 2.5: 0.162 vs 0.132 px RMS; 40°,
lens 1.5: 0.40 vs 0.23 px, 16 of 36 not converged). Unchanged by this
work (the refiner is unchanged).

Reproduce: `cargo test -p chess-corners --release --test position_covariance
-- --ignored position_covariance_calibration --nocapture` (≈ 2 min on 12
cores).

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
covariance plus a constant floor, 0.006 px per axis since `UQ-15`
([rotated boards](#rotated-boards-and-image-formation)). Förstner and CenterOfMass
covariances (`UQ-04`) are of little use while their bias is 1–7× their
scatter; document SaddlePoint as the refiner for calibration-grade work
and defer them.

**Limits.** Synthetic, straight-edged corners with a Gaussian lens PSF
before pixel integration (blur after sampling before `UQ-15`); uncorrelated
noise added after blur (no demosaicing or compression);
seeds at the rounded truth rather than the detector's NMS peak.
