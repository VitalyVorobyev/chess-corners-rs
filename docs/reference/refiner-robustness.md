# Refiner robustness beyond one image-formation model

Audit (`UQ-23`) of the refiner conclusions in `refiner-accuracy.md` and book
Parts VIII–IX, which rest on one synthetic generator (straight step edges,
isotropic Gaussian lens PSF, pixel-area integration, white Gaussian noise).
In that generator every corner is point-symmetric, so any estimator that
re-centres its window is unbiased at its fixed point. This audit re-tests the
ranking on an independent generator that breaks the symmetry, and on real
calibration data.

Scripts: `tools/refiner_robustness/` (`synth.py` generator, `estimators.py`,
`run_synth.py`, `run_real.py`, `sanity.py`, `diag_forstner.py`). Raw tables
are written to `tools/refiner_robustness/out/` (gitignored).

## Generator

16×16 supersampling per pixel; the ideal board through a homography and
radial distortion; any PSF (normalised to zero centroid); illumination;
pixel-area integration; sensor effects (Poisson-Gaussian noise, Bayer mosaic
+ bilinear demosaic, JPEG, gamma). Truth is the projected geometric corner.
On the baseline it matches `Board.render` within 0.13 gray levels (max,
0.004 mean); refiner errors on the two renders agree within 0.0002 px.

## Synthetic results

Cell 10 px, 40/200, Gaussian blur σ 1, noise σ 2, 3 boards × 16 draws.
Total RMS px (noise-free bias in parentheses). cSP-11 = `cv2.cornerSubPix`
with an 11×11 window; RC-Förstner = Förstner re-centred on its own estimate
(5×5, bilinear).

| condition | CoM | Förstner | SaddlePoint | Radon | cSP-11 | RC-Förstner |
|---|---|---|---|---|---|---|
| baseline | .099 (.097) | .209 (.202) | **.022 (.008)** | .066 (.050) | .035 (.020) | .095 (.048) |
| blur σ 2.0 | .062 (.057) | .494 (.477) | **.039 (.007)** | .250 (.121) | .071 (.013) | .525 (.336), 64% converge |
| defocus disk / decentred | .114 / .117 | .168 / .184 | **.019 / .023** | .058 / .083 | .034 / .061 | .072 / .127 |
| coma (mean shift included) | .112 (.110) | .313 | **.080 (.077)** | .221 | .194 | .395 |
| vignetting 40% / ramp ±20% | .103 / .101 | .212 / .209 | **.033 / .027** | .068 / .066 | .056 / .044 | .115 / .101 |
| shading ±5% / ±15% per 5 cells | .106 / .145 | .211 / .221 | **.040** / .103 | .066 / **.073** | .060 / .152 | .116 / .213 |
| perspective + distortion | .127 | .258 | **.023 (.010)** | .172 | .154 (78% match) | .132 |
| Poisson-Gauss / Bayer / gamma | .101 / .090 / .097 | .219 / .258 / .369 | **.033 / .027 / .022** | .083 / .100 / .076 | .050 / .043 / .038 | .140 / .152 / .169 |
| JPEG q90 / q75 | .100 / .101 | .217 / .238 | **.024 / .032** | .075 / .103 | .039 / .053 | .121 / .186 |
| combined A / B | .129 / .123 | .448 / .208 | **.105 / .041** | .349 / .101 | .282 / .078 | .643 / .173 |

- SaddlePoint has the lowest error in 18 of 19 conditions. The exception is
  strong local shading (±15% per 5 cells, local gradient up to 19% per
  cell), where the Radon peak fit is better (0.073 vs 0.103 px).
- Illumination gradients bias SaddlePoint by about 0.07 px per 1%/px of
  local gradient; cornerSubPix ≈ 0.11, CenterOfMass ≈ 0.08. Radon is
  insensitive. Gradient-based estimators are not more robust to shading.
- Under a uniform coma every estimator's bias is mostly a per-image shift
  (SaddlePoint 0.077 px, 0.008 after removing the mean), which a calibration
  absorbs; SaddlePoint's shift is the smallest. It varies by ~0.03 px with
  board roll, so a field-dependent coma is only partly absorbable.
- Perspective, distortion, gamma and Poisson noise do not hurt SaddlePoint.
  Gamma alone gives the shipped Förstner 20% outliers beyond 0.5 px.

## Re-centring versus the model

- Bilinear resampling breaks the symmetry argument on pixel data: a
  re-centred Förstner's offset at the true corner is zero only at integer
  or half-integer phases (0.030–0.038 px at blur 0.5, 0.008–0.010 at 1.5).
  SaddlePoint is far less sensitive to the same asymmetry.
- Förstner's response to a corner shift collapses with blur: gain 0.73,
  0.53, 0.22, −0.09 at blur 0.5, 1, 1.5, 2 (−1 analytically for a pure
  blurred saddle). SaddlePoint stays at 0.70–0.94.
- Re-centring removes ~75% of Förstner's bias (0.202 → 0.048 px) but leaves
  6× SaddlePoint's bias and 4× its scatter, and fails to converge at blur 2.
  A wider iterated window (cSP-11) gets within 1.5–5× of SaddlePoint's bias.
- The advantage of SaddlePoint is mostly the model, not re-centring.

## Real data

Fitted RMS / 2-fold cross-validated residual (px), OpenCV `calibrateCamera`
with k1k2k3p1p2 on corners matched by every estimator (rational and
k1k2-fixed-principal-point give the same ordering). stereo_simple: 20 views
per camera, 11×7 board, ~40 px cells, colour. Table_Calibration: 22×22
ChArUco, ~90 px cells, 27 views per camera.

| set | CoM | Förstner | SaddlePoint | Radon | cSP-11 | RC-Förstner |
|---|---|---|---|---|---|---|
| stereo L | .262/.264 | .246/.250 | **.159/.165** | .163/.170 | .189/.195 | .209/.216 |
| stereo R | .272/.280 | .229/.239 | **.156/.169** | .165/.178 | .203/.215 | .212/.222 |
| table cam1 | .385/.387 | .436/.436 | **.355/.355** | .359/.360 | .356/.356 | .377/.377 |
| table cam2 | .365/.366 | .406/.408 | **.348/.349** | .353/.354 | .349/.349 | .360/.361 |

- Implied extra error over SaddlePoint: Förstner 0.17–0.25 px, CoM
  0.11–0.22, Radon 0.04–0.06, cSP-11 0.10–0.13 (stereo) / 0.02–0.03 (table).
- SaddlePoint's lead over the runner-up is small (0.004–0.009 px RMS) next
  to the board and model error (0.1–0.27 px per axis), so differences are
  reliable in sign, not in size. Radon detects 69–70% of the ChArUco corners.

## Covariance calibration (SaddlePoint, σ̂₀²)

| condition | σ̂₀² |
|---|---|
| baseline, blur 0.5 / 2, defocus disk, perspective, Poisson | 0.86–0.99 |
| gamma | 0.62 |
| decentred defocus / ramp / vignetting | 1.45 / 1.53 / 1.91 |
| shading ±5% / ±15% | 3.3 / 22 |
| mild coma / coma (incl. absorbable shift) | 6.3 / 14 |
| Bayer, estimated σₙ (covariance on 73% of frames) / σₙ supplied | 0.33 / 2.36 |
| JPEG q90 / q75, σₙ supplied (estimator declines on every frame) | 0.95 / 1.23 |
| combined A / B, σₙ supplied | 10 / 1.34 |
| real data (includes board and model error) | 319–938 |

On the real sets the predicted σ is 0.006–0.009 px, essentially the
position floor, and the residual does not track it. On well-exposed,
sharp calibration images the noise term is negligible and the error is
systematic; weighting by the 1.5.0 covariance carries no information there.

## Consequences

1. The refiner ranking holds; "near-zero bias" holds only for
   point-symmetric corners. Illumination gradients and asymmetric PSFs bias
   every estimator, SaddlePoint by 0.02–0.10 px.
2. Re-centring Förstner is not worth shipping: it stays dominated by
   SaddlePoint and is ill-conditioned under blur.
3. A useful covariance must model the systematic error per corner; noise
   propagation alone is calibrated only on symmetric, noisy images.
4. The noise estimator must handle demosaiced and JPEG images (it misreads
   the former and declines on the latter).

## Limits

One contrast (40/200) and one cell size (10 px) in the synthetic sweep; a
spatially uniform PSF per image; shading amplitudes are a judgement call
(bias scales roughly linearly with gradient). A per-estimator decomposition
of the real-data error from pairwise disagreements contradicted the
calibration residuals and is not used.
