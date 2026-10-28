# Part VIII: Uncertainty

On request, every corner the detector refines carries a 2×2 covariance
of its position, in px², in the input image frame. A calibration or pose
solver can use it to tell well-localised corners from poorly localised
ones, to propagate corner error into its parameters, and to check its
own residuals. This Part shows what the covariance describes, how it is
computed, how well it matches measured error on synthetic and real
images, where it does not, and how to use it.

## 8.1 Two kinds of error

A refined position is wrong in two ways. Image noise scatters it from
frame to frame: that is **precision**. The localizer's model of a
corner also differs from the real intensity pattern, which moves the
estimate by the same amount on every frame: that is **bias**. A
covariance that describes only the scatter is overconfident by the
bias.

![Refined positions of one corner over 600 noise draws, for four localizers, with their predicted ellipses](img/uncertainty_bias_precision.png)

Each dot is the refined position of the same synthetic corner under a
new draw of white noise, relative to the true corner (`+`). The cloud
centre (`×`) is where the localizer lands on average; its distance from
`+` is the bias. The ellipses are the covariance the detector reports,
drawn around the true corner, where the error should be centred.

- **Förstner** scatters by 0.065 px per axis but lands 0.29 px from the
  truth on this blurred corner. Its reported σ is 0.28 px per axis: the
  ellipse is wide enough to hold the whole cloud, bias included.
- **CenterOfMass** scatters by 0.017 px and lands 0.10 px off; its
  ellipse reaches the cloud.
- **SaddlePoint** lands on the truth of this symmetric corner and
  scatters by 0.020 px. Its
  ellipse is about three times wider than the cloud: on ideal synthetic
  corners the reported covariance is conservative on purpose, because
  real images add error these corners do not have (§8.3.3).
- **The Radon peak fit** scatters more (0.072 px) but has little bias
  here.

Bias does not average out over frames or corners: it depends on blur,
contrast, rotation and where the corner sits relative to the pixel
grid. Noise-free RMS error (px) of each refiner on synthetic 8 px cells
aligned with the pixel grid, by the σ of the lens blur:

| Refiner      | blur σ 0 | 0.8   | 1.5   | 2.5   |
|--------------|----------|-------|-------|-------|
| CenterOfMass | 0.085    | 0.057 | 0.061 | 0.063 |
| Förstner     | 0.056    | 0.142 | 0.289 | 0.433 |
| SaddlePoint  | 0.000    | 0.002 | 0.006 | 0.007 |

These corners are point-symmetric: straight edges and a symmetric lens
blur. `SaddlePoint` re-centres its fit on the corner (§8.3.1), which
removes the bias of a symmetric corner, not of every corner.
Illumination gradients and asymmetric lens blur bias every localizer,
`SaddlePoint` by 0.02–0.10 px (§8.5). The Radon peak fit is biased like
CenterOfMass on these boards (noise-free, up to 0.17 px;
[Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon)).

## 8.2 What the detector reports

Each `CornerDescriptor` has a `position_covariance` field, `{ xx, xy,
yy }` in px² in the input image frame
([Part I §1.4](part-01-orientation.md#14-the-cornerdescriptor-output)):
how far the reported `(x, y)` is likely to be from the true corner,
from noise and from the localizer's systematic error together. It is
reported by every localizer: the ChESS refiners `CenterOfMass`,
`Förstner` and `SaddlePoint`, the Radon peak fit, and
`RadonRefiner::SaddlePoint`. The ML refiner reports none.

`DetectorConfig::uncertainty` selects it:

- `Uncertainty::Off` (the default) reports none and skips the per-image
  noise estimate.
- `Uncertainty::Propagated` reports the covariance.

Positions are identical under both. The covariance is opt-in because it
costs frame time (§8.2.1) and weighting a real calibration by it has not
been shown to help (§8.5); its uses are flagging poorly localised
corners and putting residuals on a known scale (§8.6).

| Binding | Covariance | Tier | Noise level |
|---------|------------|------|-------------|
| Rust    | `CornerDescriptor::position_covariance: Option<PositionCovariance>` | `DetectorConfig::uncertainty` | `DetectorConfig::noise_sigma: Option<f32>` |
| Python  | `Detections.position_covariance`: `(N, 2, 2)` float32, `NaN` rows when absent | `DetectorConfig.uncertainty` (`Uncertainty.PROPAGATED` / `OFF`) | `DetectorConfig.noise_sigma` (`None` = estimate) |
| JS/WASM | `detector.positionCovariance()`: `Float32Array`, stride 3 `[xx, xy, yy]`, `NaN` when absent | `cfg.withUncertainty(Uncertainty.Propagated)` | `cfg.noiseSigma` (`undefined` = estimate) |
| C       | `cc_corner.position_covariance`, valid when `has_position_covariance` is `1` | `cc_config.uncertainty` (`CC_UNCERTAINTY_PROPAGATED` / `_OFF`) | `cc_config.has_noise_sigma` / `noise_sigma` |
| C++     | `Corner::position_covariance: std::optional<Covariance>` | `Config::uncertainty` | `Config::noise_sigma: std::optional<float>` |

In C, start from `cc_config_default()` or a strategy preset, which select
`CC_UNCERTAINTY_OFF`: a zero-initialised `cc_config` has tag `0`, which is
`CC_UNCERTAINTY_PROPAGATED`. The integer values of the tiers differ
between the C API and the Python and WASM enums; use the named constants.

### 8.2.1 Cost

`Propagated` costs a per-frame noise estimate (43–55 µs on the
test images) and about 0.5 µs per refined corner (2.6 µs per Radon
peak that keeps its peak fit). Frame time on the three test images
(small / mid / large, default build, mean):

| Configuration      | `Off`                    | `Propagated`             | Cost               |
|--------------------|--------------------------|--------------------------|--------------------|
| ChESS single scale | 3.77 / 4.15 / 22.80 ms   | 4.02 / 4.23 / 23.22 ms   | +6.7 / +2.1 / +1.8% |
| ChESS multiscale   | 0.79 / 0.80 / 5.66 ms    | 0.89 / 0.89 / 5.93 ms    | +13 / +11 / +4.8%  |
| Radon single scale | 12.7 / 28.5 / 146.5 ms   | 13.0 / 27.8 / 147.4 ms   | within ±2.5%       |

The share is largest on small, sparse frames, where the noise estimate
is a larger part of the work. The default, `Off`, does not pay it.

### 8.2.2 When there is no covariance

`None` in Rust, a `NaN` row in Python or JS, and
`has_position_covariance == 0` in C all mean the same thing:

| Cause | What to do |
|-------|------------|
| `uncertainty` is `Off`, or the refiner is the ML refiner | Use `Uncertainty::Propagated` and a classic localizer. |
| ChESS `SaddlePoint` re-centring did not converge for this corner | The position is the single integer-grid fit, which is biased. Leave the corner out, or give it a weight well below its neighbours'. A Radon corner keeps its peak fit and that fit's covariance instead. |
| The noise estimate declined (too few edge-free areas, or JPEG input; §8.3.2) | Every corner of the frame lacks a covariance. Supply `noise_sigma`. |
| `upscale` is active | Interpolation correlates the noise of the upscaled pixels, so no covariance is reported. |

## 8.3 How it is computed

\\[
\Sigma = \max\bigl(\sigma_n^2, \tfrac{1}{12}\bigr)\, J J^\top + \Sigma_{\text{sys}} .
\\]

The first term is image noise carried through the localizer (§8.3.1),
scaled by the image's noise variance (§8.3.2). The second is the
localizer's expected error beyond noise (§8.3.3).

### 8.3.1 Noise through the localizer

Every localizer is a function of the pixels it reads. \\(J = \partial
\hat x / \partial \mathbf{I}\\) is the exact derivative of the reported
position with respect to each of those pixels, including any
resampling on the way. With white pixel noise of variance
\\(\sigma_n^2\\), the first-order covariance of the position is
\\(\sigma_n^2 J J^\top\\). Only the support differs:

- **CenterOfMass** differentiates the response-weighted centroid
  through the ChESS response of every positive pixel in its window, and
  that response through its 16 ring samples, back to the image.
- **Förstner** differentiates the solution of its structure-tensor
  system through the image gradients in its window.
- **The Radon peak fit** differentiates the 3-point peak fit through the
  response blur, the ray sums and the 2× upsampling of the image. Its
  response is a squared difference, so the noise term carries a
  second-order correction for the fit's curvature.
- **SaddlePoint** is the one localizer that iterates, so its \\(J\\)
  comes from the fixed point it converges to.

The saddle-point fit shows the principle best. It fits a quadratic
\\(I(x, y) \approx a x^2 + b x y + c y^2 + d x + e y + f\\) to a
\\((2r+1)^2\\) patch \\(\mathbf{p}\\) around the seed (\\(r = 2\\) by
default). The coefficients \\(\theta = R\,\mathbf{p}\\) are a linear
least-squares fit, and the saddle lies at

\\[
u(\theta) = -H^{-1} g, \qquad
H = \begin{pmatrix} 2a & b \\\\ b & 2c \end{pmatrix}, \quad
g = \begin{pmatrix} d \\\\ e \end{pmatrix}
\\]

from the window centre. A real corner is not a quadratic: when it is
off the window centre, the fitted saddle lands partway back toward the
centre (about 70% of the true offset on sharp corners). That pull
vanishes when the window is centred on a point-symmetric corner. So the
refiner moves its window: at offset \\(o\\) it resamples the patch
bilinearly, \\(\mathbf{p}(o) = W(o)\,\mathbf{I}\\), refits, and solves
for the offset where the fitted saddle sits on the window centre,

\\[
F(o, \mathbf{I}) = u\bigl(R\,W(o)\,\mathbf{I}\bigr) = 0 ,
\\]

by damped Newton steps, until the saddle is within 0.001 px of the
centre (at most 16 steps, four to six fits per corner). At the fixed
point the implicit-function theorem gives the sensitivity to every
pixel:

\\[
J = \frac{\partial \hat o}{\partial \mathbf{I}}
= -M^{-1} \frac{\partial F}{\partial \mathbf{I}}
= -M^{-1} \frac{\partial u}{\partial \theta}\, R\, W,
\qquad M = \frac{\partial F}{\partial o} .
\\]

\\(W\\) carries the correlation that bilinear resampling introduces
between neighbouring samples. If the iteration does not converge, the
refiner returns the single integer-grid fit with no covariance.

![Covariance ellipses of two localizers across a blur gradient](img/uncertainty_board_ellipses.png)

The ellipses are the reported 1σ regions, enlarged 30×, on a synthetic
sheared board whose lens blur grows from left to right. `SaddlePoint`'s
ellipses grow slowly with blur, from 0.06 to 0.10 px at this noise
level, and tilt with the board's axes. `Förstner`'s grow from 0.15 px to
more than half a pixel: under blur its estimate stops following the
corner (§8.3.3), and the covariance says so. Core callers get the noise
term per unit noise variance, \\(J J^\top\\), as
`RefineResult::covariance_per_noise_variance`.

### 8.3.2 The image noise level

\\(\sigma_n\\) is the standard deviation of the pixel noise in gray
levels. Unless you supply it (§8.6.2), the detector estimates it once
per frame. The image is a chessboard, full of edges, so the estimator
has to measure noise without mistaking edges for it:

- It samples about 8 000 4×4 windows on a fixed grid, so the cost does
  not grow with the image size.
- In each window it takes the product of third differences along x and
  along y, which cancels any intensity that is at most quadratic along
  either axis: flat areas, shading, and straight edges along an axis.
- Windows with oblique edges or corners are rejected by a χ² gate on
  the remaining high-order components. Sensor noise grows with
  brightness, so the windows are split into four intensity bands, and a
  band whose variance stands out is treated as structure. Windows within
  12 gray levels of 0 or 255 are skipped, because clipping removes noise
  there.

On synthetic boards with white noise the estimate is within about 4%
for σ ≥ 2 gray levels and 5% for σ ≥ 1. When fewer than 512 windows are
free of edges, it gives no estimate and the frame gets no covariance;
that happens on boards that fill the frame with small, sharp cells at
low noise.

**Demosaiced and JPEG images.** The estimator assumes noise that is
independent from pixel to pixel. Demosaicing and JPEG compression
correlate it, and the estimate is then not reliable:

- On Bayer-demosaiced boards it reads the demosaicing pattern along
  edges, not the noise: about 4.2 gray levels whatever the true σ, and
  at low noise it sometimes declines.
- On JPEG boards (quality 75–90) almost every 8×8 block holds an edge,
  and below a noise σ of about 8 the estimator declines: no covariance.

On such images supply `noise_sigma`, measured on the delivered images
from two frames (§8.6.2). The noise term then assumes white noise and is
approximate, but the systematic term (§8.3.3) dominates at moderate
noise: on synthetic Bayer-demosaiced and JPEG (quality 75–90) boards
with σ supplied this way, \\(\hat\sigma_0^2\\) was at most 1.1 for
noise σ 1–8.

The estimate is available on its own, for example to display it or to
check a value you supply: `chess_corners::estimate_noise_sigma_u8`
(Rust), `chess_corners.estimate_noise_sigma` (Python),
`estimateNoiseSigma` / `estimateNoiseSigmaRgba` (JS),
`cc_estimate_noise_sigma_u8` (C) and `chess_corners::estimate_noise_sigma`
(C++).

The variance floor of 1/12 gray level² in \\(\Sigma\\) is the variance
of rounding to 8 bits. A smoothed or compressed image can measure lower
(one of the repository's test images reads σ ≈ 0.2); the floor keeps it
from getting an overconfident covariance.

### 8.3.3 The systematic term

A localizer that fits once, on a window centred on its seed pixel
(CenterOfMass, Förstner, the Radon peak fit), is pulled toward that
pixel. For a corner at offset \\(u\\) from the window centre it reports
about \\(g\,u\\), where the gain \\(g\\) depends on the localizer and on
how sharp the corner's edges are. Sharp edges give Förstner a gain of
about 0.8; as the blur grows its gain falls through zero (near a lens
blur σ of 2–2.5 px), and from there on its estimate no longer moves
with the corner. CenterOfMass has no pull on average, but a rotation-dependent
error the edge width cannot see; the Radon peak fit has a gain of
0.65–0.95.

The detector measures each corner's edge width from the pixels around
it, looks up \\(g(w)\\) and the spread of the remaining error, and
combines them with what it observes, the offset \\(\hat u\\) of its
estimate from the window centre. Not knowing where the corner sits
within the pixel, it takes that position as uniform over the pixel:

- With \\(g\\) near 1, \\(\hat u\\) pins the corner down, and
  \\(\Sigma_{\text{sys}}\\) holds the expected pull \\((1 - 1/g)\,\hat
  u\\) and a small spread.
- As \\(g\\) approaches 0, \\(\hat u\\) says nothing about where the
  corner is, and \\(\Sigma_{\text{sys}}\\) grows to the variance of a
  position anywhere in the pixel. This is Förstner on blurred corners in
  the figure above.
- `SaddlePoint` re-centres, so it has no pull; its term keeps only the
  small error it has where a corner sits between pixels (0.004–0.011 px).

The gains and spreads are fitted on noise-free synthetic boards from the
two independent generators of §8.4 (cells 6–12 px, rotations 0–45°,
shear, lens blur σ 0.3–3 px).

Synthetic corners are straight-edged and point-symmetric; real optics
and targets are not. So each localizer's term also carries a **floor**,
added to both axes, measured on real calibration images (§8.4.2):

| Localizer | Floor per axis |
|-----------|----------------|
| SaddlePoint (ChESS or Radon seeds) | 0.053 px |
| Radon peak fit | 0.045 px |
| CenterOfMass | 0.074 px |
| Förstner | 0.110 px |

The floor is why the covariance is conservative on the ideal corners of
§8.1. It is sized for the corner-local error of real images, not for
error every corner of a view shares (§8.5).

## 8.4 How well it holds

The test is the **variance factor**. Whiten each error \\(e_i\\)
(reported position minus true position) with its own covariance, and
average:

\\[
\hat\sigma_0^2 = \frac{1}{2N} \sum_{i=1}^{N} e_i^\top \Sigma_i^{-1} e_i .
\\]

It is 1 when the covariances are right, above 1 when they are
overconfident, and below 1 when they are conservative. A calibration
that weights its residuals with \\(\Sigma_i\\) sees the same factor in
its own residuals.

### 8.4.1 Synthetic boards

![Predicted against measured error per localizer, both generators](img/uncertainty_calibration.png)

Each point is one imaging condition, rendered by one of two independent
generators:

- **Lens-blur boards** (blue): straight edges, a Gaussian lens blur of
  σ 0.5–2.5 px applied before exact integration over each pixel; 8 px
  cells, grid-aligned, rotated 8° or 45°, or sheared 60°.
- **Perspective boards** (orange): a tilted board seen through a pinhole
  camera, supersampled 16×16 per pixel; Gaussian blur on a grid of
  8/12 px cells and rotations 0/8/45°.
- **Other optics** (gray): the perspective generator with a defocus
  disk, decentred defocus, coma, vignetting, an illumination ramp,
  illumination shading, and strong perspective with lens distortion.

White noise σ is 1, 2 or 8 gray levels (2 for the other-optics
conditions), supplied to the detector; estimating it gives the same
medians. Corners are matched within 2 px, so large errors count.

- **Most points sit on or above the diagonal**: median
  \\(\hat\sigma_0^2\\) is 0.14 for SaddlePoint, 0.37 for CenterOfMass,
  0.58 for Förstner and 0.64 for the Radon peak fit. The horizontal run
  of SaddlePoint points at 0.055 px is the floor: on ideal corners σ is
  overstated by up to about 3× at low noise, by design.
- **Points below the diagonal** are the limits of §8.5: the Radon peak
  fit on blurred corners (\\(\hat\sigma_0^2\\) up to 5.9), Förstner at
  45° and under coma (up to 2.2), SaddlePoint under strong illumination
  shading (1.7) and CenterOfMass on sharp boards rotated 8° (1.6).

`Radon + SaddlePoint` gives the same numbers as `SaddlePoint` wherever
its refinement converges: the re-centred fit reaches the same position
from either seed. At lens blur σ 2.5 px and noise σ 8 a few percent of
its corners do not converge and keep the peak fit, which brings its
\\(\hat\sigma_0^2\\) to 1.4–2.4 there.
`tools/render_uncertainty_overlays.py` regenerates the figures and
prints \\(\hat\sigma_0^2\\) per condition.

### 8.4.2 Real calibration images

On real images there is no ground truth, and a calibration residual
mixes the localizer's error with errors that are not the detector's:
the lens model, the target, and imaging that changes from view to
view. The measurements below come from a real stereo calibration set
on an accurate ChArUco target: about 9 500 corner observations that
every localizer finds, in well-exposed views. The residual of
a rig calibration was split into a smooth term over each camera's
image (lens model), a 3-D displacement per target corner (target
error), a smooth term per view (common to every corner of that view),
and what is left at each corner.

![Real-data residual decomposition and corner-local error per localizer](img/uncertainty_real_data.png)

- **Most of the residual is not the detector's.** For SaddlePoint the
  lens-model and target terms are 0.23 px per axis, the same for every
  localizer; the corner-local error is 0.040 px.
- **A per-view common error of about 0.06 px per axis remains**, the
  same for every localizer and largest on defocused views: imaging that
  changes with defocus and depth. No covariance of a single corner can
  carry it (§8.5).
- **The corner-local error ranks the localizers**: Radon + SaddlePoint
  and SaddlePoint lowest (0.038–0.040 px per axis), then
  `cv2.cornerSubPix` with an 11×11 window (0.045), the Radon peak fit
  (0.056), CenterOfMass (0.10) and Förstner (0.16). Including the
  blurred corners that the Radon detector misses here (about 30% of
  them) widens the gaps: SaddlePoint stays lowest, and the error over it
  is about 0.07 px per axis for CenterOfMass, 0.16 for `cornerSubPix`
  and 0.29 for Förstner.

Against the corner-local error, the reported covariances give
\\(\hat\sigma_0^2\\) between 0.5 and 1.0 for every localizer: the floors
were fitted on this set, so that agreement is by construction, not a
test. The independent evidence is how well the reported σ ranks the
error of individual corners. It does, weakly: the rank correlation
between σ and the corner-local error is 0.13–0.19 for SaddlePoint,
CenterOfMass and the Radon peak fit, and 0.56 for Förstner, rising
(0.23–0.72) when blurred corners are included. Blur is the signal the
covariance carries best.

## 8.5 Where it does not hold

- **Illumination gradients and asymmetric lens blur.** These break the
  point symmetry that re-centring relies on. Every localizer is biased;
  `SaddlePoint` by about 0.07 px per 1%-per-pixel of local illumination
  gradient (CenterOfMass and `cornerSubPix` similarly), 0.02–0.10 px in
  the conditions tested. The floor covers this except under strong local
  shading (±15% over five cells: \\(\hat\sigma_0^2\\) 1.7). The Radon peak
  fit is insensitive to shading and is the better localizer there. A
  coma that is the same across the image shifts every corner by about
  the same amount, which a calibration absorbs as a small image shift;
  one that varies over the field is only partly absorbed.
- **The Radon peak fit on blurred corners near 45°.** Blur makes its
  response flat-topped, and noise then moves the peak by up to a pixel,
  a jump no smooth noise model describes. \\(\hat\sigma_0^2\\) is
  1.2–4 at 45° from lens blur σ 1.5 px, 3.4–5.3 on every board at blur
  σ 2.5 px and noise σ 8, and 6 under coma; some corners get a very large
  covariance instead. Use `RadonRefiner::SaddlePoint` on blurred
  images.
- **CenterOfMass on sharp boards rotated a few degrees.** Its error
  depends on the board's rotation, which the edge width cannot see:
  \\(\hat\sigma_0^2\\) up to 1.6 at 8° and lens blur σ 0.5 px.
- **Förstner on blurred corners.** The covariance mostly holds (up to
  1.9 at 45° and blur σ 1.5 px, 2.2 under coma), but it grows toward a
  pixel: the position carries little information. Use SaddlePoint.
- **Small, sheared, blurred cells.** With 5 px cells, a 40–60° shear and
  blur σ ≥ 1.5 px, the saddle-point window also covers the neighbouring
  corners; the re-centred position can then be worse than the single
  integer-grid fit, and the covariance does not see this. With 4 px
  cells under a 60° shear, 19% of Radon's saddle-point corners move more
  than 0.5 px off with a confident covariance; keep Radon's default peak
  fit there.
- **Low signal-to-noise.** When the scatter itself reaches about 0.3 px,
  first-order propagation understates it by up to 1.5× in variance, and
  some detections land more than half a pixel from any true corner; no
  covariance describes that.
- **Correlated and signal-dependent noise.** On demosaiced and JPEG
  images the noise estimate is not reliable (§8.3.2); supply
  `noise_sigma`. Real sensors are noisier in bright areas; the detector
  uses one σ per image, the root mean noise variance over its edge-free
  areas.
- **Error shared by every corner of a view.** Defocus, depth-dependent
  imaging and lens-model error move all corners of a view together:
  about 0.06 px per axis on the real set of §8.4.2, beyond any per-corner
  covariance. It shows up in the downstream \\(\hat\sigma_0^2\\), not in
  \\(\Sigma\\).
- **Optics other than those the floor was fitted on.** The floors come
  from one stereo rig. The fit removed a per-corner target term, and that
  term also absorbs each camera's repeatable corner error. When the
  target term comes from the other camera, that error roughly doubles the
  excess variance: the floors would be about 1.2–1.5× larger, and the
  rig's two cameras disagree by up to 2× (CenterOfMass). Expect
  \\(\hat\sigma_0^2\\) above 1 on other optics.
- **Weighting is neutral on real data.** Weighting the real calibration
  of §8.4.2 by these covariances did not lower its held-out error
  (SaddlePoint: unchanged within 0.1%; Förstner: 0.4% higher, 2% with
  blurred corners). Most of a corner's error there repeats across views (blur
  bias, target error), and down-weighting blurred corners removes
  geometry without removing that error.

An additional term that models illumination asymmetry per corner fixes
the strong-shading case on synthetic boards but ranked real corners
worse, so it is not part of the covariance.

## 8.6 Using the covariance

### 8.6.1 Weighting a least-squares fit

To weight corners, whiten each residual \\(r_i\\) (observed corner minus
your model's prediction) with the Cholesky factor of its covariance,
\\(\Sigma_i = L_i L_i^\top\\), and minimise the sum of squares of the
whitened residuals \\(L_i^{-1} r_i\\). This equals
\\(\sum_i r_i^\top \Sigma_i^{-1} r_i\\): corners with small
uncertainty pull harder, and the anisotropy of each ellipse is
respected.

Rust:

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig, PositionCovariance, Uncertainty};

/// `L⁻¹·r` for `Σ = L·Lᵀ`: its squared norm is `rᵀ·Σ⁻¹·r`.
fn whiten(r: [f32; 2], c: &PositionCovariance) -> [f32; 2] {
    let l11 = c.xx.sqrt();
    let l21 = c.xy / l11;
    let l22 = (c.yy - l21 * l21).sqrt();
    let w0 = r[0] / l11;
    [w0, (r[1] - l21 * w0) / l22]
}

// Any classic localizer reports a covariance; SaddlePoint is the most accurate.
let cfg = DetectorConfig::chess()
    .with_chess(|c| c.refiner = ChessRefiner::saddle_point())
    .with_uncertainty(Uncertainty::Propagated);
let mut detector = Detector::new(cfg)?;
for c in detector.detect_u8(&pixels, width, height)? {
    let Some(cov) = c.position_covariance else { continue }; // see §8.2.2
    let [px, py] = project(&c); // your model's prediction for this corner
    let w = whiten([c.x - px, c.y - py], &cov);
    // add w[0] and w[1] as residuals of the least-squares problem
}
```

Python:

```python
import numpy as np
import chess_corners

cfg = chess_corners.DetectorConfig.chess().with_uncertainty(chess_corners.Uncertainty.PROPAGATED)
cfg.strategy.chess.refiner = chess_corners.ChessRefiner.saddle_point()
det = chess_corners.Detector(cfg).detect(img)

cov = det.position_covariance                      # (N, 2, 2), NaN rows when absent
ok = ~np.isnan(cov).any(axis=(1, 2))
L = np.linalg.cholesky(cov[ok])                    # Σ = L Lᵀ
r = det.xy[ok] - predicted[ok]                     # (M, 2) residuals of your model
r_white = np.linalg.solve(L, r[..., None])[..., 0] # e.g. return r_white.ravel()
                                                   # from a scipy least_squares cost
```

After the fit, \\(\sum_i \lVert L_i^{-1} r_i \rVert^2 / (2N - p)\\) for
\\(p\\) model parameters is the variance factor of §8.4. On real images
expect it well above 1 even with a good model: the lens-model, target
and per-view errors of §8.4.2 are not in \\(\Sigma\\). To propagate the
corner uncertainty into the parameters, scale the parameter covariance
by that factor. Weighting itself buys little on sharp, uniformly lit
calibration images (§8.5); the covariance earns its place there by
flagging poorly localised corners and by putting the residuals on a
known scale.

On real, well-exposed images each localizer's floor (§8.3.3) is usually
the largest part of \\(\Sigma\\), so the reported σ varies little from
corner to corner, and weighting by it has not been shown to beat
uniform weighting (§8.5).

The covariance describes the position in the image the detector saw.
If you undistort or otherwise transform corner positions before
fitting, transform the covariance with the Jacobian \\(G\\) of that
mapping: \\(G \Sigma G^\top\\).

### 8.6.2 Supplying the noise level

When the sensor noise is known, for example from two frames of a
static scene, \\(\sigma_n = \operatorname{std}(I_1 - I_2)/\sqrt{2}\\),
supply it instead of estimating it per frame. Always supply it for
demosaiced or JPEG input (§8.3.2). Give the noise as it appears in the
8-bit image the detector receives, after any gain or processing and with
the rounding to 8 bits included, as the two-frame difference measures it.

```rust
let cfg = DetectorConfig::chess()
    .with_chess(|c| c.refiner = ChessRefiner::saddle_point())
    .with_uncertainty(Uncertainty::Propagated)
    .with_noise_sigma(1.8); // gray levels; `None` estimates it per image
```

```python
cfg.noise_sigma = 1.8      # None estimates it per image
```

```js
cfg.noiseSigma = 1.8;      // undefined estimates it per image
```

```cpp
config.refiner = CC_REFINER_SADDLE_POINT;
config.uncertainty = CC_UNCERTAINTY_PROPAGATED;
config.noise_sigma = 1.8f; // std::nullopt estimates it per image
// C: cfg.has_noise_sigma = 1; cfg.noise_sigma = 1.8f;
```

A negative or non-finite value is rejected when the detector is built
(`ChessError::InvalidNoiseSigma`).

### 8.6.3 The fast path

The default tier, `Uncertainty::Off` (Python `Uncertainty.OFF`, JS
`Uncertainty.Off`, C `CC_UNCERTAINTY_OFF`), is the fast path: it skips
the noise estimate and the per-corner Jacobians (§8.2.1) and returns the
same positions. Select `Propagated` only when something reads the
covariance.

## 8.7 Orientation uncertainty

Each axis in `axes` carries a `sigma`: a fit-quality score in radians,
where larger means a less reliable axis estimate. It is not a calibrated
1σ or a covariance, uses neither the image noise level nor the
`Uncertainty` setting, and is computed whenever orientation is enabled:

- `RingFit` takes the Gauss–Newton covariance of its ring fit, scales it
  by the fit's residual variance, and multiplies it by an empirical
  factor (1.25 for clean fits, falling to 1.0 as the residual grows).
- `DiskFit` reports a residual-based estimate with a method-specific
  floor. When its axes disagree with the ring fit's, it inflates the
  sigma to at least 10° to flag the ambiguity.

The two methods use different models, so their sigmas are not
comparable. There is no term for the correlation between the two axes,
and the values have not been checked against measured angle errors the
way §8.4 checks positions. Use `sigma` to rank or gate corners within
one method, not as an inverse-variance weight that assumes it is exact.

---

Next: [Part IX](part-09-benchmarks.md) — measured accuracy and
throughput for both detectors, every refiner, and the full multiscale
pipeline.
