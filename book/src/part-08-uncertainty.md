# Part VIII: Uncertainty

With the `SaddlePoint` refiner, every refined corner carries a 2×2
covariance of its position, in px², in the input image frame. A
calibration or pose solver can use it to weight each corner by how well
it is actually localised, instead of trusting every corner equally.

A covariance is only useful if it describes the error you actually
have. This Part shows what the covariance describes and how it is
computed, where it matches the measured error and where it does not,
and how to use it.

## 8.1 Bias and precision

A refined position is wrong in two ways. Image noise scatters it from
frame to frame: that is **precision**. The refiner's model of a corner
also differs from the real intensity pattern, which moves the estimate
by the same amount on every frame: that is **bias**. A noise
covariance describes only the scatter.

![Refined positions of one corner over 600 noise draws, for three refiners](img/uncertainty_bias_precision.png)

Each dot is the refined position of the same synthetic corner, under a
new draw of white noise, relative to the true corner (`+`). The cloud
centre (`×`) is where the refiner lands without noise; its distance
from `+` is the bias. The spread of the cloud is the precision.

- **Förstner** scatters by 0.06 px per axis but lands 0.25 px from the
  truth on this blurred corner. A covariance built from its scatter (the
  dotted ellipse) would claim 0.06 px for a corner that is four times
  further off.
- **CenterOfMass** has a smaller bias here, but it is still about
  twice its scatter.
- **SaddlePoint** lands on the truth. Its predicted 1σ and 2σ ellipses
  (zoomed in the last panel) are centred on the true corner and contain
  close to the expected fractions of the cloud (35% and 82%, against 39%
  and 86% for a normal distribution).

Bias does not average out over frames or corners: it depends on blur,
contrast and where the corner sits relative to the pixel grid. Noise-free
RMS error of each refiner on synthetic 8 px cells aligned with the pixel
grid:

| Refiner      | blur σ 0 | 0.8   | 1.5   | 2.5   |
|--------------|----------|-------|-------|-------|
| CenterOfMass | 0.085    | 0.065 | 0.061 | 0.065 |
| Förstner     | 0.056    | 0.111 | 0.276 | 0.442 |
| SaddlePoint  | 0.000    | 0.004 | 0.004 | 0.003 |

At noise σ of 1–2 gray levels the scatter of every refiner is
0.01–0.08 px, so the bias of CenterOfMass and Förstner is one to eight
times their scatter. A covariance that describes only noise would make
a calibration overconfident by that factor. This is why only
`SaddlePoint` reports a covariance: once its bias is removed (§8.3),
the noise is what remains, with the exceptions listed in §8.4.1.

## 8.2 What the detector reports

Each `CornerDescriptor` has a `position_covariance` field. It holds
`{ xx, xy, yy }` in px², in the input image frame
([Part I §1.4](part-01-orientation.md#14-the-cornerdescriptor-output)). It describes
how far the reported `(x, y)` is likely to be from the true corner.

![Covariance ellipses across a blur gradient](img/uncertainty_board_ellipses.png)

The ellipses are the predicted 1σ regions, enlarged 150×. The board is
synthetic, with blur growing from left to right and one noise level.
For sharp corners they predict about 0.02 px. At blur σ 2.5 px the same
noise moves the corners about three times further. The ellipses tilt with the
board's axes, because a sheared corner is localised better along some
directions than others.

The covariance scales with the image noise. By default the detector
estimates the noise from each image (§8.3.3); you can also supply it
(§8.5.2). Under the same noise, the covariance grows with blur and with
lower contrast, and shrinks when the board's edges are sharp.

| Binding | Covariance                                                                 | Noise level                                       |
|---------|----------------------------------------------------------------------------|---------------------------------------------------|
| Rust    | `CornerDescriptor::position_covariance: Option<PositionCovariance>`        | `DetectorConfig::noise_sigma: Option<f32>`        |
| Python  | `Detections.position_covariance`: `(N, 2, 2)` float32, `NaN` rows when absent | `DetectorConfig.noise_sigma` (`None` = estimate) |
| JS/WASM | `detector.positionCovariance()`: `Float32Array`, stride 3 `[xx, xy, yy]`, `NaN` when absent | `cfg.noiseSigma` (`undefined` = estimate) |
| C       | `cc_corner.position_covariance`, valid when `has_position_covariance` is `1` | `cc_config.has_noise_sigma` / `noise_sigma`      |
| C++     | `Corner::position_covariance: std::optional<Covariance>`                   | `Config::noise_sigma: std::optional<float>`       |

## 8.3 How the covariance is computed

### 8.3.1 A fit that is centred on the corner

`SaddlePoint` fits a quadratic surface to a \\((2r+1)^2\\) patch of
pixel values \\(\mathbf{p}\\) around the seed (\\(r = 2\\) by default):

\\[
I(x, y) \approx a x^2 + b x y + c y^2 + d x + e y + f .
\\]

The coefficients \\(\theta = (a, b, c, d, e)\\) are a linear
least-squares fit, \\(\theta = R\,\mathbf{p}\\), where \\(R\\) depends only on
the patch radius and is built once. The corner is the saddle of the
quadratic, at offset

\\[
u(\theta) = -H^{-1} g, \qquad
H = \begin{pmatrix} 2a & b \\\\ b & 2c \end{pmatrix}, \quad
g = \begin{pmatrix} d \\\\ e \end{pmatrix}
\\]

from the window centre. A real corner is not a quadratic. When the
corner is off the window centre, the fitted saddle lands partway back
toward the centre: on sharp corners, a single fit reports about 70% of
the true offset. That is the bias a single fit has.

This pull vanishes when the window is centred on the corner, because
the window is then point-symmetric about it. So the refiner moves its
window. At window offset \\(o\\) it resamples the patch bilinearly,
\\(\mathbf{p}(o) = W(o)\,\mathbf{I}\\) for image pixels \\(\mathbf{I}\\),
refits it, and solves for the offset where the fitted saddle sits on
the window centre:

\\[
F(o, \mathbf{I}) = u\bigl(R\,W(o)\,\mathbf{I}\bigr) = 0 .
\\]

It takes Newton steps \\(o \leftarrow o - \lambda M^{-1} F\\) with
\\(M = \partial F / \partial o\\), halving \\(\lambda\\) until the
saddle moves closer to the centre. It stops when the saddle is within
0.001 px of the centre, or after 16 steps. Each refit is a bilinear
blend of fits on four integer-centred patches, so a corner costs four
to six fits. If the iteration does not converge, the refiner returns
the single integer-grid fit and **no covariance**.

### 8.3.2 Propagating pixel noise

At the fixed point, the implicit-function theorem gives the sensitivity
of the estimate \\(\hat o\\) to every pixel:

\\[
\frac{\partial \hat o}{\partial \mathbf{I}}
= -M^{-1} \frac{\partial F}{\partial \mathbf{I}}
= -M^{-1} J\, R\, W, \qquad J = \frac{\partial u}{\partial \theta} .
\\]

With white pixel noise of variance \\(\sigma_n^2\\), the first-order
covariance of the position is

\\[
\sigma_n^2\, C, \qquad C = A\, R\, W W^\top R^\top A^\top, \quad A = M^{-1} J .
\\]

The \\(W W^\top\\) term accounts for the correlation that bilinear
resampling introduces between neighbouring samples. The fit residual is
not used for the scale: on a sharp corner it measures how far the patch
is from a quadratic, not the noise. Core callers get \\(C\\) per unit
noise variance as `RefineResult::covariance_per_noise_variance`.

### 8.3.3 The image noise level

\\(\sigma_n\\) is the standard deviation of the pixel noise in gray
levels. Unless you supply it, the detector estimates it once per frame,
and only when a refined corner carries a covariance. The image is a
chessboard, full of edges, so the estimator has to measure noise without
mistaking edges for it:

- It samples about 8 000 4×4 windows on a fixed grid. The cost does not
  grow with the image size and stays under 0.1 ms per frame.
- In each window it takes the product of third differences along x and
  along y. This operator cancels any intensity that is at most quadratic
  along either axis: flat areas, shading, and straight edges along an
  axis. For white noise of σ its output has standard deviation 20σ.
- Windows with sharp edges at other angles, or with corners, are
  rejected by a χ² gate on the nine other components of total degree
  3 or more. The gate threshold scales with the noise level, which is
  iterated to a fixed point starting from the quietest windows.
- Sensor noise grows with brightness, so the windows are split into four
  intensity bands. A band whose variance stands out is treated as
  structure and dropped; the rest are pooled.
- Windows within 12 gray levels of 0 or 255 are skipped, because
  clipping removes noise there.

On synthetic boards with white noise the estimate is within about 4%
for σ ≥ 2 gray levels and 5% for σ ≥ 1. When fewer than 512 windows
are free of edges, it gives no estimate, and the corners get no
covariance. That happens on boards that fill the frame with small,
sharp cells at low noise.

### 8.3.4 The reported covariance

\\[
\Sigma = \max\bigl(\sigma_n^2, \tfrac{1}{12}\bigr)\, C + f^2\, I_2,
\qquad f = 0.002 \text{ px} .
\\]

- **The variance floor of 1/12 gray level²** is the variance of
  rounding to 8 bits. A smoothed or compressed image can measure lower:
  one of the repository's test images reads σ ≈ 0.2. The floor keeps
  such an image from getting an overconfident covariance.
- **The position floor \\(f\\)** covers the small systematic error
  the re-centred fit keeps: interpolation, 8-bit rounding of the edge
  profile. It was chosen to keep the whitened error closest to unit
  variance on synthetic grid-aligned corners with 8–12 px cells. With
  no floor, the sharpest corners at the lowest noise read up to 1.27×
  their predicted variance. At 0.005 px they read 0.63×, which is too
  conservative. 0.002 px keeps them within about 0.85–1.2.

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

![Predicted vs measured error, and whitened errors](img/uncertainty_calibration.png)

Synthetic 200×200 boards with 8 px cells, 40/200 contrast, blur σ 0–2.5 px
and white noise σ 1–8 gray levels. The noise σ is estimated from each
image, end to end through the detector. Sheared boards at low noise have
no edge-free area, so the estimate declines there and those conditions
carry no covariance.

- **Boards aligned with the pixel grid** (blue, and the green sheared
  boards whose first axis is horizontal) sit on the diagonal.
  \\(\hat\sigma_0^2\\) is between 0.95 and 1.24 on every condition where
  the detections match the true corners. The whitened errors follow a
  unit normal (std 1.01).
- **A board rotated by 8°** (orange) does not. At low noise its corners
  are up to three times further from the truth than predicted.

### 8.4.1 Where it does not hold

- **Boards rotated a few degrees from the pixel grid.** On a sharp
  board rotated 2–15° (or 45°), the noise-free error of the re-centred
  fit is 0.01–0.02 px. On a grid-aligned board it is at most 0.004 px,
  and 0.002–0.004 px at 20–30°. The pattern is the same for 8 and 12 px
  cells. This bias is larger than the floor, so it shows up when the
  noise scatter is small. At 8° on a sharp board, \\(\hat\sigma_0^2\\) is
  about 10, 3.6, 1.6 and 1.1 at noise σ 1, 2, 4 and 8. At blur σ 1.5 it
  is 2.6, 1.4, 1.1 and 1.05, and at blur 2.5 about 1. In practice: at
  low noise on sharp images of an arbitrarily rotated board, the
  covariance can understate the error up to about 3× in σ.
- **Small, sheared, blurred cells.** With 5 px cells, a 40–60° shear and
  blur σ ≥ 1.5 px, the window also covers the neighbouring corners. The
  re-centred position can then be worse than the single integer-grid
  fit (60° shear, blur 2.5: 0.29 vs 0.19 px RMS). The covariance does
  not see this.
- **Low signal-to-noise.** When the scatter itself reaches about
  0.3 px, the first-order propagation is no longer accurate: the
  covariance understates the scatter by up to 1.5× in variance. A
  nearly flat fit occasionally gives a very large covariance. In these
  conditions some detections also land more than half a pixel from any
  true corner; no covariance describes that.
- **Dense, sharp, rotated boards at high noise.** When every window
  holds an edge comparable to the noise, the noise estimate reads high:
  up to 2× at σ 8 on 45° boards with 6 px cells. The covariance is then
  too large, which is conservative.
- **Signal-dependent noise.** Real sensors are noisier in bright areas.
  The detector uses one σ per image: the root mean noise variance over
  the image's edge-free areas. With synthetic signal-dependent noise,
  the predicted scatter matched the measured one when given the mean
  noise variance over the fit patch. On a sheared board the two axes
  then split by up to about 15% while the overall factor stayed near 1.
- **Correlated noise.** Demosaicing, JPEG compression and resampling
  correlate the noise of neighbouring pixels. The estimator then
  measures the noise as its operator sees it, usually below the
  per-pixel σ, and the propagation assumes white noise. The absolute
  scale of the covariance is approximate. Covariances of corners in the
  same image remain meaningful relative to each other. This case has
  not been measured.

All figures here come from synthetic boards: straight edges, Gaussian
blur, additive noise. `tools/render_uncertainty_overlays.py`
regenerates them and prints \\(\hat\sigma_0^2\\) per condition.

## 8.5 Using the covariance

### 8.5.1 Weighting a least-squares fit

To weight corners, whiten each residual \\(r_i\\) (observed corner minus
your model's prediction) with the Cholesky factor of its covariance,
\\(\Sigma_i = L_i L_i^\top\\), and minimise the sum of squares of the
whitened residuals \\(L_i^{-1} r_i\\). This equals
\\(\sum_i r_i^\top \Sigma_i^{-1} r_i\\): corners with small
uncertainty pull harder, and the anisotropy of each ellipse is
respected.

Rust:

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig, PositionCovariance};

/// `L⁻¹·r` for `Σ = L·Lᵀ`: its squared norm is `rᵀ·Σ⁻¹·r`.
fn whiten(r: [f32; 2], c: &PositionCovariance) -> [f32; 2] {
    let l11 = c.xx.sqrt();
    let l21 = c.xy / l11;
    let l22 = (c.yy - l21 * l21).sqrt();
    let w0 = r[0] / l11;
    [w0, (r[1] - l21 * w0) / l22]
}

let cfg = DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::saddle_point());
let mut detector = Detector::new(cfg)?;
for c in detector.detect_u8(&pixels, width, height)? {
    let Some(cov) = c.position_covariance else { continue }; // see §8.5.3
    let [px, py] = project(&c); // your model's prediction for this corner
    let w = whiten([c.x - px, c.y - py], &cov);
    // add w[0] and w[1] as residuals of the least-squares problem
}
```

Python:

```python
import numpy as np
import chess_corners

cfg = chess_corners.DetectorConfig.chess()
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
\\(p\\) model parameters is the variance factor of §8.4. A value near 1
says the covariances and the model agree. A value well above 1 means
model error (lens distortion not modelled, a bent board) or one of the
regimes in §8.4.1. To propagate the corner uncertainty into the
parameters, scale the parameter covariance by that factor.

The covariance describes the position in the image the detector saw.
If you undistort or otherwise transform corner positions before
fitting, transform the covariance with the Jacobian \\(G\\) of that
mapping: \\(G \Sigma G^\top\\).

### 8.5.2 Supplying the noise level

When the sensor noise is known, for example from two frames of a
uniform target, \\(\sigma_n = \operatorname{std}(I_1 - I_2)/\sqrt{2}\\),
supply it instead of estimating it per frame. Give the noise as it
appears in the 8-bit image the detector receives, after any gain or
processing. Rounding adds 1/12 gray level², which matters only below
about 1 gray level.

```rust
let cfg = DetectorConfig::chess()
    .with_chess(|c| c.refiner = ChessRefiner::saddle_point())
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
config.noise_sigma = 1.8f; // std::nullopt estimates it per image
// C: cfg.has_noise_sigma = 1; cfg.noise_sigma = 1.8f;
```

A negative or non-finite value is rejected when the detector is built
(`ChessError::InvalidNoiseSigma`).

### 8.5.3 When there is no covariance

`None` in Rust, a `NaN` row in Python or JS, and
`has_position_covariance == 0` in C all mean the same thing. They have
these causes:

| Cause | What to do |
|-------|------------|
| The refiner is not `SaddlePoint`, or the detector is Radon | Select `ChessRefiner::saddle_point()` if you need covariances. |
| Re-centring did not converge for this corner | The position is the single integer-grid fit, which is biased. Leave the corner out, or give it a weight well below its neighbours'. |
| The noise estimate declined (too few edge-free areas) | All corners of the frame lack a covariance. Supply `noise_sigma`. |
| `upscale` is active | Interpolation correlates the noise of the upscaled pixels, so no covariance is reported. |

## 8.6 Orientation uncertainty

Each axis in `axes` carries a `sigma`, a 1σ angular uncertainty in
radians. It is not a calibrated covariance:

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
