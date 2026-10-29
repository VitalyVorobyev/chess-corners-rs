# Part V: Subpixel localization

A detector's response peaks are not yet corner positions. ChESS finds
its peaks on integer pixels; Radon finds them on a 2×-supersampled
response map. Calibration, pose estimation and homography fitting need
positions to a small fraction of a pixel, so every corner goes through
a subpixel localization step.

The library splits that step in two layers:

1. **Each detector's own localizer**, which works on the detector's
   response map and is always there: the ChESS **center of mass**
   (§5.3) and the Radon **peak fit** (§5.4).
2. An optional **image refiner**, which works on the input image
   around each corner and does not care which detector found it:
   **SaddlePoint** (§5.5). It is set once, on `DetectorConfig::refiner`,
   and applies to whichever detector is active.

## 5.1 Why two layers

A response map is a smoothed picture of "corner-ness". Its maximum
sits near the corner, but how near depends on the response shape, not
on the corner: blur, board rotation and the corner's offset from the
pixel grid all bend the response, and a localizer that only looks at
the response inherits that bias. The image itself still holds the
exact geometry — two edges crossing — so a fit on the pixels can
remove it.

![Refined positions of one corner over 600 noise draws, for three localizers, with their predicted ellipses](img/uncertainty_bias_precision.png)

The figure repeats one synthetic corner under 600 draws of white
noise ([Part VIII](part-08-uncertainty.md) explains it in full). The
two response-map localizers — the center of mass and the Radon peak
fit — land beside the true corner (`+`) or scatter widely; the
saddle-point fit on the image lands on it. The response-map localizers
are cheap and need nothing but the map the detector already computed,
so they are the default; the image refiner is what to add when
accuracy matters more than a fraction of a microsecond per corner.

| | Response-map localizer (strategy config) | Image refiner (`DetectorConfig::refiner`) |
|---|---|---|
| ChESS | center of mass (`ChessConfig::center_of_mass`) | `None` (default) · `SaddlePoint(SaddlePointConfig)` |
| Radon | peak fit (`RadonConfig::peak_fit`) | `None` (default) · `SaddlePoint(SaddlePointConfig)` |

Every combination is valid: an image refiner reads only the image, so
there is no refiner that needs a response map the active detector does
not have. Where the image refiner starts, and what happens to a corner
it cannot refine, follows the detector:

- **ChESS** — the image refiner *replaces* the center of mass. It
  starts from the integer response peak, and a corner it rejects is
  dropped — unless `ChessConfig::on_refiner_reject` is
  `OnRefinerReject::KeepCenterOfMass`, which keeps that corner at its
  center-of-mass position instead. The corners the image refiner accepts
  are the same either way. On cluttered or blurred boards the refiner
  rejects real corners as well as clutter, so keeping them preserves the
  center of mass's recall while every corner it can fit gets the image
  refiner's accuracy.
- **Radon** — the image refiner runs *on top of* the peak fit. It
  starts from the peak-fit position, and a corner it does not refine to
  convergence keeps its peak-fit position; no corner is dropped.

Under `Uncertainty::Propagated` a corner's position covariance comes from whichever localizer produced
its final position ([Part VIII](part-08-uncertainty.md)).

```rust
use chess_corners::{DetectorConfig, ImageRefiner};

// ChESS with its center of mass (the default).
let cfg = DetectorConfig::chess();
// ChESS with saddle-point refinement in place of the center of mass.
let cfg = DetectorConfig::chess().with_refiner(ImageRefiner::saddle_point());
// Radon peak fit, refined by the saddle point where it converges.
let cfg = DetectorConfig::radon().with_refiner(ImageRefiner::saddle_point());
// Tune the ChESS center of mass itself.
let cfg = DetectorConfig::chess().with_chess(|c| c.center_of_mass.radius = 3);
```

The image refiner survives a strategy switch (`with_chess` /
`with_radon` leave it alone), so the same refinement can be compared
across detectors with a one-line change. In JSON it is the top-level
`"refiner"` key: `null` (or absent) or `{"saddle_point": {...}}`.

## 5.2 The refiner trait

Below the facade, every localizer that refines one seed at a time is a
`CornerRefiner` in `chess-corners-core`:

```rust
pub trait CornerRefiner {
    /// Half-width of the patch the refiner reads around the seed.
    fn radius(&self) -> i32;

    /// Refine one seed. `ctx` exposes the image and/or response map.
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}
```

A `RefineResult` carries the refined `(x, y)`, a refiner-specific
`score`, and a `RefineStatus`:

- `Accepted` — refined position is inside `radius`-sized support and
  below the refiner's rejection thresholds.
- `OutOfBounds` — the patch would read past the image border.
- `IllConditioned` — the refiner's local system was singular or too
  eccentric (edge rather than corner), or the center-of-mass window
  holds no positive response.
- `Rejected` — the refined offset exceeded the refiner's
  `max_offset`, or a per-refiner score threshold fired.

Under `Uncertainty::Propagated` (the default is `Off`), an accepted result also
carries `noise_covariance_per_unit_variance()`: the image noise propagated to
the refined position through the refiner's exact Jacobian, per unit
noise variance (px² per gray-level²). The center of mass always sets it,
the saddle-point refiner when its re-centring converges. The detector
scales it by the image's noise level and adds the localizer's
systematic term to give each corner's `position_covariance`
([Part VIII](part-08-uncertainty.md)).

The trait is sealed; the built-in implementations are
`CenterOfMassRefiner`, `SaddlePointRefiner` and the `Refiner` enum that
dispatches over them. Core's `RefinerKind` selects an image refiner
(`RefinerKind::SaddlePoint`); the center of mass is configured on
`ChessParams::center_of_mass`. The pipeline holds one `Refiner` with
its scratch buffers and reuses it across seeds — no per-corner
allocation.

## 5.3 Center of mass (ChESS)

The ChESS localizer. Operates on the ChESS response map and computes
the response-weighted centroid of a `(2r + 1)²` window around the
integer peak:

\\[
x_r = \frac{\sum_{p \in W} x_p \cdot R_p}{\sum_p R_p}, \qquad
y_r = \frac{\sum_{p \in W} y_p \cdot R_p}{\sum_p R_p}
\\]

`R_p = max(R(x, y), 0)` clips negative responses so one strong
negative pixel can't push the centroid outside the window. A window with
no positive response has no centroid and returns `IllConditioned`; the
detector's seeds are positive response peaks, so it never hits this.

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Input            | ChESS response map                              |
| Config           | `ChessConfig::center_of_mass` (`radius`)        |
| Default `radius` | 2 (5×5 window)                                  |
| Typical cost     | ~20 ns per corner; ~0.5 µs with the covariance |
| Strengths        | Cheapest option in the benchmark; closed-form   |
| Weaknesses       | Centroid bias when the response is asymmetric;  |
|                  | fails when `radius` crosses neighboring corners |

Keep it when throughput matters more than sub-0.1 px accuracy. When an
image refiner is set it does not run.

## 5.4 Peak fit (Radon)

The Radon localizer is part of peak detection itself: a closed-form
3-point fit through the response maximum and its neighbours along each
axis, on the supersampled response map (`RadonConfig::peak_fit`,
Gaussian by default, parabolic as the alternative).
[Part IV §4.4](part-04-radon-detector.md#44-peak-fit-pipeline)
describes it. Because Radon peaks arrive already subpixel, an image
refiner on Radon starts from the peak fit and keeps it as a fallback.

## 5.5 SaddlePoint (image refiner)

Fits a 2D quadratic `f(x, y) = a·x² + b·x·y + c·y² + d·x + e·y + g`
to a `(2r + 1)²` image patch and returns the saddle point (the
critical point where `∇f = 0`). The six coefficients come from a
linear least-squares fit. A quadratic fitted off-centre puts the
saddle partway back toward the window centre, so the refiner moves the
window — Newton steps, refitting on the patch resampled bilinearly at
each position — until the fitted saddle is within 0.001 px of the window
centre; re-centred, the window is point-symmetric about the corner and
that pull vanishes. Where this does not converge it returns the single
fit without a covariance; on ChESS that single fit is the corner's
position, on Radon the corner keeps its peak-fit position instead. [Part VIII §8.3.1](part-08-uncertainty.md#831-noise-through-the-localizer)
derives the covariance at the converged position.

The determinant of the Hessian is `4·a·c − b²`. A true X‑junction is a
saddle, so the Hessian should be indefinite (`det < 0`). The refiner
rejects:

- `|det| < min_abs_det` — flat patch.
- `det > -det_margin` — the quadratic is a bowl or ridge, not a saddle.
- `‖offset‖ > max_offset` — refined point outside the patch.

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Input            | Image                                           |
| Config           | `ImageRefiner::SaddlePoint(SaddlePointConfig)`  |
| Default `radius` | 2 (5×5 patch)                                   |
| Typical cost     | ~0.2 µs per corner; ~0.6 µs with the covariance |
| Strengths        | Lowest error of the geometric refiners, sharp or blurred, synthetic or real |
| Weaknesses       | Biased by illumination gradients and asymmetric lens blur (0.02–0.10 px); on ~5 px cells with strong shear and blur, the window holds neighbouring corners and the re-centred position can be worse than a single fit |

The default choice for calibration-grade work: in Part IX's synthetic
sweeps it has the lowest mean error of the built-in refiners on every
clean, blurred and noisy row up to `σ_noise = 8`, and is within 0.003 px
of (or below) `cv2.cornerSubPix` on the blurred rows.

## 5.6 Picking a localizer and refiner

The measurement-driven comparison lives in Part IX. In short:

- `SaddlePoint` has the lowest mean error of the built-in refiners on
  clean and blurred synthetic frames and on real calibration images, and
  is the default choice for calibration-grade work on either detector
  ([Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon)
  compares it with the Radon peak fit).
- With no image refiner, ChESS's center of mass and Radon's peak fit
  cost almost nothing beyond detection.

Switching is a single-line change —
`cfg.with_refiner(ImageRefiner::saddle_point())` or
`cfg.without_refiner()` — and the comparison numbers in Part IX come
from running every localizer on the same fixture at a single build.

---

Next, [Part VI](part-06-orientation-methods.md) covers the orientation
methods that produce the `axes` descriptor field,
shared by both the ChESS and Radon pipelines.
