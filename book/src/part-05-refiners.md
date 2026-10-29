# Part V: Subpixel refiners

A corner detector (ChESS or Radon) returns integer pixel positions
plus a response value. Most downstream consumers — camera calibration,
pose estimation, homography fitting — need positions to better than a
pixel. A **refiner** takes one seed `(x₀, y₀)` plus a local view into
either the image or the response map and returns a refined
`(x, y)` in subpixel coordinates.

The library ships four refiners with one trait and one configuration
enum, so swapping between them is a one-line change. The ChESS
detector always runs one of these refiners to reach subpixel accuracy.
The Radon detector already locates its peaks to a fraction of a pixel
with a 3-point peak fit (see
[Part IV §4.4](part-04-radon-detector.md#44-peak-fit-pipeline)), and can
refine them with `SaddlePoint` (§5.3).

## 5.1 The refiner trait

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
  eccentric (edge rather than corner), or the CenterOfMass window holds
  no positive response.
- `Rejected` — the refined offset exceeded the refiner's
  `max_offset`, or a per-refiner score threshold fired.

Under `Uncertainty::Propagated` (the default is `Off`), an accepted result also
carries `noise_covariance_per_unit_variance()`: the image noise propagated to
the refined position through the refiner's exact Jacobian, per unit
noise variance (px² per gray-level²). Every refiner except ML sets it;
the saddle-point refiner only when its re-centring converges. The
detector scales it by the image's noise level and adds the refiner's
systematic term to give each corner's `position_covariance`
([Part VIII](part-08-uncertainty.md)).

The user-facing selector for ChESS is `ChessRefiner` (carrying
`CenterOfMass`, `SaddlePoint`, and optionally `Ml`),
nested inside `ChessConfig`. Each enum variant carries its tuning
struct as a payload, so a refiner kind change cannot leave a stale
per-refiner config field behind. At runtime the facade stores an owned
`Refiner` enum (one allocated scratch buffer per concrete refiner) so
the same instance is reused across seeds — no per-corner allocation.

The Radon selector is `RadonRefiner`, nested inside `RadonConfig`:
`PeakFit` (the default), which keeps the 3-point peak fit
(`RadonConfig.peak_fit`) as the final position, or
`SaddlePoint(SaddlePointConfig)`.
`CenterOfMass` reads the ChESS response map and cannot refine Radon
peaks, and ML is not offered. A Radon corner whose saddle-point
refinement does not converge keeps its peak-fit position.

## 5.2 CenterOfMass

Operates on the ChESS response map. Computes the response-weighted
centroid of a `(2r + 1)²` window around the seed:

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
| Input            | Response map                                    |
| Default `radius` | 2 (5×5 window)                                  |
| Typical cost     | ~20 ns per corner; ~0.5 µs with the covariance |
| Strengths        | Cheapest option in the benchmark; closed-form   |
| Weaknesses       | Centroid bias when the response is asymmetric;  |
|                  | fails when `radius` crosses neighboring corners |

Use when throughput matters more than sub-0.1 px accuracy, or when
the ChESS response is the only signal you want the refinement to see.

## 5.3 SaddlePoint

Fits a 2D quadratic `f(x, y) = a·x² + b·x·y + c·y² + d·x + e·y + g`
to a `(2r + 1)²` image patch and returns the saddle point (the
critical point where `∇f = 0`). The six coefficients come from a
linear least-squares fit. A quadratic fitted off-centre puts the
saddle partway back toward the window centre, so the refiner moves the
window — Newton steps, refitting on the patch resampled bilinearly at
each position — until the fitted saddle is within 0.001 px of the window
centre; re-centred, the window is point-symmetric about the corner and
that pull vanishes. Where this does not converge it returns the single
fit without a covariance (a Radon corner keeps its peak-fit position
instead). [Part VIII §8.3.1](part-08-uncertainty.md#831-noise-through-the-localizer)
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
| Default `radius` | 2 (5×5 patch)                                   |
| Typical cost     | ~0.2 µs per corner; ~0.6 µs with the covariance |
| Strengths        | Lowest error of the geometric refiners, sharp or blurred, synthetic or real |
| Weaknesses       | Biased by illumination gradients and asymmetric lens blur (0.02–0.10 px); on ~5 px cells with strong shear and blur, the window holds neighbouring corners and the re-centred position can be worse than a single fit |

The default choice for calibration-grade work: in Part IX's synthetic
sweeps it has the lowest mean error of the built-in refiners on every
clean, blurred and noisy row up to `σ_noise = 8`, and is within 0.003 px
of (or below) `cv2.cornerSubPix` on the blurred rows.

## 5.4 ML (ONNX model)

A learned refiner. Feeds a 21×21 normalized grayscale patch into a
small CNN and takes `[dx, dy, conf_logit]` back out. The ChESS path
extracts the patch, stages a batch, runs ONNX inference via
`tract-onnx`, and adds `[dx, dy]` to each seed.

Available behind the `ml-refiner` feature. The default model
`chess_refiner_v4.onnx` is embedded in the
[`chess-corners-ml`](https://docs.rs/chess-corners-ml) crate at
`crates/chess-corners-ml/assets/ml/`.

### 5.4.1 Architecture

The shipped model is `CornerRefinerNet`, a CoordConv CNN with a
flatten + MLP head. About 180 K parameters:

| Layer                 | Shape                | Notes                                                    |
|-----------------------|----------------------|----------------------------------------------------------|
| Input                 | 1 × 21 × 21          | Grayscale patch, normalized `u8/255`.                    |
| CoordConv prepend     | 3 × 21 × 21          | Two extra channels with per-pixel `x`, `y` in `[-1, 1]`. |
| Conv3×3, ReLU         | 16 × 21 × 21         |                                                          |
| Conv3×3, ReLU         | 16 × 21 × 21         |                                                          |
| Conv3×3 stride 2, ReLU| 32 × 11 × 11         |                                                          |
| Conv3×3, ReLU         | 32 × 11 × 11         |                                                          |
| Conv3×3 stride 2, ReLU| 64 × 6 × 6           |                                                          |
| Flatten               | 2304                 |                                                          |
| Linear, ReLU          | 64                   |                                                          |
| Linear (no activation)| 3                    | `[dx, dy, conf_logit]`.                                  |

CoordConv (Liu et al., 2018) concatenates explicit `x`, `y` coordinate
channels to the input. Standard convolutions are translation-equivariant
and cannot reliably regress to an absolute pixel offset from a patch
center; CoordConv restores the center reference that pure convolutions
discard.

The head outputs three scalars: an offset `(dx, dy)` in patch-pixel
units (valid range about `[-0.6, 0.6]` px, matching the training
distribution) and a confidence logit. The current Rust consumer
applies the offset and ignores the confidence.

The PyTorch source in `tools/ml_refiner/model.py` also defines a
wider variant (`CornerRefinerNetLarge`, ~730 K params, GroupNorm
between convs) and a spatial-softargmax head
(`CornerRefinerNetSoftArgmax`). Both match the 1-channel in, 3-scalar
out contract so they are drop-in replacements for the inference
path. The shipped model is the small variant — the larger and
softargmax variants did not move the held-out error meaningfully in
our sweeps.

### 5.4.2 Training data and loss

The training pipeline lives in `tools/ml_refiner/`; the shipped model's
synthetic dataset is generated by the config at
`tools/ml_refiner/configs/synth_v6.yaml`. It renders 200 000 patches
with a 50/50 mix of two rendering modes:

- **Hard cells.** An anti-aliased periodic checkerboard, rendered at
  a random cell size in `[4, 12]` px, then blurred by a Gaussian
  PSF in `σ ∈ [0.3, 2.0]` px. This matches real camera output: ink/paper
  step edges, softened by the optical system's PSF, sampled by the
  sensor. The benchmark fixture in Part IX uses the same renderer.
- **Smooth saddle.** A `tanh(x)·tanh(y)` corner model, included at
  50 % weight so the model stays accurate on the smooth synthetic
  patches some callers still feed it.

Augmentations per sample: additive Gaussian noise `σ ∈ [0, 10]` gray
levels, photometric jitter (contrast, brightness, gamma), optional
projective warp for perspective robustness, and 20 % negative
samples (flat, edge, stripe, blob, pure noise, near-corner) with
`is_pos = 0`.

The true subpixel offset is sampled from `[-0.6, 0.6]` px.

Loss: Huber on `(dx, dy)` for positives, binary cross-entropy on
confidence for all samples. The regression loss is weighted up on
positives only via `is_pos` (negatives have no valid target).

### 5.4.3 Designing the training distribution

A subpixel ML refiner is only as good as the patches it trains on, and
synthetic training data has two failure modes that pull in opposite
directions:

- **Smooth-only data.** Training exclusively on smooth
  `tanh(x)·tanh(y)` corners teaches the model transitions that span the
  whole patch. Real sensor images look nothing like that — hard
  ink/paper step edges softened by a small optical PSF — so a model
  that has only seen `tanh` corners localizes real ones poorly (mean
  error ~0.5 px on the hard-cell benchmark fixture).
- **Hard-only data.** Swapping to hard cells alone flips the problem:
  the model then mislocalizes the smooth synthetic corners some callers
  still feed it (~0.6 px on `tanh` inputs). Neither distribution alone
  spans the range of edge profiles a refiner meets in practice.

The shipped dataset avoids both by mixing the two modes 50/50 (§5.4.2)
and augmenting with sensor noise. Two lessons carry over into the
Part IX benchmark. Noise augmentation is what lets the model take the
heaviest-noise row (`σ = 12` gray levels), where it posts the lowest
mean error; and capacity is not the bottleneck — neither a wider CNN
nor a spatial-softargmax head lowered held-out error. On clean and
mildly blurred data the geometric refiners still win (`SaddlePoint`
on both), so ML earns its cost only
when noise dominates.

### 5.4.4 ONNX export and inference

The export step (`tools/ml_refiner/export_onnx.py`) writes an ONNX
graph at opset 17 (falling back to 18 if a conversion is unsupported).
The graph contract is:

- **Input** `patches`: `float32 [N, 1, 21, 21]`, `u8 / 255` in `[0, 1]`.
- **Output** `pred`: `float32 [N, 3]` with `[dx, dy, conf_logit]`.

Rust inference is `chess_corners_ml::MlModel::infer_batch`. It wraps
`tract-onnx`, sizes the input to the runtime batch, and returns
`Vec<[f32; 3]>`. Dynamic batch sizing is supported via a tract
`SymbolScope`, so a single loaded model can handle variable-batch
calls without re-optimization.

## 5.5 Picking a refiner

The measurement-driven comparison lives in Part IX. In short:

- Budget matters more than anything else: the geometric refiners
  are orders of magnitude faster than ML per corner.
- `SaddlePoint` has the lowest mean error of the built-in refiners on
  clean and blurred synthetic frames and on real calibration images, and
  is the default choice for image-patch refinement.
- ML has the lowest mean error on the heaviest synthetic noise row.

The ChESS refiner is selected through `ChessConfig.refiner`:
`DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::saddle_point())`.
The Radon refiner through `RadonConfig.refiner`:
`DetectorConfig::radon().with_radon(|r| r.refiner = RadonRefiner::saddle_point())`
refines the peaks
([Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon)).
Switching is a single-line change, and the comparison numbers
in Part IX come from running all three ChESS refiners on the same
fixture at a single build.

---

Next, [Part VI](part-06-orientation-methods.md) covers the orientation
methods that produce the `axes` descriptor field,
shared by both the ChESS and Radon pipelines.
