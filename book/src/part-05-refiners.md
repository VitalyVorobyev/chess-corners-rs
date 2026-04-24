# Part V: Subpixel refiners

A corner detector (ChESS or Radon) returns integer pixel positions
plus a response value. Most downstream consumers — camera calibration,
pose estimation, homography fitting — need positions to better than a
pixel. A **refiner** takes one seed `(x₀, y₀)` plus a local view into
either the image or the response map and returns a refined
`(x, y)` in subpixel coordinates.

The library ships five refiners with one trait and one configuration
enum, so swapping between them is a one-line change. The ChESS
detector always runs one of these refiners to reach subpixel accuracy.
The Radon detector has its own peak fit built in (see
[Part IV §4.4](part-04-radon-detector.md#44-peak-fit-pipeline)), but
its output can still be post-refined if you need a different
statistical behavior.

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
  eccentric (edge rather than corner).
- `Rejected` — the refined offset exceeded the refiner's
  `max_offset`, or a per-refiner score threshold fired.

The public user-facing selector is `RefinementMethod`. At runtime the
facade stores an owned `Refiner` enum (one allocated scratch buffer
per concrete refiner) so the same instance is reused across seeds —
no per-corner allocation.

## 5.2 CenterOfMass

Operates on the ChESS response map. Computes the response-weighted
centroid of a `(2r + 1)²` window around the seed:

```text
x_r = Σ_{p in W}  (x_p · R_p)  /  Σ_p R_p
y_r = Σ_{p in W}  (y_p · R_p)  /  Σ_p R_p
```

`R_p = max(R(x, y), 0)` clips negative responses so one strong
negative pixel can't push the centroid outside the window.

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Input            | Response map                                    |
| Default `radius` | 2 (5×5 window)                                  |
| Typical cost     | ~20 ns per corner                               |
| Strengths        | Fastest option; closed-form; always converges   |
| Weaknesses       | Centroid bias when the response is asymmetric;  |
|                  | fails when `radius` crosses neighboring corners |

Use when throughput matters more than sub-0.1 px accuracy, or when
the ChESS response is the only signal you want the refinement to see.

## 5.3 Förstner

Gradient-based. Solves a weighted least-squares system for the point
closest (in a Mahalanobis sense) to all image-gradient lines in a
local window. The structure tensor

```text
M = Σ_p  w_p · [ gx²  gx·gy ]
               [ gx·gy  gy²  ]
```

is assembled with `3×3` central-difference gradients and radial
weights `w_p = 1 / (1 + 0.5·‖p − seed‖²)`. The refined position is

```text
u = M⁻¹ · Σ_p  w_p · (p − seed) · [gx·gy]ᵀ
```

Rejections:

- `trace(M) < min_trace` — low-gradient region.
- `det(M) < min_det` — singular structure tensor.
- `λ_max / λ_min > max_condition_number` — one dominant direction
  (an edge, not a corner).
- `‖u‖ > max_offset` — extrapolating beyond a trusted neighborhood.

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Input            | Image                                           |
| Default `radius` | 2 (5×5 gradient window + 1 px halo)             |
| Typical cost     | ~60 ns per corner                               |
| Strengths        | Principled on sharp, high-SNR images            |
| Weaknesses       | Relies on sharp gradients — blur flattens `M`   |

Good pick for clean calibration frames where image edges are sharp.
Gaussian blur at σ ≳ 1 px flattens the gradient magnitudes and this
refiner degrades roughly linearly with σ.

Reference: Förstner & Gülch, 1987, *"A fast operator for detection
and precise location of distinct points, corners and centres of
circular features."*

## 5.4 SaddlePoint

Fits a 2D quadratic `f(x, y) = a·x² + b·x·y + c·y² + d·x + e·y + g`
to a `(2r + 1)²` image patch and returns the saddle point (the
critical point where `∇f = 0`). The six coefficients come from a
6×6 normal-equation solve with partial pivoting.

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
| Typical cost     | ~120 ns per corner                              |
| Strengths        | Blur-robust; no gradient required               |
| Weaknesses       | Parabolic model is approximate on sharp edges   |

A reasonable default when you don't know in advance whether frames
will be sharp or blurred. Stable across the full blur sweep in Part VII.

## 5.5 RadonPeak

Per-candidate version of the Radon detector's peak fit. Computes the
local Radon response on a `(2·patch_radius + 1)²` grid around the
seed (at working resolution set by `image_upsample`), applies the
same 3×3 box blur and 3-point Gaussian peak fit used by the full
detector, and returns the refined offset. See
[Part IV §4.4](part-04-radon-detector.md#44-peak-fit-pipeline) for
the underlying formulas — this refiner shares all of them.

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Input            | Image                                           |
| Default settings | `ray_radius = 2`, `patch_radius = 3`, `image_upsample = 2` |
| Typical cost     | ~17 µs per corner                               |
| Strengths        | Lowest error on clean and blurred imagery       |
| Weaknesses       | 100–1000× slower than the structure-tensor refiners |

The accuracy ceiling of the library on clean and blurred data. Choose
it when a calibration budget accommodates an extra few ms per frame
and accuracy is the priority.

If you are already running the Radon *detector* (Part IV), its built-in
peak fit gives you the same refinement implicitly, and this refiner
is redundant.

## 5.6 ML (ONNX model)

A learned refiner. Feeds a 21×21 normalized grayscale patch into a
small CNN and takes `[dx, dy, conf_logit]` back out. The ChESS path
extracts the patch, stages a batch, runs ONNX inference via
`tract-onnx`, and adds `[dx, dy]` to each seed.

Available behind the `ml-refiner` feature. The default model
`chess_refiner_v4.onnx` is embedded in the
[`chess-corners-ml`](https://docs.rs/chess-corners-ml) crate at
`crates/chess-corners-ml/assets/ml/`.

### 5.6.1 Architecture

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

### 5.6.2 Training data and loss

The training pipeline lives in `tools/ml_refiner/`. The v4 dataset
(`configs/synth_v6.yaml`) renders 200 000 patches with a 50/50 mix of
two rendering modes:

- **Hard cells.** An anti-aliased periodic checkerboard, rendered at
  a random cell size in `[4, 12]` px, then blurred by a Gaussian
  PSF in `σ ∈ [0.3, 2.0]` px. This matches real camera output: ink/paper
  step edges, softened by the optical system's PSF, sampled by the
  sensor. The benchmark fixture in Part VII uses the same renderer.
- **Smooth saddle.** The `tanh(x)·tanh(y)` model from earlier
  dataset revisions. Included at 50 % weight so callers who still
  feed the model smooth synthetic patches — as older documentation
  suggested — keep working.

Augmentations per sample: additive Gaussian noise `σ ∈ [0, 10]` gray
levels, photometric jitter (contrast, brightness, gamma), optional
projective warp for perspective robustness, and 20 % negative
samples (flat, edge, stripe, blob, pure noise, near-corner) with
`is_pos = 0`.

The true subpixel offset is sampled from `[-0.6, 0.6]` px.

Loss: Huber on `(dx, dy)` for positives, binary cross-entropy on
confidence for all samples. The regression loss is weighted up on
positives only via `is_pos` (negatives have no valid target).

### 5.6.3 Why earlier versions failed

Historical context, for anyone wondering why v4 is the version that
ships. Versions v1–v2 trained exclusively on smooth `tanh(x)·tanh(y)`
corner patches. That patch model does not resemble what a camera
produces — real sensor images have hard step edges (ink/paper)
softened by a small optical PSF, whereas `tanh` has smooth transitions
the width of the entire patch. Models trained on `tanh` only generalized
poorly to the benchmark fixture (mean error ~0.5 px on hard cells).

v3 swapped to hard cells only and hit ~0.6 px on `tanh` inputs — the
opposite distribution failure.

v4 is the mixed dataset (50/50) plus retuned offsets and augmentations.
It is robust across both distributions and best-in-class on noise-heavy
scenes (`σ ≥ 8` gray levels) per the measurements in Part VII. It does
not beat `RadonPeak` on clean or mildly blurred data; the gap appears
to be a training-signal limit (neither a wider CNN nor a softargmax
head closed it in our sweeps), not a capacity ceiling. Closing the
gap would likely require distillation from `RadonPeak` or orders of
magnitude more training data.

### 5.6.4 ONNX export and inference

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

## 5.7 Picking a refiner

The measurement-driven comparison lives in Part VII. In short:

- Budget matters more than anything else: the structure-tensor
  refiners are 100–1000× faster than RadonPeak and ML.
- RadonPeak gives the lowest error on clean and blurred frames at
  calibration rates.
- ML wins on noise-heavy scenes (`σ ≥ 8`).
- SaddlePoint is a blur- and condition-robust default when you don't
  know the scene in advance.

The refiner is selected via `ChessConfig.refiner.kind`, which is a
simple enum — switching between them is a single-line change, and
the comparison numbers in Part VII come from running all five on the
same fixture at a single build.

---

Next, [Part VI](part-06-multiscale-and-pyramids.md) describes how the
pyramid and coarse-to-fine pipeline are built around these refiners
for wide-resolution input and small boards.
