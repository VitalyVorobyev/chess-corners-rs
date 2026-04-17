# Part III: Core ChESS Internals

In this part we leave the ergonomic `chess-corners` facade and look at
the lower‑level `chess-corners-core` crate. This crate is responsible
for:

- defining the ChESS rings and sampling geometry,
- computing dense response maps on 8‑bit grayscale images,
- turning responses into corner candidates with NMS and refinement,
- converting raw candidates into rich corner descriptors.

The public API is intentionally small and stable; feature flags (`std`,
`rayon`, `simd`, `tracing`) only affect performance and observability,
not the numerical results.

---

## 3.1 Rings and sampling geometry

The ChESS response is built around a fixed **16‑sample ring** at a
given radius. The core crate encodes these rings in
`crates/chess-corners-core/src/ring.rs`.

### 3.1.1 Canonical rings

The main types are:

- `RingOffsets` – an enum representing the supported ring radii
  (`R5` and `R10`).
- `RING5` / `RING10` – the actual offset tables for radius 5 and 10.
- `ring_offsets(radius: u32)` – helper returning the offset table for
  a given radius (anything other than 10 maps to 5).

The 16 offsets are ordered clockwise starting at the top, and are
derived from the FAST‑16 pattern:

- `RING5` is the canonical `r = 5` ring used in the original ChESS
  paper.
- `RING10` is a scaled variant (`r = 10`) with the same angles, which
  improves robustness under heavier blur at the cost of a larger
  footprint and border margin.

The exact offsets are stored as integer `(dx, dy)` pairs, so sampling
around a pixel `(x, y)` means accessing `(x + dx, y + dy)` for each
ring point.

### 3.1.2 From parameters to rings

`ChessParams` in `lib.rs` controls which ring to use:

- `use_radius10` – when `true`, `ring_radius()` returns 10 instead of
  5.
- `descriptor_use_radius10` – optional override specifically for the
  descriptor ring; when `None`, it follows `use_radius10`.

Convenience methods:

- `ring_radius()` / `descriptor_ring_radius()` return the numeric
  radii.
- `ring()` / `descriptor_ring()` return `RingOffsets` values, which
  can be turned into offset tables via `offsets()`.

The response path uses `ring()`, while descriptor estimation uses
`descriptor_ring()`. This allows you, for example, to detect corners
with a smaller ring but compute descriptors on a larger one.

---

## 3.2 Dense response computation

The main entry point in `response.rs` is:

```rust
pub fn chess_response_u8(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap
```

This function computes the ChESS response at each pixel center whose
full ring fits entirely inside the image. Pixels that cannot support a
full ring (near the border) get response zero.

### 3.2.1 ChESS formula

For each pixel center `c`, we gather 16 ring samples `s[0..16)` using
the offsets described in §3.1, and a small 5‑pixel cross at the
center:

- center `c`,
- north/south/east/west neighbors.

From these values we compute:

- `SR` – a “square” term that compares opposite quadrants on the ring:

  ```text
  SR = sum_{k=0..3} | (s[k] + s[k+8]) - (s[k+4] + s[k+12]) |
  ```

- `DR` – a “difference” term encouraging edge‑like structure:

  ```text
  DR = sum_{k=0..7} | s[k] - s[k+8] |
  ```

- `μₙ` – the mean of all 16 ring samples.
- `μₗ` – the local mean of the 5‑pixel cross.

The final ChESS response is:

```text
R = SR - DR - 16 * |μₙ - μₗ|
```

Intuitively:

- `SR` is large when opposite quadrants have contrasting intensities
  (as in an X‑junction).
- `DR` is large for simple edges, and subtracting it de‑emphasizes
  edge‑like structures.
- `|μₙ - μₗ|` penalizes isolated blobs or local illumination changes.

High positive values of `R` correspond to chessboard‑like corners.

### 3.2.2 Implementation paths and borders

`chess_response_u8` is implemented in a few interchangeable ways:

- Scalar sequential path (`compute_response_sequential` /
  `compute_row_range_scalar`) – a straightforward nested loop over
  rows and columns.
- Parallel path (`compute_response_parallel`) – when the `rayon`
  feature is enabled, the outer loop is split across threads using
  `par_chunks_mut` over rows.
- SIMD path (`compute_row_range_simd`) – when the `simd` feature is
  enabled, the inner loop vectorizes over `LANES` pixels at a time,
  using portable SIMD to gather ring samples and accumulate `SR`,
  `DR`, and `μₙ` in vector registers.

Regardless of the path, the function:

- respects a border margin equal to the ring radius so that all ring
  accesses are in bounds,
- writes responses into a `ResponseMap { w, h, data }` in row‑major
  order,
- guarantees that the scalar, parallel, and SIMD variants produce the
  same numerical result up to small floating‑point differences.

### 3.2.3 ROI support with `Roi`

For multiscale refinement, we rarely need the response over the entire
image. Instead we compute it inside small regions of interest around
coarse corner predictions.

The `Roi` struct:

```rust
pub struct Roi {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}
```

describes an axis‑aligned rectangle in image coordinates. A
specialized function:

```rust
pub fn chess_response_u8_patch(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
    roi: Roi,
) -> ResponseMap
```

computes a response map only inside that ROI, treating the ROI as a
small image with its own (0,0) origin. This is used in the multiscale
pipeline to refine coarse corners without paying the cost of
full‑frame response computation at the base resolution.

---

## 3.3 Detection pipeline

The response map is only the first half of the detector. The second
half—implemented in `detect.rs`—turns responses into subpixel corner
candidates.

### 3.3.1 Thresholding and NMS

The main stages are:

1. **Thresholding** – we reject responses that are too small to be
   meaningful. The paper's contract is "any strictly positive `R` is
   a corner candidate", which is what the default settings encode:
   - The default is an absolute threshold at `0.0` combined with a
     strict `R > thr` comparison, i.e. accept iff `R > 0`.
   - Callers can opt into a relative threshold (`threshold_rel`,
     expressed as a fraction of the maximum response in the image)
     by setting `threshold_mode = "relative"` — useful as an
     adaptive policy on high‑contrast scenes where the raw positive
     response floor contains sensor noise.
   - Or tune the absolute threshold upward directly to suppress
     flat‑region noise without committing to a scene‑max policy.
2. **Non‑maximum suppression (NMS)** – in a window of radius
   `nms_radius` around each pixel, we keep only local maxima and
   suppress weaker neighbors.
3. **Cluster filtering** – we require that each surviving corner have
   at least `min_cluster_size` positive‑response neighbors in its NMS
   window. This discards isolated noisy peaks that don’t belong to a
   structured corner.

The result of this stage is a set of raw corner candidates, each
carrying:

- integer‑like peak position,
- response strength (before refinement).

### 3.3.2 Subpixel refinement

To reach subpixel accuracy, the detector runs a 5×5 refinement step
around each candidate:

- a small window is extracted around the integer peak,
- local gradients / intensities are analyzed to estimate a more
  precise corner position,
- the refined position is stored as `(x, y)` in floating‑point form.

The internal type representing these refined candidates is
`descriptor::Corner`:

```rust
pub struct Corner {
    /// Subpixel location in image coordinates (x, y).
    pub xy: [f32; 2],
    /// Raw ChESS response at the integer peak (before COM refinement).
    pub strength: f32,
}
```

The refinement logic is designed to preserve the detector’s noise
robustness while giving more precise coordinates for downstream tasks
like calibration.

---

## 3.4 Corner descriptors

Raw corners (position + strength) are enough for many applications,
but the core crate also offers a richer `CornerDescriptor` that fits a
parametric intensity model to the ring samples around each corner. The
fit yields both local grid axes **independently**, their per‑axis 1σ
angular uncertainty, a bright/dark contrast amplitude, and the RMS fit
residual — all in one pass.

### 3.4.1 `CornerDescriptor`

Defined in `descriptor.rs`:

```rust
pub struct CornerDescriptor {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub contrast: f32,
    pub fit_rms: f32,
    pub axes: [AxisEstimate; 2],
}

pub struct AxisEstimate {
    pub angle: f32,
    pub sigma: f32,
}
```

Fields:

- `x`, `y` – subpixel coordinates in full‑resolution image pixels.
- `response` – raw, unnormalized ChESS response
  `R = SR − DR − 16·MR` at the detected peak. Units are 8‑bit pixel
  sums; the paper's contract is `R > 0`.
- `contrast` – fitted bright/dark amplitude `|A|` in gray levels.
  Independent from `response` and not comparable to it.
- `fit_rms` – root‑mean‑squared residual of the two‑axis intensity
  fit (gray levels). Smaller means the ring sampled cleanly through
  a chessboard‑like corner.
- `axes[0]`, `axes[1]` – the two local grid axis directions and
  their 1σ uncertainties.

The axis convention:

- `axes[0].angle ∈ [0, π)` — the "line direction" of axis 1.
- `axes[1].angle ∈ (axes[0].angle, axes[0].angle + π)`.
- Rotating CCW from `axes[0].angle` toward `axes[1].angle` traverses
  a **dark** sector; the second half‑turn crosses the other dark
  sector, and the remaining two sectors are bright.
- The two axes are **not** assumed orthogonal — a projective warp
  (or strong lens distortion) tilts the two sectors independently.

### 3.4.2 Two‑axis intensity model

The ring samples `s₀, …, s₁₅` at angles `φ₀, …, φ₁₅ = atan2(dy, dx)`
are fitted to

```text
I(φ) = μ + A · tanh(β·sin(φ − θ₁)) · tanh(β·sin(φ − θ₂))
```

with fixed `β = 4.0`. The four free parameters are:

- `μ` – ring‑level mean intensity,
- `A` – bright/dark amplitude (signed during optimization,
  canonicalized to non‑negative on exit),
- `θ₁`, `θ₂` – the two grid axis directions.

Intuition: each `tanh(β · sin(φ − θᵢ))` is a smooth approximation of
`sign(sin(φ − θᵢ))`, i.e. +1 on one side of the axis line and −1 on
the other. Their product is +1 in two antipodal "bright" sectors and
−1 in the two "dark" sectors, matching a chessboard X‑junction. The
fixed `β` reflects the effective ring‑integration blur at the sampled
radius and is not a fit parameter.

### 3.4.3 Gauss–Newton solver

`fit_two_axes` runs a small Gauss–Newton iteration (up to 6 steps):

1. Seed `θ₁`, `θ₂` from the 2nd‑harmonic of the centred ring samples
   (the legacy single‑axis estimator), placed at the sector midpoint
   ± π/4. Seed `A` from the harmonic magnitude.
2. At each step, evaluate the residuals and the 16×4 Jacobian of
   `I(φᵢ)` with respect to `[μ, A, θ₁, θ₂]` and solve the normal
   equations `JᵀJ · Δ = Jᵀ r` with partial pivoting.
3. Clamp angular updates to ±0.5 rad per step to prevent runaway.
4. Stop once the update falls below `‖Δθ‖ < 10⁻⁴` or the iteration
   cap is reached.
5. Canonicalize `(θ₁, θ₂, A)` so that `A ≥ 0`, `θ₁ ∈ [0, π)` and the
   CCW arc from `θ₁` to `θ₂` spans a dark sector.

Flat or near‑flat rings (ring variance below `10⁻⁶`, or 2nd‑harmonic
magnitude below `10⁻⁴`) short‑circuit to a degenerate fit:
`A = 0`, `θ₁ = 0`, `θ₂ = π/2`, and `σ = π` on both axes so downstream
consumers can detect the "no signal" case via the uncertainty field.

### 3.4.4 Per‑axis 1σ uncertainty

The `sigma` field on each `AxisEstimate` is the standard 1σ angular
uncertainty from the linearised Gauss–Newton covariance at the
optimum:

1. The sum of squared residuals is `SSR = Σᵢ (sᵢ − I(φᵢ))²`.
2. The unbiased residual variance is
   `σ̂² = SSR / (N − p) = SSR / (16 − 4) = SSR / 12`.
3. The parameter covariance is `Σ = σ̂² · (JᵀJ)⁻¹`, where `JᵀJ` is
   the final Gauss–Newton normal matrix.
4. The angle sigmas are the relevant diagonal entries:
   `σθ₁ = √Σ[2,2]`, `σθ₂ = √Σ[3,3]` (clamped to ≥ 0, capped at π).

This is the textbook Cramér–Rao‑style uncertainty for nonlinear
least squares — it assumes residuals are approximately iid Gaussian
and the linearisation around the optimum is adequate. It does **not**
account for model mismatch (e.g. a corner that is not well described
by a separable two‑axis tanh product), but it scales correctly with
SNR: noisier rings produce proportionally larger `sigma`.

Practically, `sigma` is useful for:

- Weighting corners in downstream grid fitting (inverse‑variance
  weights, or rejecting corners whose axes are too uncertain).
- Flagging degenerate fits: `sigma ≈ π` means the fit did not lock
  onto a well‑defined grid.

### 3.4.5 From corners to descriptors

The function:

```rust
pub fn corners_to_descriptors(
    img: &[u8],
    w: usize,
    h: usize,
    radius: u32,
    corners: Vec<Corner>,
) -> Vec<CornerDescriptor>
```

turns raw `Corner` values into full descriptors by:

1. Sampling the 16‑point ring around each corner with bilinear
   interpolation (`sample_ring`).
2. Running `fit_two_axes` to obtain `(μ, A, θ₁, θ₂)`, the
   Gauss–Newton covariance, and the residual RMS.
3. Canonicalising the axes and packaging everything into a
   `CornerDescriptor`.

The pass is deterministic and purely local — there is no global
optimisation or topology reasoning at this stage.

### 3.4.6 When to use descriptors

You get `CornerDescriptor` values when you use the high‑level APIs:

- `chess-corners-core` users can run the response and detector
  stages manually and then call `corners_to_descriptors`.
- `chess-corners` users get `Vec<CornerDescriptor>` directly from
  helpers such as `find_chess_corners_image`,
  `find_chess_corners_u8`, or the multiscale APIs.

For many tasks, `x`, `y`, and `response` are enough. When you need
more insight into local structure — grid fitting, lens‑distortion
modelling, calibration with per‑corner weights, or outlier rejection
before bundle adjustment — `axes`, `sigma`, `contrast`, and `fit_rms`
are the extra handles you get "for free" with each detection.

---

In this part we dissected the `chess-corners-core` crate: how rings
and sampling geometry are defined, how the dense ChESS response is
computed, how the detector turns responses into subpixel candidates,
and how those candidates are enriched into descriptors. In the next
part we will build on this by examining the multiscale pyramids and
coarse‑to‑fine refinement pipeline in more detail.
