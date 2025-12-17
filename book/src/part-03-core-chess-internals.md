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
   meaningful:
   - either via a relative threshold (`threshold_rel`) expressed as a
     fraction of the maximum response in the image,
   - or via an absolute threshold (`threshold_abs`), when provided.
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
but the core crate also offers a richer `CornerDescriptor` type that
includes an estimated grid orientation.

### 3.4.1 `CornerDescriptor`

Defined in `descriptor.rs`:

```rust
pub struct CornerDescriptor {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub orientation: f32,
}
```

Fields:

- `x`, `y` – subpixel coordinates in full‑resolution image pixels.
- `response` – ChESS response at the corner, copied from `Corner`.
- `orientation` – orientation of one grid axis at the corner, in
  radians, constrained to `[0, π)`. The other axis is at
  `orientation + π/2`.

### 3.4.2 From corners to descriptors

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

1. Sampling the ring around each corner using bilinear interpolation
   (`sample_ring`).
2. Estimating orientation from the ring samples
   (`estimate_orientation_from_ring`). This essentially measures a
   second‑harmonic over the ring’s angular positions to find the
   dominant grid axis.

All these steps are deterministic, local computations on the original
grayscale image and its immediate neighborhood.

### 3.4.3 When to use descriptors

You get `CornerDescriptor` values when you use the high‑level APIs:

- `chess-corners-core` users can run the response and detector stages
  manually and then call `corners_to_descriptors`.
- `chess-corners` users get `Vec<CornerDescriptor>` directly from
  helpers such as `find_chess_corners_image`,
  `find_chess_corners_u8`, or the multiscale APIs.

For many tasks, you might only use `x`, `y`, and `response`. When you
need more insight into local structure (for example, fitting a grid or
doing downstream topology checks), the `orientation` estimate can be
useful.

---

In this part we dissected the `chess-corners-core` crate: how rings
and sampling geometry are defined, how the dense ChESS response is
computed, how the detector turns responses into subpixel candidates,
and how those candidates are enriched into descriptors. In the next
part we will build on this by examining the multiscale pyramids and
coarse‑to‑fine refinement pipeline in more detail.
