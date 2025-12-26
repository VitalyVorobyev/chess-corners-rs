# Part IV: Multiscale and Pyramids

In Parts II and III we treated the detector mostly as a single‑scale
operation: run ChESS on an image and get back corners. In practice,
however, real‑world images often vary in scale and blur. A chessboard
might occupy just a small region of a large frame, or it might be far
from the camera so the corner pattern is heavily blurred.

To handle these cases efficiently and robustly, the `chess-corners`
crate offers a **coarse‑to‑fine multiscale detector** built on top of
simple image pyramids. This part describes:

- how the pyramid utilities work,
- how the coarse‑to‑fine detector uses them,
- how to choose multiscale configurations in practice.

---

## 4.1 Image pyramids

The multiscale code lives in `crates/chess-corners/src/pyramid.rs`.
It implements a minimal grayscale pyramid builder tuned for the
detector’s needs: no color, no arbitrary scaling; just fixed 2×
downsampling with optional SIMD/`rayon` acceleration when
`par_pyramid` is enabled.

### 4.1.1 Image views and buffers

Two basic types represent images:

- `ImageView<'a>` – a borrowed view:

  ```rust
  pub struct ImageView<'a> {
      pub width: u32,
      pub height: u32,
      pub data: &'a [u8],
  }
  ```

  - `from_u8_slice(width, height, data)` validates that
    `width * height == data.len()` and returns a view on success.
  - With the `image` feature, `ImageView` can also be constructed
    directly from `image::GrayImage` via `From<&GrayImage>`.

- `ImageBuffer` – an owned buffer:

  ```rust
  pub struct ImageBuffer {
      pub width: u32,
      pub height: u32,
      pub data: Vec<u8>,
  }
  ```

  It is used as backing storage for pyramid levels and exposes
  `as_view()` to obtain an `ImageView<'_>`.

These types keep the pyramid code decoupled from any particular image
crate while remaining easy to integrate when `image` is enabled.

### 4.1.2 Pyramid structures and parameters

An image pyramid is represented as:

- `PyramidLevel<'a>` – a single level with:

  ```rust
  pub struct PyramidLevel<'a> {
      pub img: ImageView<'a>,
      pub scale: f32, // relative to base (e.g. 1.0, 0.5, 0.25, ...)
  }
  ```

- `Pyramid<'a>` – a top‑down collection where `levels[0]` is always
  the base image (scale 1.0), and subsequent levels are downsampled
  copies:

  ```rust
  pub struct Pyramid<'a> {
      pub levels: Vec<PyramidLevel<'a>>,
  }
  ```

The shape of the pyramid is controlled by:

```rust
pub struct PyramidParams {
    pub num_levels: u8,
    pub min_size: u32,
}
```

- `num_levels` – maximum number of levels (including the base).
- `min_size` – smallest allowed dimension (width or height) for any
  level; once a level would fall below this size, construction stops.

The default is `num_levels = 1`, `min_size = 128`. If you need to speed up ceature detection, try `num_levels = 2` or `num_levels = 3`.

### 4.1.3 Reusable buffers

To avoid frequent allocations, `PyramidBuffers` holds the owned
buffers for non‑base levels:

```rust
pub struct PyramidBuffers {
    levels: Vec<ImageBuffer>,
}
```

Typical usage:

1. Construct a `PyramidBuffers` once, often using
   `PyramidBuffers::with_capacity(num_levels)` to pre‑reserve space.
2. For each frame, call the pyramid builder with a base `ImageView`
   and the same buffers. The code automatically resizes or reuses
   internal buffers as needed.

The high‑level multiscale API (`find_chess_corners`) creates and
manages its own `PyramidBuffers` internally, but the lower‑level
`find_chess_corners_buff` entry point lets you supply your own
buffers, which is useful in tight real‑time loops.

### 4.1.4 Building the pyramid

The core builder is:

```rust
pub fn build_pyramid<'a>(
    base: ImageView<'a>,
    params: &PyramidParams,
    buffers: &'a mut PyramidBuffers,
) -> Pyramid<'a>
```

It always includes the base image as level 0, then repeatedly:

1. halves the width and height (integer division by 2),
2. checks against `min_size` and `num_levels`,
3. ensures the appropriate buffer exists in `PyramidBuffers`,
4. calls `downsample_2x_box` to fill the next level.

If `num_levels == 0` or the base image is already smaller than
`min_size`, the function returns an empty pyramid.

### 4.1.5 Downsampling and feature combinations

The downsampling kernel is a simple 2×2 **box filter**:

- for each output pixel, average the corresponding 2×2 block in the
  source image (with a small rounding tweak to keep values in 0–255),
- write the result into the next level’s `ImageBuffer`.

Depending on features:

- without `par_pyramid`, downsampling always uses the scalar
  single-thread path even if `rayon` / `simd` are enabled elsewhere.
- with `par_pyramid` but no `rayon`/`simd`, `downsample_2x_box_scalar`
  runs in a single thread.
- with `par_pyramid` + `simd`, `downsample_2x_box_simd` uses portable
  SIMD to process multiple pixels at once.
- with `par_pyramid` + `rayon`, `downsample_2x_box_parallel_scalar`
  splits work over rows; with both `rayon` and `simd`,
  `downsample_2x_box_parallel_simd` combines row-level parallelism with
  SIMD inner loops.

As with the core ChESS response, all paths are designed to produce
identical results except for small rounding differences; they only
differ in performance.

---

## 4.2 Coarse-to-fine detection

The multiscale detector is implemented in
`crates/chess-corners/src/multiscale.rs`. Its job is to:

- optionally build a pyramid from the base image,
- run the ChESS detector on the **smallest** level to find coarse
  corner candidates,
- refine each coarse corner back in the base image using small ROIs,
- merge near‑duplicate refined corners,
- convert them into `CornerDescriptor` values in base‑image
  coordinates.

### 4.2.1 Coarse-to-fine parameters

The main configuration structure is:

```rust
pub struct CoarseToFineParams {
    pub pyramid: PyramidParams,
    /// ROI radius at the coarse level (ignored when num_levels <= 1).
    pub refinement_radius: u32,
    pub merge_radius: f32,
}
```

- `pyramid` – controls how many levels are built and how small the
  smallest level is allowed to be.
- `refinement_radius` – radius of the ROI around each coarse corner in the
  **coarse‑level** pixels; internally converted to a base‑level radius
  using the pyramid scale.
- `merge_radius` – radius in base‑image coordinates used to merge
  near‑duplicate refined corners (i.e., corners that end up within a
  small distance of each other).

`CoarseToFineParams::default()` provides a reasonable starting point:

- 3 pyramid levels with minimum size 128,
- ROI radius 3 at the coarse level (scaled up at the base; with 3 levels this is ≈12 px at full resolution),
- merge radius 3.0 pixels.

### 4.2.2 The `find_chess_corners_buff` workflow

The main multiscale function is:

```rust
pub fn find_chess_corners_buff(
    base: ImageView<'_>,
    cfg: &ChessConfig,
    buffers: &mut PyramidBuffers,
) -> Vec<CornerDescriptor>
```

It proceeds in several steps:

1. **Build the pyramid** using `cfg.multiscale.pyramid` and the
   provided `buffers`.
   - If the resulting pyramid is empty (e.g., base too small), return
     an empty corner set.
2. **Single‑scale special case** – if the pyramid has only one level:
   - run `chess_response_u8` on the base level,
   - run the detector on the response to get raw `Corner` values,
   - convert them with `corners_to_descriptors`,
   - return descriptors directly.
3. **Coarse detection**:
   - take the smallest level in the pyramid (`pyramid.levels.last()`),
   - run `chess_response_u8` and the detector to get coarse `Corner`
     candidates at the coarse scale.
   - if no coarse corners are found, return an empty set.
4. **ROI definition and refinement**:
   - compute the inverse scale `inv_scale = 1.0 / coarse_lvl.scale`,
   - for each coarse corner:
     - map its coordinates up to base image space,
     - skip corners too close to the base image border (to keep enough
       room for the ring and refinement window),
     - convert `cfg.multiscale.refinement_radius` from coarse pixels to base
       pixels, enforcing a minimum based on the detector’s border
       requirements,
     - clamp the ROI to keep it entirely within safe bounds,
     - compute `chess_response_u8_patch` inside this ROI,
     - rerun the detector on the patch response to get finer `Corner`
       candidates,
     - shift patch coordinates back into base‑image coordinates.
   - gather all refined corners.
5. **Merging and describing**:
   - run `merge_corners_simple` with `merge_radius` to combine refined
     corners whose positions are within `merge_radius` of each other,
     keeping the stronger one.
   - convert merged `Corner` values into `CornerDescriptor`s using
     `corners_to_descriptors` with `params.descriptor_ring_radius()`.

When the `rayon` feature is enabled, the refinement step can process
coarse corners in parallel; otherwise it uses a straightforward loop.

### 4.2.3 Convenience wrapper: `find_chess_corners`

For many applications it’s enough to let the library manage the
pyramid buffers:

```rust
pub fn find_chess_corners(
    base: ImageView<'_>,
    cfg: &ChessConfig,
) -> Vec<CornerDescriptor>
```

This helper:

- constructs a `PyramidBuffers` with capacity for
  `cfg.multiscale.pyramid.num_levels`,
- calls `find_chess_corners_buff`,
- returns the resulting descriptors.

It is the internal entry point behind:

- `find_chess_corners_image` (for `image::GrayImage`), and
- `find_chess_corners_u8` (for raw `&[u8]` buffers).

Use `find_chess_corners_buff` when you want to reuse buffers across
frames; use the higher‑level helpers when you prefer simplicity over
fine‑grained control.

---

## 4.3 Choosing multiscale configs

The behavior of the multiscale detector is driven primarily by
`CoarseToFineParams`:

- `pyramid.num_levels`,
- `pyramid.min_size`,
- `refinement_radius`,
- `merge_radius`.

Here are some practical guidelines and starting points.

### 4.3.1 Single-scale vs multiscale

- **Single-scale**:
  - Set `pyramid.num_levels = 1`.
  - The detector behaves exactly like the single‑scale path: it runs
    ChESS once at the base resolution and skips coarse refinement.
  - This is a good choice when:
    - the chessboard occupies a large portion of the frame,
    - the board is reasonably sharp, and
    - you want maximum recall at a fixed scale.

- **Multiscale**:
  - Use `pyramid.num_levels` in the range 2–4 for most use cases.
  - More levels mean:
    - coarser initial detection (smaller image yields fewer, more
      robust coarse corners),
    - more refinement work at the base level,
    - potentially better robustness when the board is small or heavily
      blurred.

As a rule of thumb, start with `num_levels = 3` and adjust only if you
have specific performance or robustness requirements.

### 4.3.2 `min_size` and pyramid coverage

`pyramid.min_size` limits how small the smallest level can be. If the
base image is small (e.g., smaller than `min_size`), the pyramid may
end up with a single level regardless of `num_levels`, effectively
falling back to single‑scale.

Recommendations:

- Choose `min_size` so that the smallest level still has a few pixels
  per square on the chessboard. If your board is already small in the
  base image, a too‑aggressive `min_size` may collapse the pyramid and
  give you no coarse‑to‑fine benefit.
- For high‑resolution inputs (e.g., 4K), a `min_size` around 128 or
  256 usually works well.

### 4.3.3 ROI radius

`refinement_radius` is specified in **coarse‑level pixels** and converted to
base‑level pixels using the pyramid scale. Internally, the code also
enforces a minimum ROI radius that respects:

- the ChESS ring radius,
- the NMS radius,
- the 5×5 refinement window.

Larger ROIs:

- cost more to process (bigger patches),
- can recover from slightly off coarse positions,
- may pick up nearby corners if multiple corners are close together.

Smaller ROIs:

- are faster,
- assume coarse positions are already fairly accurate.

The default `refinement_radius = 3` is a reasonable compromise. Increase it
if you see coarse corners that consistently refine to the wrong
locations; decrease it if performance is tight and coarse positions
are already good.

### 4.3.4 Merge radius

`merge_radius` controls the distance (in base pixels) used to merge
refined corners. If two corners fall within this radius of each other,
only the stronger one is kept.

Guidelines:

- For typical calibration boards, values around 1.5–2.5 pixels are
  common.
- If your detector tends to produce duplicate corners around the same
  junction (e.g., because the ROI refinement finds multiple close
  maxima), increase `merge_radius`.
- If you need to preserve nearby but distinct corners (e.g., very
  fine grids), consider decreasing it slightly.

### 4.3.5 Putting it together

Some example presets:

- **Default multiscale** (good starting point):

  - `num_levels = 3`
  - `min_size = 128`–`256`
  - `refinement_radius = 3`
  - `merge_radius = 3.0`

- **Fast single-scale**:

  - `num_levels = 1`
  - `min_size` ignored (no pyramid)
  - `refinement_radius` / `merge_radius` unused

- **Robust small‑board detection**:

  - `num_levels = 3`–`4`
  - `min_size` tuned so the smallest level still has a handful of
    pixels per square (e.g., 64–128)
  - `refinement_radius` slightly larger (e.g., 4–5)
  - `merge_radius` around 2.0–3.0

Once you’ve chosen parameters that work well for your dataset, you can
encode them in your `ChessConfig` for library use or in a CLI config
JSON for batch experiments.

---

In this part we explored the multiscale machinery in `chess-corners`:
the minimal pyramid builder, the coarse‑to‑fine detector, and how to
choose multiscale parameters. In the next part we will look at
performance considerations, tracing, and how to integrate the detector
into larger systems while measuring and tuning its behavior.
