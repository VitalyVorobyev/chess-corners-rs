# Part VII: Multiscale pipeline

Parts II–V treated detection mostly as a single-scale operation: one
call, one image, one response map. In practice, frames vary in scale
and blur — a chessboard can occupy a small fraction of a large sensor,
or sit far enough from the camera that the corner pattern is heavily
blurred. For those cases the `chess-corners` crate offers a
coarse-to-fine multiscale detector built on top of fixed 2× image
pyramids.

This part describes:

- the `DenseDetector` trait that abstracts over the two detectors,
- how the pyramid utilities work,
- how the coarse-to-fine detector uses them,
- how to pick a multiscale configuration.

The multiscale path is available for **both** the ChESS and Radon
detectors. The `multiscale: Option<MultiscaleParams>` field sits at the
top level of `DetectorConfig` and is honoured symmetrically by both.
See [Part IV §4.7](part-04-radon-detector.md#47-coarse-to-fine-radon) for the
Radon-specific preset and when to prefer it over single-scale Radon.

---

## 7.0 The `DenseDetector` trait

The multiscale orchestrator in `crates/chess-corners/src/multiscale.rs`
is generic over a `DenseDetector` implementor. Two zero-sized marker
types in `chess-corners-core` satisfy the trait:

- `ChessDetector` — drives the ChESS ring-based response.
- `RadonDetector` — drives the whole-image Duda-Frese Radon response.

```rust
// chess-corners-core public API (simplified)
pub trait DenseDetector {
    type Params;
    type Buffers: Default;
    type Response<'a> where Self: 'a, Self::Buffers: 'a;

    fn compute_response<'a>(
        &self,
        view: ImageView<'_>,
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a>;

    fn detect_corners(
        &self,
        response: &Self::Response<'_>,
        params: &Self::Params,
        refine_border: i32,
    ) -> Vec<Corner>;

    fn compute_response_patch<'a>(
        &self,
        base: ImageView<'_>,
        roi: (usize, usize, usize, usize),
        params: &Self::Params,
        buffers: &'a mut Self::Buffers,
    ) -> Self::Response<'a>;
}
```

`DenseDetector` and its two implementors are public re-exports of
`chess-corners-core`, so the trait is available to downstream crates
that want to extend the pipeline with a custom response kernel.
Subpixel image-domain refinement (Förstner, saddle-point, …) is
**not** part of the trait — it runs detector-agnostically via
`chess_corners_core::detect::refine_corners_on_image`.

The `chess-corners` facade routes the active `DetectorConfig::strategy`
variant to the corresponding `DenseDetector` implementor at the start of
each `detect` call; neither the user nor the multiscale code needs to
branch on the strategy explicitly.

---

## 7.1 Image pyramids

The pyramid builder itself lives in the standalone
`crates/box-image-pyramid` crate. The `chess-corners` facade depends on
it for multiscale detection and re-exports the main configuration and
buffer types (`PyramidParams`, `PyramidBuffers`, `ImageBuffer`) for
convenience.

The builder is intentionally narrow: no color, no arbitrary scaling;
just fixed 2x downsampling on `u8` grayscale images, with optional
SIMD/`rayon` acceleration when `par_pyramid` is enabled.

### 7.1.1 Image views and buffers

Two basic types represent images:

- `ImageView<'a>` – a borrowed view:

  ```rust
  pub struct ImageView<'a> {
      pub data: &'a [u8],
      pub width: usize,
      pub height: usize,
  }
  ```

  - `ImageView::new(width, height, data)` validates that
    `width * height == data.len()` and returns a view on success.

- `ImageBuffer` – an owned buffer:

  ```rust
  pub struct ImageBuffer {
      pub width: usize,
      pub height: usize,
      pub data: Vec<u8>,
  }
  ```

  It is used as backing storage for pyramid levels and exposes
  `as_view()` to obtain an `ImageView<'_>`.

These types keep the pyramid crate decoupled from any particular image
crate. When you call `Detector::detect` on an `image::GrayImage`, the
`chess-corners` facade converts from `image::GrayImage` to the
raw-slice pyramid API internally.

### 7.1.2 Pyramid structures and parameters

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
    pub min_size: usize,
}
```

- `num_levels` – maximum number of levels (including the base).
- `min_size` – smallest allowed dimension (width or height) for any
  level; once a level would fall below this size, construction stops.

The actual type is `#[non_exhaustive]`, so external code should start
from `PyramidParams::default()` and mutate the public fields.

The default is `num_levels = 1`, `min_size = 128`. If you need more
coarse-to-fine help on small or blurred boards, `num_levels = 2` or
`num_levels = 3` is a common starting point.

### 7.1.3 Reusable buffers

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

The [`Detector`](https://docs.rs/chess-corners) struct in the
`chess-corners` facade owns a `PyramidBuffers` internally; building it
once and calling `detect`/`detect_u8` repeatedly reuses the same
buffers across frames.

### 7.1.4 Building the pyramid

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

### 7.1.5 Downsampling and feature combinations

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

## 7.2 Coarse-to-fine detection

The multiscale detector is implemented in
`crates/chess-corners/src/multiscale.rs`. Its job is to:

- optionally build a pyramid from the base image,
- run the active `DenseDetector` on the **smallest** level to find
  coarse corner candidates,
- refine each coarse corner back in the base image using small ROIs,
- merge near‑duplicate refined corners,
- convert them into `CornerDescriptor` values in base‑image
  coordinates.

### 7.2.1 Coarse-to-fine parameters

Multiscale settings are expressed through `MultiscaleParams`, which is
the `Option<MultiscaleParams>` value at `DetectorConfig.multiscale`:

```rust
pub struct MultiscaleParams {
    pub pyramid_levels: u8,
    pub pyramid_min_size: usize,
    /// ROI radius at the coarse level (ignored when pyramid_levels <= 1).
    pub refinement_radius: u32,
}
```

- `pyramid_levels` – maximum number of levels (including base).
- `pyramid_min_size` – smallest allowed dimension; stops halving once
  a level would fall below this size.
- `refinement_radius` – radius of the ROI around each coarse corner in
  **coarse-level** pixels; converted to base-level pixels internally.

The top-level `DetectorConfig.merge_radius` (in base-image pixels)
controls duplicate suppression after refinement.

`MultiscaleParams::default()` provides a reasonable starting point:

- 3 pyramid levels with minimum size 128,
- ROI radius 3 at the coarse level (scaled up at the base; with 3 levels this is ≈12 px at full resolution).

### 7.2.2 Multiscale workflow under `Detector::detect`

The [`Detector`](https://docs.rs/chess-corners) struct in
`chess-corners` owns a `PyramidBuffers` internally. The multiscale
pipeline is opt-in via `DetectorConfig.multiscale = Some(MultiscaleParams { … })`;
when `multiscale` is `None` the detector takes the single-scale path.
Both the ChESS and Radon strategies are routed through the same
coarse-to-fine orchestrator via the `DenseDetector` trait. The
pipeline on each `detect` / `detect_u8` call is:

1. **Build the pyramid** using the multiscale settings and the
   detector's owned buffers.
   - If the resulting pyramid is empty (e.g., base too small), return
     an empty corner set.
2. **Single‑scale special case** – if the pyramid has only one level:
   - run `chess_response_u8` on the base level,
   - run the detector on the response to get raw `Corner` values,
   - convert them with `describe_corners`,
   - return descriptors directly.
3. **Coarse detection**:
   - take the smallest level in the pyramid (`pyramid.levels.last()`),
   - run `DenseDetector::compute_response` and `DenseDetector::detect_corners`
     to get coarse `Corner` candidates at the coarse scale.
   - if no coarse corners are found, return an empty set.
4. **ROI definition and refinement**:
   - compute the inverse scale `inv_scale = 1.0 / coarse_lvl.scale`,
   - for each coarse corner:
     - map its coordinates up to base image space,
     - skip corners too close to the base image border (to keep enough
       room for the ring and refinement window),
     - convert `cfg.refinement_radius` from coarse pixels to base
       pixels, enforcing a minimum based on the detector's border
       requirements,
     - clamp the ROI to keep it entirely within safe bounds,
     - compute `DenseDetector::compute_response_patch` inside this ROI,
     - rerun `DenseDetector::detect_corners` on the patch response to
       get finer `Corner` candidates,
     - shift patch coordinates back into base‑image coordinates.
   - gather all refined corners.
5. **Merging and describing**:
   - run `merge_corners_simple` with `merge_radius` to combine refined
     corners whose positions are within `merge_radius` of each other,
     keeping the stronger one.
   - convert merged `Corner` values into `CornerDescriptor`s using
     `describe_corners` with `params.descriptor_ring_radius()`.

When the `rayon` feature is enabled, the refinement step processes
coarse corners in parallel; otherwise it uses a straightforward loop.

### 7.2.3 Buffer reuse across frames

`Detector` owns the pyramid and upscale scratch buffers, so calling
`detector.detect(&img)` (or `detector.detect_u8(...)`) repeatedly does
not re-allocate. Build the detector once at start-up and feed
successive frames to it.

---

## 7.3 Choosing multiscale configs

The behavior of the multiscale detector is driven primarily by
`MultiscaleParams` (exposed as `DetectorConfig.multiscale`) plus the
top-level `DetectorConfig.merge_radius`:

- `pyramid_levels`,
- `pyramid_min_size`,
- `refinement_radius`,
- `merge_radius` (top-level field).

Here are some practical guidelines and starting points. These apply
equally to both detectors.

### 7.3.1 Single-scale vs multiscale

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

### 7.3.2 `min_size` and pyramid coverage

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

### 7.3.3 ROI radius

`MultiscaleParams.refinement_radius` is specified in **coarse-level pixels**
and converted to base-level pixels using the pyramid scale. Internally,
the code also enforces a minimum ROI radius that respects:

- the detector's own support radius (ChESS ring or Radon ray length),
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

### 7.3.4 Merge radius

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

### 7.3.5 Putting it together

Some example presets:

- **Default multiscale** (good starting point, ChESS):

  - `DetectorConfig::multiscale()` — 3 levels, `min_size = 128`,
    `refinement_radius = 3`, `merge_radius = 3.0`.

- **Coarse-to-fine Radon** (blurry / low-contrast large frames):

  - `DetectorConfig::radon_multiscale()` — same pyramid shape,
    Radon response kernel.
  - See [Part IV §4.7](part-04-radon-detector.md#47-coarse-to-fine-radon).

- **Fast single-scale** (ChESS, sharp calibration boards):

  - `DetectorConfig::single_scale()` — no pyramid, minimal memory.

- **Robust small-board detection**:

  - `pyramid_levels = 3–4`, `pyramid_min_size` tuned to a handful of
    pixels per square (e.g., 64–128), `refinement_radius = 4–5`,
    `merge_radius = 2.0–3.0`.

Once you’ve chosen parameters that work well for your dataset, you can
encode them in your `DetectorConfig` for library use or in a CLI config
JSON for batch experiments.

---

Next: [Part VIII](part-08-benchmarks.md) — measured accuracy and
throughput for both detectors, every refiner, and the full multiscale
pipeline.
