# Part IV: The Radon response detector

The ChESS detector of Part III samples a ring of 16 pixels around each
candidate and computes a single response value per pixel. It assumes
enough image support to fill the ring cleanly. Under strong blur,
severe defocus, low contrast, or cell sizes smaller than about twice
the ring radius, that support breaks down and the ring response stops
being selective.

The Radon response detector is an alternative based on Duda & Frese
(2018), *"Accurate Detection and Localization of Checkerboard Corners
for Calibration"* (BMVC). Instead of a ring, it integrates along four
rays through each candidate pixel and uses the gap between the
strongest and weakest ray as the corner response. The computation is
kept O(1) per pixel using summed-area tables.

For a self-contained overview of the algorithm, see the
[duda-radon-corners atlas page](https://vitavision.dev/atlas/duda-radon-corners)
on vitavision.dev.

The Radon detector is a full alternative to ChESS — same input
(`&[u8]` grayscale), same output
([`CornerDescriptor`](part-03-chess-detector.md#34-corner-descriptors)),
same place in the pipeline. It is selected via `DetectorConfig.strategy`
by setting it to `DetectionStrategy::Radon(RadonConfig)`.

## 4.1 Ray response

For a candidate pixel `(x, y)` and a ray half-length `r` (working-resolution
pixels), we sum pixel intensities along four rays passing through the
pixel at angles `α ∈ {0, π/4, π/2, 3π/4}`:

```text
S_α(x, y) = Σ_{k=-r}^{r}  I(x + k·cos α,  y + k·sin α)
```

The four sums `S₀, S_{π/4}, S_{π/2}, S_{3π/4}` sample two pairs of
orthogonal directions (horizontal/vertical and the two diagonals). At
an X‑junction, two of the four rays lie near the dark/bright axes and
average toward opposite extremes; the other two cross the junction and
average toward the scene mean. The **Radon response** is the squared
gap between the largest and smallest ray sum:

```text
R(x, y) = (max_α S_α  −  min_α S_α)²
```

`R` is always non-negative. On the idealized checkerboard model used
by the tests, it peaks at X-junctions and stays small on flat regions,
straight edges, and blobs because in each of those cases all four ray
sums end up close to each other.

The squaring keeps the response proportional to the square of image
contrast, which is the natural scale for a log-fit later. It also
removes the sign ambiguity between `dark / bright` and `bright / dark`
corner polarities.

## 4.2 O(1) ray sums via summed-area tables

Naively, each ray sum costs `2r + 1` pixel reads. Computing the full
response map at every pixel is then `O(r · W · H)`, which scales with
the ray length. Four summed-area tables (SATs) reduce it to `O(W · H)`,
independent of `r`:

| Table            | Definition                                              | Reads for `S_α`          |
|------------------|---------------------------------------------------------|--------------------------|
| `row_cumsum`     | `T[y][x] = Σ_{k=0..x} I[y][k]`                          | `S_0` (horizontal ray)   |
| `col_cumsum`     | `T[y][x] = Σ_{k=0..y} I[k][x]`                          | `S_{π/2}` (vertical)     |
| `diag_pos_cumsum`| `T[y][x] = T[y-1][x-1] + I[y][x]` (NW-SE)               | `S_{π/4}` (main diag)    |
| `diag_neg_cumsum`| `T[y][x] = T[y-1][x+1] + I[y][x]` (NE-SW)               | `S_{3π/4}` (anti-diag)   |

Each SAT is built in one pass over the image. A ray sum then costs two
table lookups: `S = T[end] − T[one past start]`. See
`chess_corners_core::detect::radon::response::build_cumsums` for the
construction and `compute_response` for the lookup pattern.

### SAT element type

The default SAT element is `i64`, which handles any image size up to
host memory. The `radon-sat-u32` crate feature switches to `u32`,
which halves SAT memory and widens SIMD-friendly lanes. The tradeoff
is a safe-input cap of `255 · W · H ≤ u32::MAX`, about 16 megapixels.
The entry point rejects inputs beyond that cap rather than letting
the SAT wrap silently.

## 4.3 Working resolution and image upsampling

The paper samples the image on a 2× supersampled grid to place ray
endpoints on half-pixel centers. `RadonDetectorParams.image_upsample`
controls this:

- `image_upsample = 1` — detector operates on the input grid.
- `image_upsample = 2` — input is bilinearly upsampled 2× before
  SATs are built; all subsequent steps run at the higher resolution.
  This is the paper default and the facade preset value.

Higher factors are clamped to 2 (`MAX_IMAGE_UPSAMPLE`). The response
map and detected peaks live at working resolution; the detector
divides peak coordinates by `image_upsample` before returning them,
so output coordinates are always in the input pixel frame.

At `image_upsample = 2`, a `ray_radius` of 4 working-resolution
pixels covers 2 physical input pixels, which is the length used in
the paper's benchmarks.

## 4.4 Peak fit pipeline

Once the dense response `R(x, y)` has been computed, the detector
turns it into subpixel corner positions with the same pipeline used
by the Radon refiner (see Part V):

1. **Box blur.** A separable `(2·blur_radius + 1)²` box filter
   smooths the response map in place. `blur_radius = 1` (a 3×3 box)
   is the facade preset and matches the paper-style peak-fit pipeline.
2. **Threshold.** Pixels below `threshold_abs` (if set) or
   `threshold_rel · max(R)` are dropped. Because `R ≥ 0` everywhere,
   there is no useful "strictly positive" selection analogous to
   ChESS's `R > 0`; callers must pick a non-zero floor. Default is
   `threshold_rel = 0.01` (1% of the map maximum).
3. **Non-maximum suppression.** Each surviving pixel must be the
   strict maximum within a `(2·nms_radius + 1)²` window.
4. **Cluster filter.** The pixel must have at least `min_cluster_size`
   positive neighbors in the NMS window. Rejects isolated noise.
5. **3-point peak fit.** Given the NMS winner `R_c` at `(x, y)` and
   its four axial neighbors `R_{x±1}`, `R_{y±1}`, fit a 1D parabola
   along each axis to find the subpixel offset. The paper's
   Gaussian peak fit (`PeakFitMode::Gaussian`) fits the parabola to
   `log R` instead, which is the MLE under the assumption that the
   peak is locally Gaussian. The parabolic fallback
   (`PeakFitMode::Parabolic`) operates on the raw `R` values.

The peak fit is a closed-form operation on three samples per axis —
no iteration. Implementation: `radon::fit_peak_frac` in
`chess-corners-core`.

### Why these defaults

| Parameter               | Default | Reason                                                                                                 |
|-------------------------|---------|--------------------------------------------------------------------------------------------------------|
| `ray_radius`            | 4       | Paper value at `image_upsample = 2`; 2 physical pixels of support.                                     |
| `image_upsample`        | 2       | Paper default. Halves aliasing on the ray endpoints.                                                   |
| `response_blur_radius`  | 1       | 3×3 box used by the preset and by the repository's Radon tests.                                        |
| `threshold_rel`         | 0.01    | 1 % of `max(R)`. `R ≥ 0` means a strictly positive threshold is required; 0.01 is a conservative floor. |
| `nms_radius`            | 4       | Matches `ray_radius` — local maxima should be at least one ray length apart.                           |
| `min_cluster_size`      | 2       | Requires at least one supporting positive neighbor inside the NMS window.                              |
| `peak_fit`              | Gaussian | Log-space 3-point fit used by the paper-style pipeline; parabolic fit is also available.              |

## 4.5 When to pick ChESS vs Radon

Both detectors run on the same input and produce `CornerDescriptor`
values. The practical differences:

| Property                          | ChESS (Part III)                 | Radon (this chapter)                      |
|-----------------------------------|-----------------------------------|-------------------------------------------|
| Support required                  | 16 samples at radius 5 or 10      | 4 rays of length `2·ray_radius + 1`       |
| Minimum cell size in repo sweep   | Degrades when the ring crosses neighbouring cells | Tested down to ~4 px with `image_upsample = 2` |
| Gaussian blur in repo sweep       | Degrades near σ ≈ 1.5 px          | Lower error through σ ≈ 2.5 px on that fixture |
| Defocus / blur mechanism          | Ring samples can smear across cells | Ray integral averages over a short support |
| Cost per image                    | One ring sample per pixel (+SIMD) | 4 SATs + one dense map per pixel          |
| Memory overhead                   | One `ResponseMap`                 | Four SATs + response + blur scratch       |
| Subpixel refinement               | Needs a separate refiner stage    | Built into the detector's peak fit        |

ChESS is faster on the measured clean test images. The Radon detector
is useful when the ChESS ring response does not seed enough corners,
especially in the synthetic small-cell, blur, and low-contrast cases
covered by the tests and Part VIII. Treat those measurements as
guidance and validate on your own image distribution when the tradeoff
matters.

The two detectors are not meant to be stacked. They are peers, and
the `DetectorConfig.strategy` enum picks one.

## 4.6 Public API

### Core crate

`chess_corners_core::detect::radon` exposes the pipeline at two levels.

**Low-level** — response map only:

```rust
use chess_corners_core::{
    RadonBuffers, RadonDetectorParams, radon_response_u8,
};

let mut buffers = RadonBuffers::new();
let params = RadonDetectorParams::default();
let resp = radon_response_u8(&img_u8, width, height, &params, &mut buffers);

println!("response map is {}×{}", resp.width(), resp.height());
```

`RadonBuffers` holds the upsampled image, the four SATs, the response
map, and the blur scratch. Construct it once and reuse across frames
to avoid per-frame allocations.

**Full pipeline** — detect corners:

```rust
use chess_corners_core::{
    detect_peaks_from_radon, radon_response_u8, RadonBuffers, RadonDetectorParams,
};

let mut buffers = RadonBuffers::new();
let params = RadonDetectorParams::default();
let resp = radon_response_u8(&img_u8, width, height, &params, &mut buffers);
let corners = detect_peaks_from_radon(&resp, &params);
// corners: Vec<Corner> — each has (x, y, strength) in input pixels.
```

### Facade crate

`DetectorConfig::radon()` is a preset that selects the Radon detector
and all its defaults:

```rust
use chess_corners::{DetectorConfig, Detector};

let cfg = DetectorConfig::radon();      // strategy = Radon(RadonConfig)
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&gray_image)?;
// corners: Vec<CornerDescriptor>
```

The facade runs the Radon detector, feeds its output into
`describe_corners` (the shared descriptor stage from Part III §3.4),
and returns `CornerDescriptor` values in the input pixel frame. The
orientation method that fills `axes` and `sigma_theta*` is shared with
the ChESS pipeline; see [Part VI: Orientation methods](part-06-orientation-methods.md).

## 4.7 Coarse-to-fine Radon

`DetectorConfig::radon_multiscale()` enables the same coarse-to-fine
2× pyramid that the ChESS pipeline uses (see
[Part VII](part-07-multiscale-and-pyramids.md)), driven by the Radon
response kernel. The `multiscale` field is top-level on `DetectorConfig`
and is honoured by both detectors symmetrically:

```rust
use chess_corners::{DetectorConfig, Detector, MultiscaleConfig};

// Three-level coarse-to-fine Radon preset:
let cfg = DetectorConfig::radon_multiscale();

// Tune a nested Radon field via the closure mutator:
let cfg = DetectorConfig::radon_multiscale()
    .with_radon(|r| r.ray_radius = 6);

// Or set the pyramid depth directly:
let mut cfg = DetectorConfig::radon_multiscale();
cfg.multiscale = MultiscaleConfig::pyramid(2, 128, 3);

let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&gray_image)?;
```

When to try `radon_multiscale` before single-scale `radon`:

- **Large frames** (≥ 1280 × 960) where full-resolution Radon response
  cost dominates the frame budget.
- **Blurry or low-contrast imagery** across a large field of view where
  the ChESS preset does not emit enough seeds.
- **Tight latency budgets** on large sensors where single-scale
  `radon` is too slow and the ChESS multiscale preset misses corners.

For small frames or when per-frame latency is not a concern,
single-scale `radon` is simpler and worth measuring against the
multiscale preset.

### Tuning

The active [`RadonConfig`] inside
[`DetectionStrategy::Radon`](https://docs.rs/chess-corners/) exposes
every field in §4.4's defaults table. The most common tweaks:

- Heavy blur or low contrast: try `ray_radius` in the 5–6 working-pixel range.
- Small cells (3–4 physical pixels): keep `image_upsample = 2`;
  reduce `ray_radius` to 2–3.
- Very clean data: try `response_blur_radius = 0` and a higher relative
  threshold (for example 0.05) to cut weak peaks.
- Noise-heavy scenes: try `response_blur_radius = 2` and verify the
  result count and residuals.

---

Next we move from response maps to what happens after detection:
[Part V](part-05-refiners.md) describes the subpixel refiners that
can replace the Radon peak-fit when the ChESS detector is in use, and
that can also refine Radon output when extra precision is needed.
