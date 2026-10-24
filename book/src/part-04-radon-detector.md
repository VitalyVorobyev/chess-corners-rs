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

\\[
S_\alpha(x, y) = \sum_{k=-r}^{r} I(x + k \cos\alpha,\; y + k \sin\alpha)
\\]

The four sums `S₀, S_{π/4}, S_{π/2}, S_{3π/4}` sample two pairs of
orthogonal directions (horizontal/vertical and the two diagonals). At
an X‑junction, two of the four rays lie near the dark/bright axes and
average toward opposite extremes; the other two cross the junction and
average toward the scene mean. The **Radon response** is the squared
gap between the largest and smallest ray sum:

\\[
R(x, y) = \left( \max_\alpha S_\alpha - \min_\alpha S_\alpha \right)^2
\\]

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
table lookups: `S = T[end] − T[one past start]`. The construction
logic and the lookup pattern are in the Radon response source inside
`chess-corners-core`.

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

Higher factors are clamped to 2. The response
map and detected peaks live at working resolution; the detector
divides peak coordinates by `image_upsample` before returning them,
so output coordinates are always in the input pixel frame.

At `image_upsample = 2`, a `ray_radius` of 4 working-resolution
pixels covers 2 physical input pixels, which is the length used in
the paper's benchmarks.

## 4.4 Peak fit pipeline

Not every peak of `R` is a corner. Beside a straight edge that runs
along one of the four ray directions, the ray parallel to the edge stays
inside one cell while the others cross the edge, and `R` forms a ridge at
about a third of a corner's response. Ripples along the ridge become
peaks — on a sharp board rotated 45°, where every edge follows a diagonal
ray, more of them than corners:

![Left: a sharp synthetic board rotated 45° with every response peak above the threshold; corners circled, the peaks beside the edges crossed. Middle: the response, with bright corner peaks and dimmer ridges along every edge. Right: the saddle ratio of corner peaks clusters near 1, that of the edge peaks near 0.](img/radon_saddle_test.png)

The image tells them apart. Around an X-junction it is a saddle — it
curves up along one diagonal between the grid axes and down along the
other; beside an edge it curves across the edge only. Step 5 below keeps
a peak only when a quadratic fitted to the image there has two curvatures
of opposite sign and comparable size (right panel; synthetic boards,
generated by `tools/render_radon_saddle_overlay.py`).

Once the dense response `R(x, y)` has been computed, the detector
turns it into subpixel corner positions:

1. **Box blur.** A separable `(2·blur_radius + 1)²` box filter
   smooths the response map in place. `blur_radius = 1` (a 3×3 box)
   is the facade preset and matches the paper-style peak-fit pipeline.
2. **Threshold.** The top-level `threshold` field feeds this stage as a
   **relative** fraction: pixels below `threshold · max(R)` are dropped
   (default `0.28`, i.e. 28% of the per-frame maximum). The fraction is
   relative rather than absolute because the Radon score
   `(max_α S_α − min_α S_α)²` is built from ray sums whose magnitude
   grows with ray length and image scale — no fixed `R` means "corner"
   across resolutions, whereas a fixed fraction of this frame's peak tracks the
   scene automatically. (ChESS, whose `R` has a stable per-corner scale,
   instead reads `threshold` as an absolute floor; see
   [Part III §3.3.1](part-03-chess-detector.md#331-thresholding-and-nms).)
   On the coarse-to-fine path (§4.7) `max(R)` is the strongest peak
   found across the refinement windows.
3. **Non-maximum suppression.** Each surviving pixel must be the
   strict maximum within a `(2·nms_radius + 1)²` window. (`nms_radius`
   is set via `DetectorConfig.detection`, shared with the ChESS detector.)
4. **Cluster filter.** The pixel must have at least `min_cluster_size`
   positive neighbors in the NMS window. Rejects isolated noise.
   (`min_cluster_size` is likewise on `DetectorConfig.detection`.)
5. **Saddle test.** A quadratic is fitted to the working image over a
   disk of radius `ray_radius` around the peak, and the peak is kept only
   when the fit's two principal curvatures have opposite signs and the
   smaller is at least 0.2 of the larger (on a sharp board, an
   X-junction whose grid axes are 48° or more apart).
   Heavy blur on small sheared cells washes most of the saddle out, and
   those corners are dropped with the edge peaks.
6. **3-point peak fit.** Given the NMS winner `R_c` at `(x, y)` and
   its four axial neighbors `R_{x±1}`, `R_{y±1}`, fit a 1D parabola
   along each axis to find the subpixel offset. The paper's
   Gaussian peak fit (`PeakFitMode::Gaussian`) fits the parabola to
   `log R` instead, which is the MLE under the assumption that the
   peak is locally Gaussian. The parabolic fallback
   (`PeakFitMode::Parabolic`) operates on the raw `R` values.

The peak fit is a closed-form operation on three samples per axis —
no iteration. Implementation: `radon::fit_peak_frac` in
`chess-corners-core`.

7. **Saddle-point refinement** (optional). With
   `RadonConfig.refiner = RadonRefiner::SaddlePoint(_)`, each peak seeds
   the `SaddlePoint` refiner of [Part V §5.4](part-05-refiners.md#54-saddlepoint)
   on the input image, which re-centres a quadratic fit on the corner
   ([Part VIII](part-08-uncertainty.md)). A corner whose refinement does
   not converge keeps its peak-fit position and covariance; no
   peak is dropped. The default, `RadonRefiner::PeakFit`, stops at
   step 6.

The peak fit locates a maximum of a blurred, supersampled response
map, so on synthetic boards its error grows with lens blur and with
board rotation; the saddle-point fit on the pixels measures lower there
(§4.5).

### Why these defaults

| Parameter               | Default | Reason                                                                                                 |
|-------------------------|---------|--------------------------------------------------------------------------------------------------------|
| `ray_radius`            | 4       | Paper value at `image_upsample = 2`; 2 physical pixels of support.                                     |
| `image_upsample`        | 2       | Paper default. Halves aliasing on the ray endpoints.                                                   |
| `response_blur_radius`  | 1       | 3×3 box used by the preset and by the repository's Radon tests.                                        |
| `threshold` (top-level) | 0.28    | Relative fraction of `max(R)`. `R ≥ 0` everywhere, so a non-zero floor is required; the default drops everything below 28% of the per-frame peak. |
| `nms_radius` (`detection`)      | 4       | Matches `ray_radius` — local maxima should be at least one ray length apart. Shared with ChESS; lives on `DetectorConfig.detection`. |
| `min_cluster_size` (`detection`)| 2       | Requires at least one supporting positive neighbor inside the NMS window. Shared with ChESS; lives on `DetectorConfig.detection`. |
| `peak_fit`              | Gaussian | Log-space 3-point fit used by the paper-style pipeline; parabolic fit is also available.              |
| `refiner`               | PeakFit | The paper's pipeline; `SaddlePoint` is an opt-in refinement on the image (§4.5). |

## 4.5 When to pick ChESS vs Radon

Both detectors run on the same input and produce `CornerDescriptor`
values. The practical differences:

| Property                          | ChESS (Part III)                 | Radon (this chapter)                      |
|-----------------------------------|-----------------------------------|-------------------------------------------|
| Support required                  | 16 samples at radius 5 or 10      | 4 rays of length `2·ray_radius + 1`       |
| Minimum cell size in repo sweep   | Degrades when the ring crosses neighbouring cells | Tested down to ~4 px with `image_upsample = 2` |
| Seeding small / blurred cells     | Misses most corners on 4–5 px cells, and on 60°-sheared 8 px cells at blur σ 2.5 | Seeds 5–12 px cells through blur σ 2.5 at noise σ ≤ 2 (4 px cells through σ 1.5); 60°-sheared 4–6 px cells only through blur σ 1–1.5, past which too little saddle remains for the saddle test (§4.4) |
| Defocus / blur mechanism          | Ring samples can smear across cells | Ray integral averages over a short support |
| Cost per image                    | One ring sample per pixel (+SIMD) | 4 SATs + one dense map per pixel          |
| Memory overhead                   | One `ResponseMap`                 | Four SATs + response + blur scratch       |
| Subpixel refinement               | A refiner stage (`ChessRefiner`)  | Peak fit; optional saddle-point refinement (`RadonRefiner`) |

On synthetic boards, the saddle-point refinement localizes Radon's
corners more precisely than the peak fit. RMS error per axis (px) on
boards formed as a camera forms them (Gaussian lens blur, pixel
integration, exact truth; corners matched within 0.5 px), Radon's peak
fit → with `RadonRefiner::SaddlePoint`:

| Board, cell 8 px           | blur σ 1.5, noise 0 | blur 2.5, noise 0 | blur 1.5, noise 2 | blur 2.5, noise 2 |
|----------------------------|---------------------|-------------------|-------------------|-------------------|
| grid-aligned               | 0.007 → 0.002       | 0.024 → 0.005     | 0.056 → 0.021     | 0.179 → 0.042     |
| rotated 45°                | 0.097 → 0.006       | 0.168 → 0.007     | 0.130 → 0.019     | 0.233¹ → 0.040    |
| sheared 60°                | 0.058 → 0.005       | 0.051 → 0.009     | 0.117 → 0.023     | 0.208 → 0.059     |

¹ only 58% of the corners land within 0.5 px with the peak fit (100%
refined). Refined positions match ChESS with the same refiner wherever
both detect the corner. On synthetic 5–6 px cells the refinement is
never worse than the peak fit; on **4 px cells** the refinement window
spans the neighbouring corners and is no better — with a 60° shear it is
worse (blur 1.5, noise 2: 0.12 px with the peak fit, 0.17 px refined,
and 19% of the refined corners move more than 0.5 px off), so keep the
peak fit there. It adds about 0.5 µs per corner (≤ 3% of a Radon
frame on the test images).

On a real stereo calibration set with an accurate ChArUco target the
refinement also lowers the error: 0.038 px per axis at each corner
against 0.056 px for the peak fit, the same as ChESS with `SaddlePoint`
([Part IX §9.6](part-09-benchmarks.md#96-accuracy-on-real-calibration-images)).
Select `RadonRefiner::SaddlePoint` for calibration. The peak fit keeps
one advantage: it is barely moved by illumination gradients, which bias
the saddle-point fit, so under strong local shading it is the more
accurate of the two ([Part VIII §8.5](part-08-uncertainty.md#85-where-it-does-not-hold)).
On that set the Radon detector found about 70% of the corners, missing
the most blurred ones that ChESS found.

ChESS is faster on the measured clean test images. The Radon detector
is useful when the ChESS ring response does not seed enough corners,
especially in the synthetic small-cell, blur, and low-contrast cases
covered by the tests and Part IX. Treat those measurements as
guidance and validate on your own image distribution when the tradeoff
matters.

The two detectors are not meant to be stacked. They are peers, and
the `DetectorConfig.strategy` enum picks one.

## 4.6 Public API

### Core crate

`chess_corners_core` exposes the Radon pipeline at two levels.

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

The facade runs the Radon detector, refines its peaks per
`RadonConfig.refiner` (§4.4), feeds them into `describe_corners` (the
shared descriptor stage from Part III §3.4), and returns
`CornerDescriptor` values in the input pixel frame. To refine the peaks
with the saddle-point fit:

```rust
use chess_corners::{DetectorConfig, RadonRefiner};

let cfg = DetectorConfig::radon().with_radon(|r| r.refiner = RadonRefiner::saddle_point());
```
 The
orientation method that fills `axes` and `sigma_theta*` is shared with
the ChESS pipeline; see [Part VI: Orientation methods](part-06-orientation-methods.md).

## 4.7 Coarse-to-fine Radon

`DetectorConfig::radon_multiscale()` enables the coarse-to-fine 2×
pyramid the ChESS pipeline uses (see
[Part VII](part-07-multiscale-and-pyramids.md)), driven by the Radon
response kernel, with two levels instead of three. A Radon peak on the
coarse level can sit up to a ray length off the corner, and each level
doubles the input pixels that ray covers: at quarter resolution the
offset outgrows the refinement window, and boards whose cells carry fine
detail lose their corners altogether (on `testimages/small.png`, a
ChArUco board, three levels recover 4 of the 72 single-scale corners,
two levels every board corner among them). The `multiscale` field is top-level on
`DetectorConfig` and is honoured by both detectors symmetrically:

```rust
use chess_corners::{DetectorConfig, Detector, MultiscaleConfig};

// Two-level coarse-to-fine Radon preset:
let cfg = DetectorConfig::radon_multiscale();

// Tune a nested Radon field via the closure mutator:
let cfg = DetectorConfig::radon_multiscale()
    .with_radon(|r| r.ray_radius = 6);

// Or set the pyramid shape directly:
let mut cfg = DetectorConfig::radon_multiscale();
cfg.multiscale = MultiscaleConfig::pyramid(2, 128, 4);

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
multiscale preset. The coarse-to-fine pass only finds corners the
coarse level seeds, so check its recall against single-scale `radon`
before adding levels.

### Tuning

The active [`RadonConfig`] inside
[`DetectionStrategy::Radon`](https://docs.rs/chess-corners/) exposes
every field in §4.4's defaults table. The most common tweaks:

- Heavy blur or low contrast: try `ray_radius` in the 5–6 working-pixel range.
- Small cells (3–4 physical pixels): keep `image_upsample = 2`;
  reduce `ray_radius` to 2–3, and keep the default peak fit rather than
  the saddle-point refinement (§4.5).
- Very clean data: try `response_blur_radius = 0` and a higher relative
  threshold (for example 0.05) to cut weak peaks.
- Noise-heavy scenes: try `response_blur_radius = 2` and verify the
  result count and residuals.

---

Next we move from response maps to what happens after detection:
[Part V](part-05-refiners.md) describes the subpixel refiners, among
them the saddle-point fit that can also refine Radon peaks.
