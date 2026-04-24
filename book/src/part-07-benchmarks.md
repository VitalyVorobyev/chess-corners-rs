# Part VII: Benchmarks and performance

This chapter measures what the library does on controlled synthetic
fixtures and on real calibration frames. It covers the five refiners
of [Part V](part-05-refiners.md), the full detector pipeline with
and without optional features, and a comparison against OpenCV's
corner pipeline. All plots are reproducible from the commands at the
end of the chapter.

All absolute numbers are measured on a MacBook Pro M4 with a release
build of the current tree. Hardware changes absolute timings;
relative ordering between refiners is stable across machines we
have tested.

## 7.1 The benchmark fixture

Every accuracy plot in this chapter uses one synthetic fixture, kept
identical between `crates/chess-corners/examples/bench_sweep.rs` (Rust
refiners) and `tools/book/opencv_subpix_sweep.py` (OpenCV). For each
measurement condition:

1. Render a 45×45 grayscale image of a periodic checkerboard with a
   given cell size `c` and a chosen true corner position
   `(x₀, y₀)`. The renderer samples each output pixel at 8×8
   sub-pixel positions inside the pixel footprint and averages the
   samples to produce the gray value. That anti-aliasing step is
   what lets the corner land at a continuous sub-pixel offset
   instead of being quantized to half-pixels by the rasterizer.
2. Apply a Gaussian blur of standard deviation `σ_blur`.
3. Add independent Gaussian noise of standard deviation `σ_noise`
   to every pixel, seeded per offset for reproducibility.
4. Place the true corner `(x₀, y₀)` on a 6×6 regular grid of
   sub-pixel offsets inside one cell — 36 offsets total per condition.
5. Seed each refiner at `round(x₀, y₀)` so the starting error is
   at most ±0.5 px in each axis. Measure Euclidean distance from
   the refined position to the true corner.

![Benchmark fixture samples](img/bench/synth_grid.png)

The red cross in each panel is the true corner. The 36 sub-pixel
offsets cover one full cell period, so each refiner sees the same
range of phase offsets within a cell. That is the only source of
within-condition variability on clean data — `σ_noise = 0` means the
36 error values for a refiner at a given `(cell, blur)` setting are
fully deterministic. This matters for the CDF plot in §7.2.

This rendering mode is what we call "hard cells" elsewhere in this
book: sharp ink/paper step edges, softened by a Gaussian PSF, sampled
by pixels. It is what a camera produces when it images a printed
checkerboard. An alternative fixture common in machine-learning
papers uses a smooth saddle `tanh(x)·tanh(y)` model, which is easier
to differentiate analytically but has no sharp edges and does not
resemble sensor output.

![Hard cells vs tanh saddle](img/bench/synth_modes.png)

We use hard cells as the primary fixture. Part V §5.6.3 discusses
what happens when an ML refiner is trained on `tanh` data only.

## 7.2 Accuracy on clean cells

The figure below is the empirical CDF of Euclidean refinement errors
on the clean fixture (cell = 8 px, no blur, no noise). Each curve has
exactly 36 points — one per sub-pixel offset — so it rises in steps
of `1 / 36 ≈ 0.028`. The stair pattern is not noise; it is the
empirical distribution of a deterministic 36-sample grid, the
standard way to draw an ECDF.

![Error CDF on clean data, cell = 8 px](img/bench/error_cdf_clean.png)

Reading the CDF:

- **RadonPeak** has the lowest errors across the whole distribution.
  Its 95th percentile sits below 0.12 px.
- **Förstner** and **cv2.cornerSubPix** cluster in the next band,
  with mean around 0.06 px and `p95` around 0.10 px.
- **CenterOfMass** and **ML (ONNX v4)** land around 0.09 px mean /
  0.15 px `p95`.
- **SaddlePoint** has a fat right tail on this fixture: its
  parabolic fit becomes ill-conditioned on a subset of the 36
  offsets, and those outliers drag the mean up. With noise or
  blur added, the tail closes (see §7.3).

## 7.3 Accuracy vs blur

Log-y axis. The shaded band at 0.05–0.10 px marks the error range
where a refiner is usable without per-scene tuning.

![Accuracy vs Gaussian blur σ](img/bench/accuracy_vs_blur.png)

- **Förstner** is gradient-based. Gaussian smoothing flattens the
  gradient magnitudes its structure tensor depends on, so its error
  grows roughly linearly in `σ_blur`.
- Every other refiner stays inside the shipping band up to
  `σ_blur = 2.5 px` — heavier blur than most real camera systems
  produce.
- **RadonPeak** and **cv2.cornerSubPix** are the most blur-robust:
  both integrate over neighborhoods that remain informative even
  when the step edge is smoothed.

The Radon detector (not shown on this plot — it is a detector, not a
refiner) inherits this blur tolerance because RadonPeak is exactly
its per-candidate response.

## 7.4 Accuracy vs additive noise

Log-y axis, same shaded band as §7.3.

![Accuracy vs additive noise σ](img/bench/accuracy_vs_noise.png)

This is the regime where the ML refiner wins. At `σ_noise ≥ 8` gray
levels (common in low-light scenes) the ONNX v4 model is the most
accurate refiner in the plot, ahead of every hand-coded method. The
mechanism is that the CNN was trained with noise augmentation over
`σ ∈ [0, 10]` and its early convolutional layers act as a learned
denoiser. Refiners that fit a local quadratic or Radon structure
take the noise in at face value.

## 7.5 Accuracy vs cell size

Largest operational plot. Two refiners break on small cells:

![Accuracy vs cell size](img/bench/accuracy_vs_cell.png)

- **cv2.cornerSubPix** uses a default `winSize = (5, 5)`, which
  means an 11×11 search window. At cell size 5–6 px the window
  crosses into the two neighboring corners and `cornerSubPix`
  collapses to around 3 px mean error. Passing a smaller `winSize`
  (e.g. `(2, 2)` at cell = 5) fixes this, but the default is easy
  to overlook.
- **CenterOfMass** uses the ChESS response ring at radius 5. At
  cell = 5 the ring crosses cell boundaries and the response-map
  centroid is biased by the neighbor corners, giving ~0.4 px mean
  error. At cell = 6 the crossover is minimal and the refiner
  recovers.

**RadonPeak**, **Förstner**, **SaddlePoint**, and **ML** look at a
2–3 px local neighborhood and are cell-size-agnostic down to 4 px.
For dense targets (ChArUco, compact calibration boards) pick from
those four.

## 7.6 Throughput vs accuracy

Log-log axes. Two orders of magnitude separate the fastest refiner
(CenterOfMass at ~20 ns/corner) from ML inference (~250 µs/corner at
batch = 1).

![Throughput vs accuracy](img/bench/throughput_vs_accuracy.png)

The Pareto frontier, fast-to-slow:

| Refiner             | Time / corner | Clean-data mean error | Notes                                                                 |
|---------------------|---------------|-----------------------|-----------------------------------------------------------------------|
| `CenterOfMass`      | ~20 ns        | 0.08 px               | Unmatched throughput; fails at cell ≤ 5 (see §7.5).                   |
| `Förstner`          | ~60 ns        | 0.06 px               | Good on sharp images; degrades with blur.                             |
| `SaddlePoint`       | ~120 ns       | 0.11 px               | Stable across conditions; the no-surprise default.                    |
| `cv2.cornerSubPix`  | ~2.7 µs       | 0.05 px               | OpenCV's refiner. Comparable accuracy to RadonPeak on clean data.     |
| `RadonPeak`         | ~17 µs        | 0.049 px              | Lowest clean/blur error. ~140× SaddlePoint cost.                      |
| `ML (ONNX v4)`      | ~250 µs (b=1) | 0.09 px               | Best on noise-heavy scenes. ONNX runtime batches in production.       |

The OpenCV timing in earlier revisions of this chapter was reported
at ~300 µs because the measurement loop included fixture construction;
the refinement-only value (~2.7 µs) is what the table above quotes.
See `tools/book/opencv_subpix_sweep.py` for the exact measurement
boundaries.

## 7.7 Whole-pipeline throughput

The per-refiner timings above are the refinement step alone. End-to-end
detector-plus-refiner wall times on three test images, release build,
averaged over 10 runs (milliseconds):

| Config           | Features     | small 720×540 | mid 1200×900 | large 2048×1536 |
|------------------|--------------|--------------:|-------------:|----------------:|
| Single-scale     | none         | 3.01          | 4.46         | 26.02           |
| Single-scale     | simd         | 1.29          | 1.74         | 10.00           |
| Single-scale     | rayon        | 1.14          | 1.41         |  6.63           |
| Single-scale     | simd+rayon   | 0.92          | 1.15         |  5.34           |
| Multiscale (3 l) | none         | 0.63          | 0.70         |  4.87           |
| Multiscale (3 l) | simd         | 0.40          | 0.42         |  2.77           |
| Multiscale (3 l) | rayon        | 0.48          | 0.52         |  1.94           |
| Multiscale (3 l) | simd+rayon   | 0.49          | 0.54         |  1.59           |

Observations:

- Refinement dominates: the coarse detect stage takes 0.08–0.75 ms
  across these sizes; merge is a negligible fraction.
- Picking RadonPeak over SaddlePoint adds ~5–15 ms on a 1000-corner
  calibration frame. Usually acceptable for offline calibration,
  marginal for real-time loops.
- Picking ML at batch = 1 adds ~30 ms per 100 corners.

**Feature flag guidance:**

- Enable `simd` on any target that supports portable SIMD. It is
  the single largest win, regardless of image size.
- Add `rayon` for images ≥ 1080p. For smaller images, thread-startup
  overhead makes `rayon` slightly slower than scalar.
- Multiscale is a good default. Single-scale is only worthwhile when
  you want maximum seed stability and can afford 3–5× the wall time.

## 7.8 Comparison with OpenCV

OpenCV ships two reference paths standard in calibration pipelines:

- `cv2.cornerSubPix` — iterative gradient-based refiner, covered in
  §7.2–7.6 above as one of the refiner candidates.
- `cv2.findChessboardCornersSB` — a complete chessboard detector.
  Not a direct refiner comparison, but a useful reference for
  whole-pipeline accuracy on real calibration frames.

On the public
[Stereo Chessboard](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures)
dataset (2 × 20 frames, 77 corners each):

- Pairwise distance, `findChessboardCornersSB` vs our ChESS +
  default refiner: mean 0.21 px.
- Pairwise distance, `cv2.cornerHarris + cornerSubPix` vs ChESS:
  mean 0.24 px.

Neither reference is ground truth; both pipelines have their own
sub-pixel biases on real imagery. The useful takeaway is that on
real calibration frames the three methods agree to within about
0.25 px, consistent with the distributions in §7.2–7.5.

Wall-time, same dataset: `findChessboardCornersSB` takes ~115 ms
per frame; our detector takes ~4 ms per frame — a ~30× speedup.

## 7.9 Tracing and diagnostics

Build with the `tracing` feature to emit JSON spans for each pipeline
step. Named spans:

- `find_chess_corners` — total detect-and-refine wall time.
- `single_scale` — single-scale path body.
- `coarse_detect` — response computation + candidate extraction.
- `refine` — per-seed refinement (carries `seeds` count).
- `merge` — cross-level duplicate suppression.
- `build_pyramid` — pyramid construction.
- `ml_refiner` — ML path only; carries a batch-size event.
- `radon_detect` — Radon path only.

Capture traces via the CLI's `--json-trace` flag or the
`tools/perf_bench.py` script. `tools/trace/parser.py` extracts spans
into CSV or JSON for aggregation, and `tools/perf_bench.py` produces
the table in §7.7.

When a frame returns fewer corners than expected:

1. Check the `coarse_detect` span's candidate count. If it is zero
   or very low, the detector failed to seed and no refinement ran.
   Consider switching `detector_mode` (ChESS ↔ Radon) or lowering
   `threshold_value`.
2. If seeds are present but most `refine` results are rejected,
   the refiner's thresholds are firing. Swap via
   `ChessConfig.refiner.kind` or relax the specific rejection
   (e.g. raise `SaddlePointConfig.max_offset`).
3. If accuracy is fine but wall time is large, `refine` almost
   always dominates — switch RadonPeak / ML to a structure-tensor
   refiner for 100–1000× the throughput at a bounded accuracy cost.

## 7.10 Integration recipes

### Real-time loop, 30+ fps

```rust
use chess_corners::{
    find_chess_corners_buff_with_refiner, ChessConfig, PyramidBuffers, RefinementMethod,
};

let mut buffers = PyramidBuffers::default();
let mut cfg = ChessConfig::multiscale();
cfg.refiner.kind = RefinementMethod::SaddlePoint;  // stable and fast

loop {
    let frame = next_frame();  // your camera source
    let corners = find_chess_corners_buff_with_refiner(
        frame, &cfg, &mut buffers, &cfg.refiner.to_refiner_kind());
    // ...
}
```

Reusing `PyramidBuffers` avoids per-frame allocation. With
`simd + rayon` this stays under 2 ms on 1080p.

### Offline calibration, maximum accuracy

```rust
use chess_corners::{find_chess_corners_image, ChessConfig, RefinementMethod};

let mut cfg = ChessConfig::multiscale();
cfg.refiner.kind = RefinementMethod::RadonPeak;
cfg.merge_radius = 2.0;

let corners = find_chess_corners_image(&image, &cfg);
```

The extra 5–15 ms per frame is invisible in an offline run.

### Low-light / noise-heavy imagery

```rust
use chess_corners::{find_chess_corners_image_with_ml, ChessConfig};

let cfg = ChessConfig::multiscale();
let corners = find_chess_corners_image_with_ml(&image, &cfg);
```

Requires the `ml-refiner` feature. The ML path batches candidates
internally; effective cost on a 100-corner frame is ~30 ms.

### Small cells or heavy defocus — switch detector

```rust
use chess_corners::{find_chess_corners_image, ChessConfig};

let cfg = ChessConfig::radon();  // Radon detector with paper defaults
let corners = find_chess_corners_image(&image, &cfg);
```

The Radon detector remains selective on cells down to ~4 physical
pixels and on Gaussian blurs up to `σ ≈ 2.5 px`, where the ChESS
ring begins to fail (see [Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon)).

## 7.11 Reproducing every plot

```sh
# Cross-refiner accuracy sweep (Rust), feeds all Part VII plots
cargo run --release -p chess-corners --example bench_sweep \
    --features ml-refiner > book/src/img/bench/bench_sweep.json

# Same fixture, OpenCV cornerSubPix
python tools/book/opencv_subpix_sweep.py \
    --out book/src/img/bench/opencv_subpix_sweep.json

# Fixture panels (synth_grid.png, synth_modes.png)
python tools/book/synth_examples.py

# All accuracy + throughput plots
python tools/book/plot_benchmark.py

# Whole-pipeline timings on the three test images
python tools/perf_bench.py

# Integration test that gates refiner accuracy on CI
cargo test --release -p chess-corners --test refiner_benchmark \
    --features ml-refiner -- --nocapture --test-threads=1
```

---

[Part VIII](part-08-contributing.md) describes how to contribute
code, tests, or datasets back to the project.
