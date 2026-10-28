# Part IX: Benchmarks and performance

This chapter measures what the library does on controlled synthetic
fixtures and on real calibration frames. It covers the four refiners
of [Part V](part-05-refiners.md), the full detector pipeline with
and without optional features, and a comparison against OpenCV's
corner pipeline. All plots are reproducible from the commands at the
end of the chapter.

All absolute numbers are measured on a MacBook Pro M4 with a release
build of the current tree. Hardware changes absolute timings. Treat
the relative ordering as a property of these fixtures and this
implementation, not as a promise for every camera dataset.

## 9.1 The benchmark fixture

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
fully deterministic. This matters for the CDF plot in §9.2.

This rendering mode is what we call "hard cells" elsewhere in this
book: sharp ink/paper step edges, softened by a Gaussian PSF, sampled
by pixels. It is what a camera produces when it images a printed
checkerboard. An alternative fixture common in machine-learning
papers uses a smooth saddle `tanh(x)·tanh(y)` model, which is easier
to differentiate analytically but has no sharp edges and does not
resemble sensor output.

![Hard cells vs tanh saddle](img/bench/synth_modes.png)

We use hard cells as the primary fixture. [Part V §5.5.3](part-05-refiners.md#553-designing-the-training-distribution) discusses
what happens when an ML refiner is trained on `tanh` data only.

## 9.2 Accuracy on clean cells

The figure below is the empirical CDF of Euclidean refinement errors
on the clean fixture (cell = 8 px, no blur, no noise). Each curve has
exactly 36 points — one per sub-pixel offset — so it rises in steps
of `1 / 36 ≈ 0.028`. The stair pattern is not noise; it is the
empirical distribution of a deterministic 36-sample grid, the
standard way to draw an ECDF.

![Error CDF on clean data, cell = 8 px](img/bench/error_cdf_clean.png)

Reading the CDF:

- **SaddlePoint** has the lowest error on clean data: 0.045 px mean,
  0.06 px worst. It refits on a patch re-centred at its own estimate,
  so it carries no pull toward the seed pixel; what remains is this
  fixture's 1/8 px supersampling of the edges.
- **Förstner** and **cv2.cornerSubPix** follow, with mean around
  0.06 px and `p95` around 0.12–0.13 px.
- **CenterOfMass** and **ML (ONNX)** land around 0.08–0.09 px mean /
  0.12–0.15 px `p95`.

## 9.3 Accuracy vs blur

Log-y axis. The shaded band at 0.05–0.10 px marks the error range
where a refiner is usable without per-scene tuning.

![Accuracy vs Gaussian blur σ](img/bench/accuracy_vs_blur.png)

- **Förstner** loses its hold on the corner as the blur grows. Its
  estimate moves by only a fraction \\(g\\) of the corner's offset from
  the seed pixel, about 0.8 on sharp edges; blur lowers that gain until
  it passes through zero (near a lens blur σ of 2–2.5 px), where the
  estimate no longer depends on where the corner is. Its error grows
  roughly linearly in `σ_blur`
  ([Part VIII §8.3.3](part-08-uncertainty.md#833-the-systematic-term)).
- Every other refiner stays inside the plotted 0.05–0.10 px band up to
  `σ_blur = 2.5 px` on this fixture.
- **SaddlePoint** stays at about 0.045 px at every `σ_blur` from 0 to
  2.5 px, the lowest of the built-in refiners. `cv2.cornerSubPix` falls
  from 0.067 px to the same level by `σ_blur = 1` and is within 0.003 px
  of `SaddlePoint` on the blurred rows (slightly lower at `σ_blur = 1`
  and 2.5).

The Radon detector is not shown on this plot because it is a detector,
not a refiner; detector-level recall and precision are measured separately.

## 9.4 Accuracy vs additive noise

Log-y axis, same shaded band as §9.3.

![Accuracy vs additive noise σ](img/bench/accuracy_vs_noise.png)

Mean error grows with noise for every geometric refiner, and the ML
(ONNX) curve is nearly flat. Among all plotted series, including
`cv2.cornerSubPix`, `SaddlePoint` has the lowest mean error for
`σ_noise` from 1 to 8 gray levels (0.074 px at 8; `cv2.cornerSubPix`
0.086 px); the two tie at `σ_noise = 0`. At `σ_noise = 12` the ML
model is ahead of the hand-coded methods on this synthetic fixture. The model was trained
with noise augmentation over `σ ∈ [0, 10]`; that is the strongest
supported explanation for the flat curve. Validate this separately on
real low-light frames before relying on it as a general rule.

## 9.5 Accuracy vs cell size

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

**Förstner**, **SaddlePoint**, and **ML** look at a 2–3 px local
neighborhood and are cell-size-agnostic down to 4 px.
For dense targets (ChArUco, compact calibration boards) pick from
those three.

## 9.6 Accuracy on real calibration images

The synthetic fixtures have exact truth but idealised optics: straight
edges, a symmetric blur, uniform light. On real images there is no
truth, and a calibration residual mostly measures the lens model and
the target, not the detector. On a real stereo calibration set with an
accurate ChArUco target, the residual of a rig calibration was split
into a smooth term over each camera's image (lens model), a 3-D
displacement per target corner (target error), a smooth term per view,
and the error left at each corner. The first three are the same for
every localizer; the last ranks them
([Part VIII §8.4.2](part-08-uncertainty.md#842-real-calibration-images)):

| Localizer | Corner-local error, RMS per axis | Error over SaddlePoint, blurred corners included |
|-----------|---------------------------------:|-------------------------------------------------:|
| Radon + `RadonRefiner::SaddlePoint` | 0.038 px | — |
| `SaddlePoint`                       | 0.040 px | — |
| `cv2.cornerSubPix`, 11×11 window    | 0.045 px | +0.16 px |
| Radon peak fit                      | 0.056 px | — |
| `CenterOfMass`                      | 0.103 px | +0.07 px |
| `Förstner`                          | 0.164 px | +0.29 px |

The first column covers the corners every localizer finds, about 70%
of the board corners; the Radon detector misses the most blurred ones.
The second adds them for the ChESS refiners and `cornerSubPix` (error
over SaddlePoint, \\(\sqrt{\text{rms}^2 - \text{rms}_{SP}^2}\\) per
axis). The synthetic ranking holds on real data, and Förstner, which
is competitive on the sharp synthetic fixture, is the least accurate on
real images, sharp or blurred. This is one dataset, one camera type and
one target; treat the sizes as indicative and the ordering as the
finding.

## 9.7 Throughput vs accuracy

Log-log axes. Two orders of magnitude separate the fastest refiner
(CenterOfMass at ~16 ns/corner) from ML inference (~150 µs/corner at
batch = 1).

![Throughput vs accuracy](img/bench/throughput_vs_accuracy.png)

The Pareto frontier, fast-to-slow:

| Refiner             | Time / corner | Clean-data mean error | Notes                                                                 |
|---------------------|---------------|-----------------------|-----------------------------------------------------------------------|
| `CenterOfMass`      | ~16 ns        | 0.08 px               | Unmatched throughput; fails at cell ≤ 5 (see §9.5).                   |
| `Förstner`          | ~60 ns        | 0.06 px               | Degrades with blur.                                                   |
| `SaddlePoint`       | ~180 ns       | 0.045 px              | Lowest mean error among the built-in refiners on every row up to `σ_noise = 8`. |
| `cv2.cornerSubPix`  | ~2.7 µs       | 0.07 px               | OpenCV's iterative gradient refiner.                                  |
| `ML (ONNX)`         | ~150 µs (b=1) | 0.09 px               | Lowest error on the heaviest noise row (`σ_noise = 12`).             |

The times are the refinement alone. Under the default
`Uncertainty::Propagated` each geometric refiner also computes its
position covariance, which brings it to about 0.5 µs per corner
([Part VIII §8.2.1](part-08-uncertainty.md#821-cost)).

A measurement loop that includes fixture construction in the timed
region reports ~300 µs for `cv2.cornerSubPix`; the refinement-only
value (~2.7 µs) is what the table above quotes. See
`tools/book/opencv_subpix_sweep.py` for the exact measurement
boundaries.

## 9.8 Whole-pipeline throughput

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
- Picking ML at batch = 1 adds ~15 ms per 100 corners; the geometric
  refiners (SaddlePoint, Förstner) add under 1 µs per corner.

**Feature flag guidance:**

- Enable `simd` when you are on nightly Rust and the target supports
  portable SIMD. It is the largest ChESS speedup in the table above.
- Add `rayon` for images ≥ 1080p; on the small image in this benchmark,
  thread overhead can cancel the benefit.
- Multiscale is the faster preset in this table. Use single-scale when
  you specifically want to compare against the full-resolution detector
  path.

### 9.8.1 Whole-image Radon detector

The Duda-Frese Radon path (`DetectorConfig::radon()`) is an alternative
detector for frames where ChESS's 16-sample ring does not produce enough
seeds in the repository fixtures: heavy blur, low contrast, or small
cells. In this workload the Radon pipeline is several times slower than
the ChESS pipeline at the same resolution. For large frames,
`DetectorConfig::radon_multiscale()` can reduce cost by seeding on a
coarser level before refining in the input image.

Wall times on the same three test images, release build, averaged
over 10 runs (milliseconds), 8-core M-class CPU:

| Stage              | Features    | small 720×540 | mid 1024×576 | large 2048×1536 | 1920×1080 synth |
|--------------------|-------------|--------------:|-------------:|----------------:|----------------:|
| Response only      | none        |               |              |                 | 130.6           |
| Response only      | rayon       |               |              |                 |  27.9           |
| Response only      | simd+rayon  |               |              |                 |  28.0           |
| Full pipeline      | none        |  28.6         |  31.3        | 298             | 162             |
| Full pipeline      | rayon+simd  |  28.8         |  19.6        | 188             | 162             |

How the Radon pipeline uses the `rayon` and `simd` features:

1. **`compute_response`** runs row-parallel under `rayon`, with a
   `Simd<i64, 8>` inner kernel under `simd`. This is the largest
   effect on the response stage (130.6 → 28.0 ms on the 1920×1080
   frame).
2. **`box_blur_inplace`** runs row-parallel under `rayon`; both passes
   traverse the image row-major.
3. **`upsample_bilinear_2x`** runs row-parallel, and the four
   independent cumulative sums are built in parallel with
   `rayon::join`.
4. **`merge_corners_simple`** buckets candidates on a uniform grid
   whose cell size is the merge radius, so merging is close to linear
   in the number of candidates. This matters on synthetic chessboards
   and low-contrast frames, where Radon emits thousands of candidates.
5. **NMS and the peak fit** in `detect_peaks_from_radon` run
   row-parallel under `rayon`, with per-row results concatenated in row
   order so the output is deterministic.

Caveats and notes:

- Portable SIMD on the inner kernel adds ~50% over scalar
  single-thread, but stacks weakly with rayon — the loop is
  memory-bound on M-class CPUs and rayon already saturates DRAM
  bandwidth. SIMD is the bigger relative win on AVX-512 hosts where
  the i64 lane width matches natively.
- The four cumsum directions remain individually serial (each has a
  prefix-sum data dependency along its direction). Only the
  inter-direction parallelism via `rayon::join` is added.
- Small images (`small.png` at 720×540) fall outside the regime
  where rayon's thread-startup overhead amortizes. The
  optimization mostly preserves their wall time rather than
  improving it.

The accuracy guard test
(`crates/chess-corners/tests/perf_accuracy_guard.rs`) locks down
recall, precision, and p95 subpixel error per detector mode so
optimization patches that change correctness fail at `cargo test`
rather than silently drifting in production.

## 9.9 Comparison with OpenCV

OpenCV ships two reference paths standard in calibration pipelines:

- `cv2.cornerSubPix` — iterative gradient-based refiner, covered in
  §9.2–9.7 above as one of the refiner candidates.
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
0.25 px, consistent with the distributions in §9.2–9.5.

Wall-time, same dataset: `findChessboardCornersSB` takes ~115 ms
per frame; our detector takes ~4 ms per frame — a ~30× speedup.

## 9.10 Tracing and diagnostics

Build with the `tracing` feature to emit JSON spans for each pipeline
step. Named spans:

- `single_scale` — single-scale path body.
- `coarse_detect` — response computation + candidate extraction.
- `refine` — per-seed refinement (carries `seeds` count).
- `merge` — cross-level duplicate suppression.
- `build_pyramid` — pyramid construction.
- `ml_refiner` — ML path only; carries a batch-size event.
- `radon_detect` — Radon path only.

Capture traces via the CLI's `--json-trace` flag: one JSON object per
line, with each span's busy time on its close event.

When a frame returns fewer corners than expected:

1. Check the `coarse_detect` span's candidate count. If it is zero
   or very low, the detector failed to seed and no refinement ran.
   Consider switching `DetectorConfig::strategy` (ChESS ↔ Radon) or
   lowering the threshold (`cfg.threshold = Threshold::Relative(_)` or
   `Threshold::Absolute(_)`).
2. If seeds are present but most `refine` results are rejected,
   the refiner's thresholds are firing. Swap via
   `DetectorConfig.refiner.kind` or relax the specific rejection
   (e.g. raise `SaddlePointConfig.max_offset`).
3. If accuracy is fine but wall time is large, inspect the `refine`
   span. On the measured refiner benchmark, replacing ML
   with a structure-tensor refiner (SaddlePoint or Förstner) is much
   cheaper per corner, with the accuracy tradeoff shown in §9.2–§9.7.

## 9.11 Integration recipes

### Real-time loop, 30+ fps

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig};

let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::saddle_point());

let mut detector = Detector::new(cfg)?;

loop {
    let frame = next_frame();  // your camera source
    let corners = detector.detect(&frame)?;
    // ...
}
```

`Detector` reuses pyramid scratch buffers across frames; with
`simd + rayon` this stays under 2 ms on 1080p.

### Offline calibration, maximum accuracy

For still calibration frames, `SaddlePoint` has the lowest mean error
among the built-in refiners on clean and blurred data alike (§9.2, §9.3),
and on real calibration images (§9.6).

```rust
use chess_corners::{ChessRefiner, Detector, DetectorConfig};

let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::saddle_point())
    .with_merge_radius(2.0);

let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&image)?;
```

### Low-light / noise-heavy imagery

```rust
# #[cfg(feature = "ml-refiner")]
# {
use chess_corners::{ChessRefiner, Detector, DetectorConfig};

let cfg = DetectorConfig::chess_multiscale()
    .with_chess(|c| c.refiner = ChessRefiner::Ml);

let mut detector = Detector::new(cfg).unwrap();
let corners = detector.detect(&image).unwrap();
# }
```

Requires the `ml-refiner` feature. The ML path batches candidates
internally; effective cost on a 100-corner frame is ~30 ms.

### Small cells or heavy defocus — switch detector

```rust
use chess_corners::{DetectorConfig, Detector};

let cfg = DetectorConfig::radon();  // Radon detector with paper defaults
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&image)?;
```

In the synthetic sweeps, the Radon detector remains usable on smaller
cells and heavier blur than the default ChESS ring (see
[Part IV §4.5](part-04-radon-detector.md#45-when-to-pick-chess-vs-radon)).

### Blurry / low-contrast imagery on large frames — Radon multiscale

```rust
use chess_corners::{DetectorConfig, Detector};

let cfg = DetectorConfig::radon_multiscale();
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&image)?;
```

Combines the Radon response kernel with the coarse-to-fine pyramid.
Prefer this over single-scale `radon` when the frame is ≥ 1280×960
and latency matters.

## 9.12 Reproducing every plot

```sh
# Cross-refiner accuracy sweep (Rust), feeds all Part IX plots
cargo run --release -p chess-corners --example bench_sweep \
    --features ml-refiner > book/src/img/bench/bench_sweep.json

# Same fixture, OpenCV cornerSubPix
python tools/book/opencv_subpix_sweep.py \
    --out book/src/img/bench/opencv_subpix_sweep.json

# Fixture panels (synth_grid.png, synth_modes.png)
python tools/book/synth_examples.py

# All accuracy + throughput plots
python tools/book/plot_benchmark.py

# Integration test that gates refiner accuracy on CI
cargo test --release -p chess-corners --test refiner_benchmark \
    --features ml-refiner -- --nocapture --test-threads=1

# Recall / precision / p95-error guard against optimization regressions
cargo test --release -p chess-corners --test perf_accuracy_guard \
    --all-features

# Criterion benchmarks (default features = scalar reference); the
# *_pipeline benches time the whole detector on the three test images
cargo bench -p chess-corners        --bench radon_pipeline
cargo bench -p chess-corners        --bench chess_pipeline
cargo bench -p chess-corners-core   --bench refiners
cargo bench -p chess-corners-core   --bench radon_response

# Same benches under SIMD + rayon
cargo bench -p chess-corners        --bench radon_pipeline --features rayon,simd
cargo bench -p chess-corners-core   --bench radon_response --features rayon,simd

# Save a baseline before optimizing, then re-run with --baseline <name>
cargo bench --workspace -- --save-baseline before
cargo bench --workspace -- --baseline before

# Flamegraph / samply harness for hot-path profiling.
# Outputs under testdata/out/profiles/.
tools/profile.sh chess  testimages/large.png
tools/profile.sh radon  testimages/large.png
tools/profile.sh refiner saddle  testimages/mid.png
tools/profile.sh samply chess testimages/large.png
```

---

Next: [Part X](part-10-cpp-bindings.md) — using the detector from
C and C++ through the stable C ABI.
