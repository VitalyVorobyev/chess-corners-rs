# Part V: Performance, accuracy, integration

This part is a decision guide. It does not claim to be a research
paper. The goal is that, after reading it, you can answer three
practical questions for your own use case:

1. Which refiner to pick for *your* imagery.
2. What accuracy and latency to expect.
3. Where the library's limits are, so you know when to look
   elsewhere.

All numbers on this page are measured on a MacBook Pro M4 with the
release-mode `chess-corners` built at the current tip. Hardware moves
the absolutes, not the relative ordering between refiners.

## 5.1 Picking a refiner in 30 seconds

| Your scene                                                  | Use                                  |
|-------------------------------------------------------------|--------------------------------------|
| High-contrast calibration board, cell ≥ 7 px, low noise     | **RadonPeak**                        |
| Same but latency-critical (≫ 100 k corners/s)               | **SaddlePoint**                      |
| Small cells (4–6 px) — ChArUco, dense targets               | **Förstner** or **RadonPeak**        |
| Heavy additive noise (σ ≥ 10 gray levels)                   | **ML (ONNX v4)** or **CenterOfMass** |
| Blurred image (σ ≥ 1 px camera PSF)                         | **RadonPeak** or **SaddlePoint**     |
| You already use OpenCV and want a one-line refiner          | **`cv2.cornerSubPix`** (cell ≥ 7)    |
| You need a learned component for sim-to-real downstream ML  | **ML (ONNX v4)**                     |

The rest of the chapter explains these choices with plots.

## 5.2 The benchmark fixture

Every comparison on this page uses the same synthetic fixture, run
by `crates/chess-corners/examples/bench_sweep.rs` (Rust refiners)
and `tools/book/opencv_subpix_sweep.py` (OpenCV). The fixture is an
8× supersampled, AA-rasterised periodic chessboard at 45 × 45
pixels, with a corner placed on a 6 × 6 grid of sub-pixel offsets
(36 offsets) inside a single cell. Each condition layers on a
Gaussian blur, additive Gaussian noise, or contrast compression:

![Benchmark fixture examples](img/bench/synth_grid.png)

The red cross marks the true sub-pixel corner position. Every
refiner is seeded at `round(true_corner)`, so it has at most a
±0.5 px starting error. We measure how close its refinement lands
to the ground truth across 36 offsets per condition.

**Why AA hard cells?** That is what a real camera produces when it
images a printed chessboard: hard intensity transitions between ink
and paper, softened at edges by the optical PSF, then sampled
discretely by the sensor. The "tanh-saddle" that the v2 ML model
was trained on is a convenient mathematical idealisation but does
not look like this; see
[§5.7](#57-the-ml-refiner-what-we-can-and-cannot-do) for what that
distribution-mismatch cost us.

![Hard-cells vs tanh](img/bench/synth_modes.png)

## 5.3 Accuracy, the full picture

### Clean data, cell size 8 px

![Error CDF on clean cell=8](img/bench/error_cdf_clean.png)

The CDFs speak for themselves: `RadonPeak` (red) has the lowest
errors across the whole distribution — its 95th percentile is under
0.12 px. `Förstner` and `cv2.cornerSubPix` cluster in the next band
(mean ≈ 0.06 px, p95 ≈ 0.1 px). `CenterOfMass` and ML are around
0.09 mean / 0.15 p95. `SaddlePoint` has a fat right tail on this
particular AA-rendered fixture — one of its parabolic-fit
ill-conditioned cases hits the same offsets repeatedly, dragging
its mean up.

### Varying Gaussian blur σ

![Accuracy vs blur](img/bench/accuracy_vs_blur.png)

**Note:** log y-axis. The shaded 0.05–0.10 px strip marks the
shipping band — where a refiner needs to land to be usable without
ad-hoc per-scene tuning. `Förstner` is a gradient-based method:
Gaussian smoothing collapses the gradient magnitudes its structure
tensor depends on, so its error grows roughly linearly with blur σ.
Every other refiner stays inside the shipping band up to σ = 2.5 px
— heavier blur than most real cameras produce. `RadonPeak` and
`cv2.cornerSubPix` are the most blur-robust: their Radon /
gradient-flow formulations integrate over neighbourhoods that remain
informative even when the step edge is smoothed.

### Varying additive noise σ

![Accuracy vs noise](img/bench/accuracy_vs_noise.png)

**Note:** log y-axis, same shipping band. Noise is where ML shines.
At σ ≥ 8 gray levels (not uncommon under low light) the ONNX v4
model is the *most* accurate refiner — ahead of every hand-rolled
classical method. The mechanism: the CNN was trained with
noise σ ∈ [0, 10] and learned a denoising-like feature extractor in
its first few layers. Classical refiners that fit local quadratic /
Radon structure take the noise in at face value.

### Varying cell size

![Accuracy vs cell size](img/bench/accuracy_vs_cell.png)

This is the plot with the most operational content. Two refiners
break at small cell sizes:

- **`cv2.cornerSubPix`** — the default `winSize=(5, 5)` gives an
  11 × 11 search window. When the cell is 5–6 px wide, the window
  crosses into the two neighbouring corners and `cornerSubPix`
  collapses into a ~3 px mean error. The fix is to pass a smaller
  `winSize` (e.g. `(2, 2)` for cell = 5), but callers often forget.
- **`CenterOfMass`** — uses the radius-5 ChESS response ring. At
  cell = 5 the ring crosses cell boundaries and the response's
  centroid is biased by the neighbour, giving 0.4 px mean error.
  At cell = 6 the cross-over is minimal and it recovers.

`RadonPeak`, `Förstner`, `SaddlePoint`, and ML all look at a 2–3 px
local neighbourhood and are cell-size-agnostic. If you know your
targets use small cells (ChArUco markers, dense patterns), pick one
of those four.

## 5.4 Throughput and the Pareto frontier

![Throughput-accuracy trade-off](img/bench/throughput_vs_accuracy.png)

Note the log-log axes. Two orders of magnitude separate the fastest
classical refiner (0.02 µs / corner for `CenterOfMass`) from the ML
inference (~250 µs / corner at batch = 1). The Pareto frontier, from
fast-and-loose to slow-and-tight:

- **`CenterOfMass`** — 0.02 µs, 0.08 px. Unbeatable throughput; use
  only where cells match the ChESS ring (see §5.3 cell plot).
- **`Förstner`** — 0.06 µs, 0.06 px. Better accuracy than CoM in
  exchange for 3× the cost. Breaks under blur.
- **`SaddlePoint`** — 0.12 µs, 0.11 px. Stable across conditions.
  The sensible default when you're not sure which refiner fits.
- **`RadonPeak`** — 17 µs, 0.049 px. Most accurate on clean /
  blurred data. 140× the cost of `SaddlePoint`, but still fits
  comfortable calibration-rate budgets (thousands of corners per
  frame, < 100 ms).
- **`ML (ONNX v4)`** — 250 µs, 0.09 px. Only worth its cost when
  you're noise-dominated. The ONNX path batches in production, so
  amortised cost is typically 2–5× lower than the batch-1 number
  above.
- **`cv2.cornerSubPix`** — 2.7 µs at our default settings (measured
  refinement-only; earlier revisions of this chapter reported ~300 µs
  because the timing loop also included fixture construction, see
  `tools/book/opencv_subpix_sweep.py`). As accurate as `RadonPeak` on
  clean data and similarly blur-robust, at a fraction of the cost;
  the footgun is the `winSize=(5, 5)` default, which fails at cell≤6
  (see §5.3).

Benchmark reproduction:

```sh
# Rust refiners — produces bench_sweep.json
cargo run --release -p chess-corners --example bench_sweep \
    --features ml-refiner > book/src/img/bench/bench_sweep.json

# OpenCV cornerSubPix (same fixture) — produces opencv_subpix_sweep.json
python tools/book/opencv_subpix_sweep.py \
    --out book/src/img/bench/opencv_subpix_sweep.json

# Plot everything
python tools/book/plot_benchmark.py
```

## 5.5 Pipeline throughput (whole detector)

The per-refiner numbers above measure only the refinement step. For
a full detect-and-refine call on a real image, the dominant cost is
elsewhere. Below, timings on three test images at release build
averaged over 10 runs:

| Config           | Features     | small (720×540) | mid (1200×900) | large (2048×1536) |
|------------------|--------------|------:|-----:|------:|
| Single-scale     | none         | 3.01  | 4.46 | 26.02 |
| Single-scale     | simd         | 1.29  | 1.74 | 10.00 |
| Single-scale     | rayon        | 1.14  | 1.41 |  6.63 |
| Single-scale     | simd+rayon   | 0.92  | 1.15 |  5.34 |
| Multiscale (3 l) | none         | 0.63  | 0.70 |  4.87 |
| Multiscale (3 l) | simd         | 0.40  | 0.42 |  2.77 |
| Multiscale (3 l) | rayon        | 0.48  | 0.52 |  1.94 |
| Multiscale (3 l) | simd+rayon   | 0.49  | 0.54 |  1.59 |

Numbers in milliseconds. The breakdown — refine dominates, coarse
detect sits around 0.08–0.75 ms, merge is negligible — means that
picking the slower `RadonPeak` refiner adds 5–15 ms on a 1000-corner
calibration image, which is usually fine. Picking the ML refiner
adds ~30 ms per 100 corners at batch = 1.

**Feature guidance:**

- Enable **`simd`** on any target that has it. It's the dominant
  win regardless of image size.
- Add **`rayon`** for large inputs (≥ 1080p); it hurts slightly on
  small images due to thread-startup overhead.
- **Multiscale** is the right default. Single-scale is only
  worthwhile when you need maximal seed stability at the cost of
  3–5× wallclock.

## 5.6 Comparison with OpenCV's chessboard detectors

OpenCV ships two classical corner refinement paths that are
standard in calibration pipelines:

- **`cv2.cornerSubPix`** — iterative gradient-based refiner. Works
  on any candidate seed, not just chessboard corners. Covered on
  the accuracy plots above.
- **`cv2.findChessboardCornersSB`** — a full chessboard detector.
  Not a direct refiner comparison, but a useful reference for
  whole-pipeline accuracy on real calibration images.

On the public
[Stereocamera Chessboard](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures)
dataset (2 × 20 frames, 77 corners each):

- Pairwise distance, **`findChessboardCornersSB`** vs our ChESS +
  default refiner: mean 0.21 px.
- Pairwise distance, `cv2.cornerHarris + cornerSubPix` vs ChESS:
  mean 0.24 px.

Neither "truth" is perfect; both pipelines have their own
sub-pixel biases on real imagery. The take-away is that on real
calibration frames the three methods agree to within ~0.25 px,
which is consistent with the distributions we see on synthetic
fixtures. Our detector is also ~30× faster than
`findChessboardCornersSB` on the same frames (≈ 4 ms vs ≈ 115 ms).

## 5.7 The ML refiner: what we can and cannot do

The embedded ONNX model is trained on a 50/50 mix of AA hard-cell
chessboard patches (the benchmark fixture's distribution) and
legacy tanh saddles (a smoother idealisation used by earlier
models). Training code and configs live under `tools/ml_refiner/`;
see `docs/proposal-ml-refiner-v3.md` for the retrain history and
`docs/refiner-comparison.md` for the head-to-head table.

**What v4 achieves:**

- Robust across distribution shifts: hard-cells and tanh both
  work. Versions ≤ 3 failed catastrophically on one or the
  other (`v2` was tanh-only → 0.5 px on hard-cells; `v3` was
  hard-cells-only → 0.6 px on tanh).
- Best-in-class on noise: wins at σ ≥ 8 (see §5.3 noise plot).

**What v4 does not achieve:**

- It does not beat `RadonPeak` on clean / blurred data
  (0.09 px vs 0.05 px mean). We tried three architectures at
  increasing capacity (180 K → 730 K params → 50 K-param
  spatial-softargmax head); all converged to the same ~0.14 px
  plateau on the held-out val set. This is a **learning-gap**
  (generic regression on 200 K synthetic patches has not
  discovered RadonPeak's 4-angle Radon + Gaussian-log peak-fit
  structure), not a capacity gap. Closing it likely requires
  distillation from `RadonPeak` or orders of magnitude more data;
  both are explicitly out of scope for the current release.

**When to use ML:**

- Noise-dominated scenes (σ ≥ 8 on 8-bit).
- You want a single learned component you can retrain for
  a specific target distribution (e.g. particular lens, lighting).
- Downstream pipeline already runs ONNX and batching is free.

**When not to:**

- Clean calibration work — classical is faster and more accurate.
- Mixed CPU/GPU budgets where 250 µs per corner is a problem.

## 5.8 Tracing and diagnostics

Build with the `tracing` feature to emit JSON spans for every step
of the pipeline. The named spans are:

- `find_chess_corners` — total detect-and-refine wall time.
- `single_scale` — single-scale path body.
- `coarse_detect` — response computation + candidate extraction.
- `refine` — per-seed refinement (carries `seeds` count as an
  attribute).
- `merge` — duplicate suppression across pyramid levels.
- `build_pyramid` — pyramid construction.
- `ml_refiner` — emitted only on the ML path; includes a timing
  event with batch size.

Capture traces via the CLI's `--json-trace` flag or
`tools/perf_bench.py`. `tools/trace/parser.py` extracts them into
CSV / JSON for aggregation and `tools/perf_bench.py` produces the
table in §5.5.

A typical diagnostic workflow when a frame produces fewer corners
than expected:

1. Rerun with `tracing` + `--json-trace` and inspect the
   `coarse_detect` span's `candidates` count — if it's zero or
   very low, ChESS failed to seed and the refiner never runs.
2. If seeds are present but the `refine` span rejects most of
   them, the refiner's acceptance thresholds are tripping. Swap
   refiner via `ChessConfig::refiner` or relax the specific
   rejection (e.g. `SaddlePointConfig::max_offset`).
3. If accuracy is fine but wall time is bad, the `refine` span
   usually dominates — switch from `RadonPeak` / ML to
   `SaddlePoint` for 100–1000× the throughput with a modest
   accuracy cost.

## 5.9 Integration recipes

### Real-time loop (camera at ≥ 30 fps)

```rust
use chess_corners::{ChessConfig, DetectorMode, RefinementMethod,
                    find_chess_corners_buff_with_refiner,
                    ImageBuffer, PyramidBuffers};

let mut buffers = PyramidBuffers::default();
let mut cfg = ChessConfig::multiscale();
cfg.refiner.kind = RefinementMethod::SaddlePoint;  // stable + fast

loop {
    let frame = next_frame();  // your camera source
    let corners = find_chess_corners_buff_with_refiner(
        frame, &cfg, &mut buffers, &cfg.refiner.to_refiner_kind());
    // use corners…
}
```

Reusing `PyramidBuffers` avoids per-frame allocations. Combined
with `simd` + `rayon` this keeps the detector under 2 ms on 1080p.

### Calibration (offline, maximum accuracy)

```rust
use chess_corners::{ChessConfig, RefinementMethod,
                    find_chess_corners_image};

let mut cfg = ChessConfig::multiscale();
cfg.refiner.kind = RefinementMethod::RadonPeak;
cfg.merge_radius = 2.0;  // tight merge for calibration

let corners = find_chess_corners_image(&image, &cfg);
```

RadonPeak's extra 5–15 ms / frame is invisible in an offline
calibration run.

### Noisy low-light imagery

```rust
use chess_corners::{ChessConfig, find_chess_corners_image_with_ml};

let cfg = ChessConfig::multiscale();
let corners = find_chess_corners_image_with_ml(&image, &cfg);
```

Requires the `ml-refiner` feature. The ML path batches candidates
internally; on a 100-corner frame the effective cost is ~30 ms.

## 5.10 Reproducing everything on this page

```sh
# Rust quality gates + cross-refiner benchmark
cargo test --release -p chess-corners --test refiner_benchmark \
    --features ml-refiner -- --nocapture --test-threads=1

# Dense sweep JSON (feeds Part V plots)
cargo run --release -p chess-corners --example bench_sweep \
    --features ml-refiner > book/src/img/bench/bench_sweep.json

# OpenCV cornerSubPix on the same fixture
python tools/book/opencv_subpix_sweep.py \
    --out book/src/img/bench/opencv_subpix_sweep.json

# Synthetic fixture examples + all Part V plots
python tools/book/synth_examples.py
python tools/book/plot_benchmark.py

# Whole-pipeline timings on the three test images
python tools/perf_bench.py
```
