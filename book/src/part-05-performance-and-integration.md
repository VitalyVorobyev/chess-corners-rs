# Part V: Performance, Accuracy, and Integration
This part summarizes where the ChESS detector stands today on accuracy and speed (measured on a MacBook Pro M4), how well it matches classic OpenCV detectors on a stereo calibration dataset, how to interpret the traces we emit, and how to integrate the detector into larger pipelines.

We took three test images as use cases:

1. A clear 1200x900 image of a chessboard calibration target:

![](img/mid_chess.png)

2. A 720x540 image of a ChArUco target with not perfect focus:

![](img/small_chess.png)

3. A 2048x1536 image of another ChArUco calibration target:

![](img/large_chess.png)

We traced the ChESS detector for each of these images, and the results are discussed in this part. The first image was also used to compare with OpenCV Harris features and the `findChessboardCornersSB` function. For accuracy, we additionally evaluated the detector on a 2×20‑frame stereo dataset (77 corners per frame).

## 5.1 Performance

The tests below were run on a MacBook Pro M4 (release build). Absolute numbers will vary on your hardware, but the **relative** behavior between configurations is quite stable.

Per‑image timings (ms, averaged over 10 runs; see `book/src/perf.txt` and `testdata/out/perf_report.json` for the full breakdown):

| Config           | Features      | small |  mid | large |
|------------------|--------------|------:|-----:|------:|
| Single‑scale     | none         | 3.01  | 4.46 | 26.02 |
| Single‑scale     | simd         | 1.29  | 1.74 | 10.00 |
| Single‑scale     | rayon        | 1.14  | 1.41 |  6.63 |
| Single‑scale     | simd+rayon   | 0.92  | 1.15 |  5.34 |
| Multiscale (3 l) | none         | 0.63  | 0.70 |  4.87 |
| Multiscale (3 l) | simd         | 0.40  | 0.42 |  2.77 |
| Multiscale (3 l) | rayon        | 0.48  | 0.52 |  1.94 |
| Multiscale (3 l) | simd+rayon   | 0.49  | 0.54 |  1.59 |

Highlights from the timing profiles on small/mid/large images:

- **Multiscale** is the clear winner for speed and robustness.
  - Large image: best total ≈ **1.6 ms** with `simd+rayon` (vs ≈4.9 ms with no features, ≈1.9 ms with `rayon` only).
  - Mid: best total ≈ **0.42 ms** with `simd` alone (rayon adds a bit of overhead at this size).
  - Small: best total ≈ **0.40 ms** with `simd`.
  - Breakdown: refine dominates (0.1–3 ms depending on seeds); coarse_detect sits around 0.08–0.75 ms; merge is negligible.
- **Single-scale** is slower across the board:
  - Large: ≈26 ms, mid: ≈4.5 ms, small: ≈3.0 ms. Use when you need maximal stability and can tollerate some performance drawback.
- **Feature guidance**:
  - Enable **simd** by default; it’s the dominant win on all sizes (although, it requires nightly RUST).
  - Add **rayon** for large inputs (wins on the largest image, minor cost on small/mid).

## 5.2 Accuracy vs OpenCV

The OpenCV `cornerHarris` gives the following result:

![](img/mid_harris.png)

Here is the result of the OpenCV `findChessboardCornersSB` function:

![](img/mid_chessboard.png)

Harris pixel-level feature detection took 3.9 ms. The final result is obtained by using the `cornerSubPix` and manual merge of duplicates. Chessboard detection took about 115 ms. The ChESS detector is much faster as is evident from the previous section. Also, it provides corner orientation that can be handy for a grid reconstruction.

Below we compare the ChESS corners location with the two classical references. We took all images from the [Chessboard Pictures for Stereocamera Calibration](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures) public repository as input. Below are distributions of pairwise distances between corresponding features:

![](img/harris_dist.png)

![](img/chessboard_dist.png)

![](img/harris_vs_chessboard_dist.png)

- Harris vs ChESS: 0.24 pix
- Chessboard vs ChESS: 0.21 pix
- Harris vs Chessboard: 0.12 pix

It is important that the offsets are not biased:
![](img/chessboard_dx.png)
![](img/chessboard_dy.png)

Mean values are much smaller than standard deviation.

## 5.3 Refiner comparison

All five shipped refiners (`CenterOfMass`, `Forstner`, `SaddlePoint`,
`RadonPeak`, and the embedded ONNX `ML`) are benchmarked together in
[`crates/chess-corners/tests/refiner_benchmark.rs`](https://github.com/VitalyVorobyev/chess-corners-rs/blob/main/crates/chess-corners/tests/refiner_benchmark.rs).
The fixture is an 8×-supersampled, AA-rasterised chessboard (36
subpixel offsets on an 8×8 grid inside a cell), with optional
Gaussian PSF and additive noise. Run it with:

```sh
cargo test --release -p chess-corners --test refiner_benchmark \
    --features ml-refiner -- --nocapture --test-threads=1
```

### Accuracy (mean / worst Euclidean-px error, 36 offsets per row)

| condition        | CenterOfMass    | Forstner        | SaddlePoint     | RadonPeak          | ML (ONNX)         |
|------------------|-----------------|-----------------|-----------------|--------------------|-------------------|
| clean (cell=8)   | 0.080 / 0.123   | 0.061 / 0.165   | 0.114 / 0.177   | **0.049** / 0.103  | 0.084 / 0.147     |
| clean (cell=5)   | 0.390 / 0.707   | 0.061 / 0.165   | 0.114 / 0.177   | **0.049** / 0.103  | 0.091 / 0.170     |
| blur σ=1.5       | 0.056 / 0.124   | 0.266 / 0.471   | 0.047 / 0.079   | **0.046** / 0.073  | 0.088 / 0.170     |
| noise σ=5        | 0.088 / 0.156   | 0.135 / 0.301   | 0.095 / 0.220   | **0.085** / 0.183  | 0.086 / 0.150     |
| noise σ=10       | 0.123 / 0.272   | 0.201 / 0.474   | 0.126 / 0.257   | 0.128 / 0.302      | **0.096** / 0.178 |

Accept rate is 36/36 for every refiner in every condition. The
`Forstner / SaddlePoint / RadonPeak` rows are identical between
cell=5 and cell=8 because these refiners only examine a 2–3 px
neighbourhood — the anti-aliased corner looks the same locally at
both cell sizes.

### Throughput (release build, per refinement)

| refiner        | time/call | throughput      | what it costs                                                                 |
|----------------|-----------|-----------------|-------------------------------------------------------------------------------|
| CenterOfMass   | 0.02 µs   | ~50 M corners/s | 5×5 weighted moment over the ChESS response map                               |
| Forstner       | 0.06 µs   | ~17 M corners/s | 5×5 gradient structure tensor solve                                           |
| SaddlePoint    | 0.12 µs   | ~8 M corners/s  | 6-param quadratic LS via Gauss-Jordan                                         |
| **RadonPeak**  | 17.0 µs   | ~59 K corners/s | 13×13 dense Radon (169 samples × 4 rays × 9 bilinear taps) + box blur + fit   |
| ML (ONNX)      | 250 µs    | ~4 K corners/s  | 21×21 patch extract + ONNX inference, batch=1                                 |

The ML path batches better in production (the facade groups
candidates into batches of up to 64), so the per-corner amortised
cost is typically 2–5× lower than the batch=1 figure above.

### Per-refiner notes

- **RadonPeak** — most accurate on clean / blurred data, implements
  the full Duda-Frese (2018) subpixel pipeline (image supersampling,
  4-angle Radon response, box blur, log-space Gaussian peak fit).
  ~300× slower than SaddlePoint but still sub-20 µs.
- **SaddlePoint** — the accuracy/speed sweet spot for most uses.
  Stable across conditions, 0.12 µs per corner.
- **CenterOfMass** — fastest option, but only when the ChESS
  response ring matches the cell size. Cell=5 with the default
  radius-5 ring causes the ring to cross into neighbouring cells
  and the centroid is dragged off the true corner (0.39 px mean
  vs 0.08 px at cell=8).
- **Forstner** — middle-of-the-pack on clean inputs, degrades
  sharply under blur because Gaussian smoothing collapses the
  gradient magnitudes its structure tensor depends on.
- **ML (ONNX)** — v3 model trained on AA hard-cell chessboards
  matching this benchmark's distribution (see
  `tools/ml_refiner/configs/synth_v5.yaml`). Robust across all
  conditions, strongest refiner at σ=10 noise. The previous v2
  model trained on a smooth tanh-saddle distribution scored
  ~0.5 px mean everywhere — a pure distribution-mismatch problem
  fixed by generating training data that matches the AA-rasterised
  chessboards the refiner actually sees in production.

### The ML refiner's training pipeline

The training code lives under `tools/ml_refiner/`:

- `synth/generate_dataset.py` — hard-cell chessboard patches with
  sub-pixel corner offsets, Gaussian PSF, additive noise, and
  photometric jitter (contrast/brightness/gamma). Supports both
  the legacy tanh-saddle renderer (`render_mode: tanh`) and the
  current AA hard-cell renderer (`render_mode: hard_cells`).
- `train.py` — 5-layer CoordConv CNN, smooth-L1 regression on
  `(dx, dy)` plus BCE-with-logits on the `conf` head. MPS + CUDA
  supported.
- `eval_hardcell.py` — post-training evaluation on a held-out
  hard-cell dataset with per-cell / per-blur / per-noise error
  breakdowns and an optional `--gate` flag enforcing the <0.1 px
  shipping threshold on the clean subset.
- `export_onnx.py` — exports a PyTorch checkpoint to ONNX plus
  parity fixtures (`patches.npy`, `torch_out.npy`) that
  `crates/chess-corners-ml/tests/onnx_parity.rs` uses to verify
  the Rust inference path matches PyTorch to <2e-4.

See [`docs/proposal-ml-refiner-v3.md`](https://github.com/VitalyVorobyev/chess-corners-rs/blob/main/docs/proposal-ml-refiner-v3.md)
for the detailed training-distribution diagnosis that led to v3.

## 5.4 Tracing and diagnostics

- Build with the `tracing` feature (the perf script does this) to emit INFO-level JSON spans.
- Key spans now covered in both paths:
  - `find_chess_corners` (total)
  - `single_scale` (single-path body)
  - `coarse_detect` (coarse response + detection)
  - `refine` (per-seed refinement, includes `seeds` count)
  - `merge` (duplicate suppression)
  - `build_pyramid` (pyramid construction)
  - `ml_refiner` (ML refinement span + timing event when enabled)
- Parsing: `tools/trace/parser.py` extracts the spans above; `perf_report.json` is produced by `tools/perf_bench.py`, and accuracy overlays/timings come from `tools/accuracy_bench.py`.
- Use `--json-trace` on the CLI (or run via `perf_bench.py`) to capture traces; visualize or aggregate with your preferred JSON tools.

## 5.5 Integration patterns

- **Real-time loops**: reuse `PyramidBuffers` via `find_chess_corners_buff`; prefer multiscale + simd, add rayon for larger frames. Keep `merge_radius` modest (2–3 px) to avoid duplicate corners without throwing away tight clusters.
- **Calibration/pose**: the accuracy report shows sub-pixel consistency; feed detected corners directly into calibration routines. Use the accuracy histograms to validate new camera data.
- **Diagnostics in the field**: capture a short trace with `--json-trace` and inspect `refine_seeds` and span timings; spikes usually indicate harder scenes (more seeds) or contention (misconfigured features).
- **Reproducibility**: keep generated reports under `testdata/out/`; rerun `accuracy_bench.py --batch` and `perf_bench.py` after algorithm or config changes, and drop updated plots into docs for before/after comparisons.
