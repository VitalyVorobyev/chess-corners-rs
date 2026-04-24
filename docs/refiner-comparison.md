# Refiner comparison — accuracy and throughput

This document reports the results of the unified cross-refiner
benchmark at
[`crates/chess-corners/tests/refiner_benchmark.rs`](../crates/chess-corners/tests/refiner_benchmark.rs).
It measures the five refiners shipped by this workspace on a common
synthetic fixture and reports both subpixel accuracy and per-corner
throughput so they can be compared at a glance.

## Reproducing

```sh
cargo test --release -p chess-corners --test refiner_benchmark \
    --features ml-refiner -- --nocapture --test-threads=1
```

Without `--features ml-refiner`, the ML row is omitted and the rest of
the table still prints.

## Fixture

- Anti-aliased chessboard, rendered at 8× supersampling then box-
  averaged to preserve subpixel corner positions.
- Image size 45×45, corner near the center, 6×6 = 36 subpixel
  offsets on an 8×8 grid inside a cell.
- Blur and additive Gaussian noise are layered on top for robustness
  sweeps. Photometric jitter is held fixed (`dark=30, bright=230`).

## Accuracy

Euclidean pixel error against the ground-truth subpixel corner. All
36 offsets accepted by every refiner in every condition. The best
mean per condition is **bolded**.

| condition        | CenterOfMass       | Forstner         | SaddlePoint      | RadonPeak              | ML (ONNX v4)        |
|------------------|--------------------|------------------|------------------|------------------------|---------------------|
| clean (cell=8)   | 0.080 / 0.123      | 0.061 / 0.165    | 0.114 / 0.177    | **0.049** / 0.103      | 0.094 / 0.181       |
| clean (cell=5)   | 0.390 / 0.707      | 0.061 / 0.165    | 0.114 / 0.177    | **0.049** / 0.103      | 0.091 / 0.150       |
| blur σ=1.5       | 0.056 / 0.124      | 0.266 / 0.471    | 0.047 / 0.079    | **0.046** / 0.073      | 0.092 / 0.173       |
| noise σ=5        | 0.088 / 0.156      | 0.135 / 0.301    | 0.095 / 0.220    | **0.085** / 0.183      | 0.093 / 0.168       |
| noise σ=10       | 0.123 / 0.272      | 0.201 / 0.474    | 0.126 / 0.257    | 0.128 / 0.302          | **0.101** / 0.200   |

Cells are `mean / worst` px; accept rate is 36/36 for every refiner in
every condition. `RadonPeak` wins mean-error on clean and blurred
inputs; the ML refiner wins under heavy noise (σ=10). All five
refiners come in under the ~0.13 px mean-error bar on clean cell=8.
The ML model (v4 ONNX) is trained on a mixed tanh + AA-hard-cell
distribution so it handles both regimes without the 0.5 px
distribution-mismatch failure that bit v2 on this fixture.

**Why ML loses to RadonPeak on clean data.** We tried three
architectures (~180K → 730K → 50K-param soft-argmax) on the same
synthetic data; all converged to the same ~0.14 px plateau on the
held-out hard-cell val set, ~0.09 px on the Rust benchmark. The gap
to RadonPeak is a **learning gap, not an information gap** — a CNN
is mathematically capable of computing any deterministic function
of the 21 × 21 patch, including RadonPeak's own algorithm, but
discovering that structure from generic smooth-L1 / MSE regression
on 200K patches has not been enough to match a hand-designed
algorithm built specifically for the task. RadonPeak encodes a
strong geometric prior (4-angle Radon response, Gaussian log peak
fit) as closed-form operations; the ML refiner has to learn the
equivalent structure through gradient descent, and current
training regime does not get there.

The ML refiner remains useful where classical refiners struggle
(heavy noise, see σ=10 row) and as a single deployable ONNX
artifact for callers who prefer a learned component over a
hand-tuned pipeline. See `docs/proposal-ml-refiner-v3.md` for the
architecture exploration and the honest accounting of what we
tried.

The `Forstner / SaddlePoint / RadonPeak` rows are identical between
`clean (cell=5)` and `clean (cell=8)` because they're all local
refiners that only examine a 2–3 px neighbourhood of the corner. The
anti-aliased corner looks the same locally at both cell sizes, so the
numbers match — that's expected and actually validates that the
harness is measuring what it should.

## Throughput

Release build, per refinement on the 45×45 fixture. Warm-up call
precedes the timed loop.

| refiner        | time/call | throughput      | what it costs                                                                 |
|----------------|-----------|-----------------|-------------------------------------------------------------------------------|
| CenterOfMass   | 0.02 µs   | ≈50 M corners/s | 5×5 weighted moment over the ChESS response map                               |
| Forstner       | 0.06 µs   | ≈17 M corners/s | 5×5 gradient structure tensor solve                                           |
| SaddlePoint    | 0.12 µs   | ≈8 M corners/s  | 6-param quadratic LS via Gauss-Jordan                                         |
| **RadonPeak**  | 17.3 µs   | ≈58 K corners/s | 13×13 dense Radon (169 samples × 4 rays × 9 bilinear taps) + box blur + fit   |
| ML (ONNX)      | 252 µs    | ≈4 K corners/s  | 21×21 patch extract + ONNX inference, batch=1                                 |

## Per-refiner takeaways

- **RadonPeak** — most accurate, ~300× slower than SaddlePoint but
  still sub-20 µs per corner. Comfortable for calibration-rate
  workloads (thousands of corners per image). This is the refiner we
  actively tuned against the paper in this round, and its accuracy
  floor (`mean < 0.05` px on clean cell=8) is asserted in the benchmark
  to guard against regression.
- **SaddlePoint** — the accuracy/speed sweet spot. 0.12 µs, under
  0.12 px mean on clean input, stable across all conditions. This is
  the right default unless the extra RadonPeak accuracy matters or the
  workload is ~million-corners-per-second throughput.
- **CenterOfMass** — the fastest, but **only when the ChESS ring
  matches the cell size**. At cell=5 with the default radius-5 ring,
  the ring crosses into neighbouring cells and the response centroid
  is dragged off the true corner (mean 0.39 px vs 0.08 px at cell=8).
  Caveat: the benchmark uses the *default* CoM config (radius 2 on
  the response map) — tuning it for small cells will help, but the
  sensitivity is real.
- **Forstner** — middle of the pack on clean inputs, but degrades
  sharply under blur (mean 0.27 px at σ=1.5) because Gaussian smoothing
  collapses the gradient magnitudes its structure tensor depends on.
  Good pick only on sharp, high-contrast imagery.
- **ML (ONNX) v3** — the retrained model. Reaches 0.08–0.10 px
  across all conditions and **wins under heavy noise (σ=10)**.
  Retrained on AA-hard-cell synthetic data matching this fixture
  (`tools/ml_refiner/configs/synth_v5.yaml`), replacing the v2
  tanh-saddle training that produced the ~0.5 px
  distribution-mismatch failure. Still ~15× slower than `RadonPeak`
  per corner in the batch=1 harness; use when you want one
  refiner that stays under 0.1 px across the full noise/blur
  envelope, or for GPU/NPU-heavy deployments where ONNX
  inference can batch better than the CPU-bound hand-rolled
  refiners.

## Notes on the measurement

- **Timing** — 400 deterministic refinements per subpixel offset per
  refiner, averaged. Warm-up call before the timed loop.
- **Accuracy** — one sample per offset (refiners are deterministic),
  36 offsets per condition.
- **ML batching** — ONNX inference runs at batch=1 here. The
  production facade batches candidates, so a real pipeline will see
  2–5× better amortised cost per corner for the ML refiner.
- **Noise/blur recipe** — blur is a separable Gaussian; noise is
  additive PCG+Box-Muller Gaussian, seeded deterministically.

## Follow-up ideas

- Retrain or fine-tune the ML refiner on the AA-rendered chessboard
  distribution and re-run the comparison; the current 0.5 px result
  is a distribution-mismatch artefact, not a capacity limit.
- Wire `tools/synthimages.py` output (full-image OpenCV renders with
  pose, blur, noise, vignetting, JSON ground truth) into a Rust
  integration benchmark to validate the comparison on realistic
  camera-like imagery in addition to the synthetic fixture.
- Sweep the RadonPeak `image_upsample` knob (1, 2, 4) against
  accuracy and runtime; the current default (2) is the paper's
  recommendation but a chart would make the tradeoff concrete.
