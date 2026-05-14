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
every condition. In this fixture, `RadonPeak` has the lowest mean error
on the clean and blurred rows, and the ML refiner has the lowest mean
error on the heaviest noise row (`σ=10`). All five refiners come in
under the ~0.13 px mean-error bar on clean cell=8. The ML model (v4
ONNX) is trained on a mixed tanh + AA-hard-cell distribution, which
avoids the 0.5 px hard-cell mismatch seen with the older v2 fixture.

**Why ML trails RadonPeak on clean data.** We tried three
architectures (~180K → 730K → 50K-param soft-argmax) on the same
synthetic data; all converged to the same ~0.14 px plateau on the
held-out hard-cell val set, ~0.09 px on the Rust benchmark. The
evidence points to the training setup rather than ONNX export or patch
normalization: RadonPeak encodes a specific geometric prior (4-angle
Radon response, Gaussian log peak fit) as closed-form operations, while
the ML refiner has to learn an equivalent mapping from data. The current
training regime did not close that gap.

The ML refiner remains useful in the measured heavy-noise row and as a
single deployable ONNX artifact for callers who prefer a learned
component over a hand-tuned pipeline. See
`docs/proposal-ml-refiner-v3.md` for the architecture exploration and
the accounting of what was tried.

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

- **RadonPeak** — lowest mean error in the clean and blurred rows,
  ~300× slower than SaddlePoint but still sub-20 µs per corner in this
  harness. Its clean-cell floor (`mean < 0.05` px on clean cell=8) is
  asserted in the benchmark to guard against regression.
- **SaddlePoint** — 0.12 µs in this harness, under 0.12 px mean on the
  clean row, and below 0.13 px mean in every row above. It is a good
  default when you want image-patch refinement without RadonPeak's cost.
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
  Consider it when the input edges are sharp and high contrast.
- **ML (ONNX) v4** — the shipped mixed-fixture model. Reaches
  0.09–0.12 px mean across the rows above and has the lowest mean error
  under heavy noise (`σ=10`) in this benchmark.
  Retrained on AA-hard-cell synthetic data matching this fixture
  (`tools/ml_refiner/configs/synth_v6.yaml`), replacing the v2
  tanh-saddle training that produced the ~0.5 px
  distribution-mismatch failure. Still ~15× slower than `RadonPeak`
  per corner in the batch=1 harness; use it when the measured noise
  behavior matters or when a deployable learned component is a better
  integration fit than a CPU-only geometric refiner.

## Notes on the measurement

- **Timing** — 400 deterministic refinements per subpixel offset per
  refiner, averaged. Warm-up call before the timed loop.
- **Accuracy** — one sample per offset (refiners are deterministic),
  36 offsets per condition.
- **ML batching** — ONNX inference runs at batch=1 here. The detector
  facade batches candidates, so measure end-to-end cost on your target
  workload rather than extrapolating directly from this row.
- **Noise/blur recipe** — blur is a separable Gaussian; noise is
  additive PCG+Box-Muller Gaussian, seeded deterministically.

## Follow-up ideas

- Fine-tune the ML refiner on additional AA-rendered chessboard data
  and re-run the comparison; the older v2 0.5 px hard-cell result was
  a distribution-mismatch artifact, not evidence from the v4 model.
- Wire `tools/synthimages.py` output (full-image OpenCV renders with
  pose, blur, noise, vignetting, JSON ground truth) into a Rust
  integration benchmark to validate the comparison on realistic
  camera-like imagery in addition to the synthetic fixture.
- Sweep the RadonPeak `image_upsample` knob (1, 2, 4) against
  accuracy and runtime; the current default (2) is the paper's
  recommendation but a chart would make the tradeoff concrete.
