# Refiner comparison — accuracy and throughput

This document reports the results of the unified cross-refiner
benchmark at
[`crates/chess-corners/tests/refiner_benchmark.rs`](../../crates/chess-corners/tests/refiner_benchmark.rs).
It measures the four refiners shipped by this workspace on a common
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

| condition        | CenterOfMass       | Forstner             | SaddlePoint          | ML (ONNX v4)        |
|------------------|--------------------|----------------------|----------------------|---------------------|
| clean (cell=8)   | 0.080 / 0.123      | 0.061 / 0.165        | **0.045** / 0.059    | 0.094 / 0.181       |
| clean (cell=5)   | 0.390 / 0.707      | 0.061 / 0.165        | **0.045** / 0.059    | 0.091 / 0.150       |
| blur σ=1.5       | 0.056 / 0.124      | 0.266 / 0.471        | **0.044** / 0.065    | 0.092 / 0.173       |
| noise σ=5        | 0.078 / 0.164      | 0.113 / 0.238        | **0.058** / 0.126    | 0.091 / 0.175       |
| noise σ=10       | **0.091** / 0.178  | 0.159 / 0.344        | **0.091** / 0.179    | 0.094 / 0.167       |

Cells are `mean / worst` px; accept rate is 36/36 for every refiner in
every condition. In this fixture, `SaddlePoint` has the lowest mean
error on every row, tied with `CenterOfMass` at noise σ=10. Its
remaining ~0.045 px on the clean rows is the fixture's own floor:
8× supersampling quantizes the rendered edge positions to 1/8 px, and
on an exact pixel-area rendering the same refiner is within 0.001 px (see
[`refiner-accuracy.md`](refiner-accuracy.md)).

The ML refiner remains useful as a
single deployable ONNX artifact for callers who prefer a learned
component over a hand-tuned pipeline. See Part V of the book for the
ML refiner architecture.

The `Forstner / SaddlePoint` rows are identical between
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
| CenterOfMass   | 0.016 µs  | ≈60 M corners/s | 5×5 weighted moment over the ChESS response map                               |
| Forstner       | 0.06 µs   | ≈17 M corners/s | 5×5 gradient structure tensor solve                                           |
| SaddlePoint    | 0.18 µs   | ≈5 M corners/s  | 5×5 quadratic LS on 4–6 integer-centred patches, re-centred by bilinear blend |
| ML (ONNX)      | 151 µs    | ≈7 K corners/s  | 21×21 patch extract + ONNX inference, batch=1                                 |

## Per-refiner takeaways

- **SaddlePoint** — 0.18 µs in this harness and the lowest mean on
  every row (0.045 px clean, 0.044 px at blur σ=1.5). It refits on a
  patch re-centred at its own estimate, so it carries no pull toward the
  seed pixel. The default choice for calibration pipelines.
- **CenterOfMass** — the fastest, but **only when the ChESS ring
  matches the cell size**. At cell=5 with the default radius-5 ring,
  the ring crosses into neighbouring cells and the response centroid
  is dragged off the true corner (mean 0.39 px vs 0.08 px at cell=8).
  Caveat: the benchmark uses the *default* CoM config (radius 2 on
  the response map) — tuning it for small cells will help, but the
  sensitivity is real.
- **Forstner** — best mean on clean inputs (0.061 px at cell=8), but
  degrades sharply under blur (mean 0.27 px at σ=1.5) because Gaussian
  smoothing collapses the gradient magnitudes its structure tensor
  depends on. Consider it when the input edges are sharp and high
  contrast.
- **ML (ONNX) v4** — the shipped mixed-fixture model. Reaches
  0.09–0.12 px mean across the rows above, with the flattest response to
  noise.
  Retrained on AA-hard-cell synthetic data matching this fixture
  (`tools/ml_refiner/configs/synth_v6.yaml`), replacing the v2
  tanh-saddle training that produced the ~0.5 px
  distribution-mismatch failure. Use it when a deployable learned component is a better
  integration fit than a CPU-only geometric refiner; note it is ~600×
  slower than SaddlePoint per corner at batch=1.

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
