# Detector comparison — ChESS ring vs whole-image Radon

This document explains when to pick `DetectorMode::Canonical`
(equivalently `Broad`) — the classic ChESS ring kernel — versus
`DetectorMode::Radon` — the Duda-Frese whole-image localized Radon
detector added in M2. It is the companion to
[`refiner-comparison.md`](refiner-comparison.md), which covers the
subpixel refinement stage.

The two detectors consume the same `ChessConfig` surface. Switching
between them is a single field change:

```rust
let mut cfg = ChessConfig::default();
cfg.detector_mode = DetectorMode::Radon;     // or ::Canonical, ::Broad
```

Both return corners in base-image coordinates and both emit the same
`CornerDescriptor` struct, so callers downstream (calibration,
tracking, UI overlays) do not care which path produced the corners.

## TL;DR

| Scenario                                               | Pick         |
|--------------------------------------------------------|--------------|
| High-contrast calibration board, well-focused          | **Canonical** |
| You need to recover corners at cell ≲ 2·ring_radius    | **Radon**    |
| Heavy motion blur (camera shake, rolling shutter)      | **Radon**    |
| Low-contrast targets (printed in grey, screen glare)   | **Radon**    |
| HD / 4K frame, real-time budget                        | **Canonical** |
| Offline, accuracy over latency                         | **Radon** (with `image_upsample=2`) |

ChESS stays the default. Radon earns its keep when ChESS's 16-sample
ring kernel fails to pick up a usable signal.

## Why the two paths coexist

The ChESS kernel computes, at every pixel, a single statistic built
from the difference between "same-side" and "neighbouring" ring
samples. On a clean corner at the right scale it gives a razor-sharp
local maximum. But the recipe is brittle:

- **Blur.** Gaussian smoothing spreads the bimodal intensity
  signature around the ring into near-uniform grey, and the SR − DR
  term collapses into the noise floor.
- **Low contrast.** The raw magnitude of `SR − DR − 16·|μₙ − μₗ|`
  scales with intensity difference across the ring. A 30-grey-level
  target (e.g. 108..138) yields responses an order of magnitude
  weaker than a 200-level target.
- **Small cells.** At `cell ≲ 2·ring_radius` the fixed-radius
  sampling ring crosses into the neighbouring cells; the ring is no
  longer sampling a single corner.

Radon sidesteps all three problems: its response is
`R(x, y) = (maxₐ Sₐ − minₐ Sₐ)²` where each `Sₐ` is a ray integral
across 4 angles. Blur distributes intensity but does not cancel the
max/min contrast across perpendicular rays. Low contrast scales the
response linearly but preserves peak location. And rays are
parameterised by `ray_radius` in working-resolution pixels
independently of the board cell.

## Accuracy

The facade-level integration test
[`crates/chess-corners/tests/radon_pipeline.rs`](../crates/chess-corners/tests/radon_pipeline.rs)
exercises both modes end-to-end through `find_chess_corners`:

| Fixture (129×129, cell=10)              | ChESS default | Radon (`image_upsample=2`) |
|-----------------------------------------|---------------|----------------------------|
| Clean (contrast 200, no blur)           | ≳80 % recall  | ≳80 % recall               |
| Hostile (σ=2.5 blur, contrast 30)       | ~0–40 corners | **Recovers the board**    |

On the clean board, both modes produce corners with subpixel
accuracy (the pipeline test locks ≤ 0.2 px mean residual to the
nearest grid intersection). The hostile fixture is where Radon
earns its keep: with the parameters above ChESS produces only a
handful of corners while Radon recovers the full grid minus the
image-border band — see the test assertion:

```rust
assert!(
    radon_corners.len() > chess_corners.len() + 8,
    "Radon must beat ChESS+8 on hostile fixture"
);
```

For the per-refiner subpixel accuracy once corners are found, see
[`refiner-comparison.md`](refiner-comparison.md). The detector and
refiner choices are independent — in Radon mode the detector's
3-point Gaussian peak fit supplies the subpixel refinement directly,
so `RefinerConfig::kind` is not consulted.

## Throughput

Measured by the Criterion bench at
[`crates/chess-corners-core/benches/radon_response.rs`](../crates/chess-corners-core/benches/radon_response.rs)
on a single core. Indicative numbers on a MacBook Pro M-series (your
mileage will vary, but relative scale is stable):

| Image size  | ChESS response | Radon `up=1` | Radon `up=2` |
|-------------|----------------|--------------|--------------|
| 640 × 480   | ~0.5 ms        | 2.8 ms       | 14 ms        |
| 1280 × 720  | ~1.5 ms        | 8.7 ms       | 48 ms        |
| 1920 × 1080 | ~3.5 ms        | 19 ms        | ~110 ms      |

ChESS is the latency winner by a factor of 5–15×. Radon at `up=1` is
comfortable for calibration-rate (1–10 Hz) work on HD; at `up=2` it
is offline territory on full-HD frames.

Reproduce:

```sh
cargo bench --bench radon_response \
    -- --warm-up-time 1 --measurement-time 3
```

## Picking parameters

The two knobs that matter for the Radon detector are:

- **`image_upsample`**: `1` operates on the input pixel grid; `2`
  (paper default, recommended) bilinearly upsamples first and
  doubles response resolution before peak fit. Values `≥ 3` are
  clamped to `2` (see
  [`chess_corners_core::MAX_IMAGE_UPSAMPLE`](../crates/chess-corners-core/src/radon_detector.rs)).
- **`ray_radius`**: half-length of each ray in working-resolution
  pixels. At `image_upsample=2` the paper default is 4 (2 physical
  pixels on each side of the centre).

The detector is otherwise self-tuning: NMS radius 4 and
`min_cluster_size=2` filter out isolated noise peaks; a 1 %
relative threshold rejects the intensity-scaled non-corner
background. You can tighten `nms_radius` or raise `threshold_abs`
if you need stricter filtering (e.g. dense corners in a small
image), but the defaults work on everything the test suite throws
at them.

## Pipeline details

On a `DetectorMode::Radon` call, the facade pipeline runs:

```text
u8 image ──► [optional 2× bilinear upsample]
                      │
                      ▼
       [4 summed-area tables: row / col / ±diag]
                      │
                      ▼
       [Radon response R(x, y) = (maxₐ Sₐ − minₐ Sₐ)²]
                      │
                      ▼
       [box blur, threshold, NMS, min-cluster]
                      │
                      ▼
       [3-point Gaussian peak fit → subpixel]
                      │
                      ▼
       CornerDescriptor in input-pixel coordinates
```

Multiscale Radon (coarse Radon + fine Radon across a pyramid) is
explicitly out of scope for M2 — at single scale the detector already
recovers corners across a 5–40 px cell range, and the SAT-based O(1)
ray sum makes the dense response cheap enough at base resolution.
Callers who want a pyramid today can still run the classic coarse-
to-fine path and flip to Radon on the base level by calling
`detect_corners_from_radon` directly.

## See also

- [`proposal-radon-detector.md`](proposal-radon-detector.md) — design
  notes and roadmap (M1 through M3).
- [`refiner-comparison.md`](refiner-comparison.md) — per-refiner
  subpixel accuracy and throughput.
- [`crates/chess-corners/tests/radon_pipeline.rs`](../crates/chess-corners/tests/radon_pipeline.rs)
  — facade-level end-to-end test (the contract covered in this
  document).
