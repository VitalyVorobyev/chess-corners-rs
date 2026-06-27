# Detector Comparison: ChESS vs Radon

This document explains the practical difference between the two detector
families exposed by `DetectorConfig`:

- `DetectionStrategy::Chess(ChessConfig)` computes the ChESS 16-sample
  ring response.
- `DetectionStrategy::Radon(RadonConfig)` computes a localized
  Duda-Frese-style ray-sum response from summed-area tables.

Both paths return `CornerDescriptor` values in input-image coordinates,
so downstream code can usually switch strategies without changing its
output handling.

## Short Guide

| Situation | Start with |
|-----------|------------|
| Normal high-contrast printed board | `DetectorConfig::chess_multiscale()` |
| Small cells near the ChESS ring support | `DetectorConfig::radon()` or `radon_multiscale()` |
| Synthetic blur + low-contrast fixture used in this repo | `DetectorConfig::radon()` |
| Large clean frame with tight latency budget | `DetectorConfig::chess_multiscale()` |
| Offline calibration where detector cost is less important | Compare both on the target images |

The important qualification is that these are starting points. The repo
has tests and benchmarks for specific synthetic fixtures; it does not
claim that one detector dominates on every camera, lens, board material,
or lighting setup.

## Why Both Exist

ChESS samples a fixed ring around every candidate pixel. It is cheap and
selective when the ring lands on a well-resolved chessboard corner. The
same fixed support can become a poor match when:

- the cell size is close to the ring diameter,
- blur flattens the alternating bright/dark pattern around the ring,
- low contrast pushes the response toward the noise floor.

Radon uses four ray sums through each candidate:

```text
R(x, y) = (max_alpha S_alpha - min_alpha S_alpha)^2
```

Those sums are computed from four summed-area tables, so the ray length
does not add a per-pixel loop. The tradeoff is memory and setup cost:
Radon builds SAT buffers and a dense response map before peak detection.

## Evidence In The Repository

The relevant tests are:

- [`crates/chess-corners-core/tests/radon_vs_chess.rs`](../../crates/chess-corners-core/tests/radon_vs_chess.rs)
  — compares raw ChESS/Radon response paths on a synthetic
  low-contrast, blurred board.
- [`crates/chess-corners/tests/radon_pipeline.rs`](../../crates/chess-corners/tests/radon_pipeline.rs)
  — verifies the public `Detector` facade routes the Radon strategy
  end-to-end and returns descriptors in base-image coordinates.

The hostile fixture is intentionally narrow:

```text
129x129 image, cell = 10 px,
contrast = 108..138,
Gaussian blur sigma = 2.5 px
```

The contract is relative: Radon must recover substantially more corners
than the default ChESS path on that fixture, and it must recover at
least 60% of the expected interior grid intersections. On a clean
high-contrast fixture, both paths must recover most corners.

For measured wall time and broader synthetic sweeps, see
[Part VIII of the book](../../book/src/part-08-benchmarks.md).

## Configuration

Switching strategies is explicit:

```rust
use chess_corners::{Detector, DetectorConfig};

let cfg = DetectorConfig::radon();
let mut detector = Detector::new(cfg)?;
let corners = detector.detect(&img)?;
```

`DetectorConfig::radon_multiscale()` combines the Radon response with
the same coarse-to-fine pyramid machinery used by the ChESS preset.
This can reduce cost on some larger frames because the detector can seed
at a coarser level and refine in the input image.

The two Radon fields most callers tune first are:

- `image_upsample`: `1` uses the input grid; `2` bilinearly upsamples
  before building SATs. Values above `2` are clamped by the core.
- `ray_radius`: half-length of each ray in working-resolution pixels.

Thresholds are detector-specific in scale. ChESS presets use an
absolute threshold by default; Radon presets use a relative threshold
because the squared ray-range response has a different magnitude.

## Core API

The low-level Radon API lives in `chess_corners_core::detect::radon`:

```rust
use chess_corners_core::{
    detect_peaks_from_radon, radon_response_u8,
    RadonBuffers, RadonDetectorParams,
};

let mut buffers = RadonBuffers::new();
let params = RadonDetectorParams::default();
let response = radon_response_u8(&img_u8, width, height, &params, &mut buffers);
let peaks = detect_peaks_from_radon(&response, &params);
```

Most applications should use the facade `Detector` API instead. The
core functions are useful when you need response maps, custom filtering,
or test fixtures around the detector internals.
