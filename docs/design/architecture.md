# Design Summary

This repository is split into small crates with explicit ownership:

```text
chess-corners-py       Python package, built on the Rust facade
chess-corners-wasm     WebAssembly package, built on the Rust facade
       │
       ▼
chess-corners          public Rust API, CLI, multiscale/upscale pipeline
       │
       ▼
chess-corners-core     detector kernels, refiners, descriptors

box-image-pyramid      standalone u8 2× pyramid builder
chess-corners-ml       optional ONNX refiner used by `ml-refiner`
```

The dependency direction is one-way. `chess-corners-core` does not
depend on `chess-corners`, and `box-image-pyramid` has no chess-specific
dependencies.

## Pipeline

The facade detector follows the same high-level shape for both detector
families:

1. Optionally upscale the input with `UpscaleConfig`.
2. Optionally build a 2× image pyramid with `MultiscaleConfig`.
3. Run either the ChESS ring response or the Radon ray-sum response.
4. Threshold, suppress non-maxima, and reject isolated peaks.
5. Refine each accepted seed with the active per-detector refiner.
6. Merge near duplicates in input-image coordinates.
7. Build `CornerDescriptor` values with a two-axis orientation fit.

The ChESS path computes the Bennett-Lasenby ring response
`R = SR - DR - 16 * |mean_ring - mean_cross|` at radius 5 or 10. The
Radon path computes `(max_alpha S_alpha - min_alpha S_alpha)^2` from
four summed-area-table ray sums.

## Public Configuration

`DetectorConfig` is the compatibility surface. Cross-cutting fields sit
at the top level:

- `strategy`: `DetectionStrategy::Chess(ChessConfig)` or
  `DetectionStrategy::Radon(RadonConfig)`.
- `threshold`: `Threshold::Absolute(value)` or
  `Threshold::Relative(fraction)`.
- `multiscale`: `SingleScale` or `Pyramid { levels, min_size,
  refinement_radius }`.
- `upscale`: disabled or fixed integer pre-upscale.
- `orientation_method`: `RingFit` or `DiskFit`.
- `merge_radius`: duplicate-suppression distance in input pixels.

Detector-specific fields live inside the active strategy. This avoids
parallel knobs such as "detector mode" plus a stale strategy-specific
configuration block.

## Refiners

Refinement is pluggable through `CornerRefiner`:

```rust
pub trait CornerRefiner {
    fn radius(&self) -> i32;
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}
```

`ChessRefiner` carries `CenterOfMass`, `Förstner`, `SaddlePoint`, and
optionally `Ml`. `RadonRefiner` carries `RadonPeak` and `CenterOfMass`.
Each variant owns its tuning struct, so switching variants cannot leave
old tuning fields active by accident.

`RefineResult` reports the refined point, a refiner-specific score, and
`RefineStatus` (`Accepted`, `Rejected`, `OutOfBounds`, or
`IllConditioned`). Runtime refiners own their scratch buffers and reuse
them across seeds.

## Feature Flags

Compile-time features control implementation paths and optional
dependencies:

| Feature | Effect |
|---------|--------|
| `image` | `image::GrayImage` integration |
| `rayon` | parallel response/refinement work |
| `simd` | portable-SIMD inner loops where implemented |
| `par_pyramid` | SIMD/Rayon pyramid downsampling |
| `tracing` | structured diagnostic spans |
| `ml-refiner` | ONNX-backed ML refinement |
| `cli` | builds the CLI binary |
| `radon-sat-u32` | lower-memory Radon SATs with an input-size cap |

Feature combinations are expected to preserve numerical output. Parallel
paths sort final outputs by stable keys before returning.

## Bindings

The Python package exposes a Python-native `chess_corners.Detector` and
configuration classes that round-trip through the same JSON shape as the
Rust facade. `Detector.detect(image)` accepts a 2D C-contiguous `uint8`
NumPy array and returns a `float32 (N, 7)` array:

```text
x, y, response,
axis0_angle, axis0_sigma, axis1_angle, axis1_sigma
```

The WebAssembly package exposes the same detector/configuration concepts
for JavaScript and TypeScript, returning a `Float32Array` with the same
stride-7 corner layout.
