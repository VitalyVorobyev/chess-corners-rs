# Design Summary

## Crate Layering

```
chess-corners-py    (PyO3 Python bindings, module: chess_corners)
       |
chess-corners       (High-level facade, multiscale pipeline, CLI)
       |
chess-corners-core  (Low-level: response, detection, refinement)

chess-corners-ml    (ONNX ML refiner, optional via ml-refiner feature)

box-image-pyramid   (Standalone u8 pyramid, 2x box-filter downsample)
```

**Dependency rule:** `chess-corners-core` must never depend on `chess-corners`.
`box-image-pyramid` is fully independent with zero chess-specific coupling.

## Core Algorithm Pipeline

1. **Response computation** (`core/response.rs`) -- Dense ChESS response using 16-sample rings at radius 5 (or 10 for heavy blur). Formula: `R = SR - DR - 16 * |mean_ring - mean_cross|`.
2. **Detection** (`core/detect.rs`) -- Relative/absolute thresholding, non-maximum suppression (NMS), minimum cluster size filtering.
3. **Refinement** (`core/refine.rs`) -- Pluggable via `CornerRefiner` trait with three built-in implementations: CenterOfMass, Forstner, SaddlePoint. Each returns refined xy + score + status.
4. **Descriptor generation** (`core/descriptor.rs`) -- Converts raw corners to `CornerDescriptor` with subpixel position, response strength, and orientation estimated from ring samples.

## Multiscale Architecture

When `pyramid_levels > 1`, the detector follows a coarse-to-fine strategy:

1. Build image pyramid via `box-image-pyramid` (2x box-filter downsample at each level)
2. Detect corners on the smallest (coarsest) level
3. Project each coarse detection to a region-of-interest (ROI) in the base image
4. Run response computation and refinement within each ROI at full resolution
5. Merge near-duplicate corners within `merge_radius`
6. Generate descriptors at base resolution

`PyramidBuffers` allows reusing allocated memory across successive frames.

## Feature Gating Strategy

Performance choices are compile-time features. Behavioral choices are runtime configuration.

| Feature | Type | Effect |
|---------|------|--------|
| `rayon` | Compile-time | Parallel response + refinement across rows |
| `simd` | Compile-time | Portable SIMD inner loops (nightly only) |
| `par_pyramid` | Compile-time | SIMD/rayon in pyramid downsampling |
| `image` | Compile-time | `image::GrayImage` integration |
| `ml-refiner` | Compile-time | ONNX ML refinement via `chess-corners-ml` |
| `tracing` | Compile-time | Structured diagnostic spans |
| `threshold_mode` | Runtime | Relative vs absolute threshold interpretation |
| `threshold_value` | Runtime | Threshold value used by the selected mode |
| `detector_mode` | Runtime | Semantic detector mode: canonical or broad response sampling |
| `descriptor_mode` | Runtime | Descriptor/orientation sampling: follow detector, canonical, or broad |
| `nms_radius` | Runtime | Non-maximum suppression radius |
| `refiner.kind` | Runtime | Which subpixel refiner is active |
| `pyramid_levels` | Runtime | Number of pyramid levels |

All feature combinations produce numerically identical results (outputs sorted by stable keys when using rayon).

## Refinement Trait Design

```rust
pub trait CornerRefiner {
    fn radius(&self) -> i32;
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}
```

- **Mutable self** allows internal scratch buffers without per-call allocation
- **`RefineStatus` enum** (Accepted/Rejected/OutOfBounds/IllConditioned) enables post-hoc filtering
- **`RefinerKind` enum** dispatches statically via match (no vtable overhead)
- Configuration structs (`CenterOfMassConfig`, `ForstnerConfig`, `SaddlePointConfig`) are cheap to clone

## ML Integration

- Separate `chess-corners-ml` crate wrapping `tract-onnx`
- ONNX model embedded in binary by default (`embed-model` feature)
- Extracted to temp file on first use via `OnceLock`
- Input: normalized u8 patches (/ 255.0) around each candidate
- Output: `[dx, dy, conf_logit]` -- confidence currently unused
- Slower than classical refiners (~23 ms vs ~0.6 ms) but more accurate on synthetic data

## Python Bindings

- Mixed Rust/Python package built with `maturin`
- Public package: pure-Python `chess_corners`
- Private extension module: `chess_corners._native`
- Input: 2D `uint8` NumPy array (must be C-contiguous)
- Output: `(N, 4)` float32 array `[x, y, response, orientation]`
- Public config is Python-native and uses the same flat schema as Rust and the CLI
