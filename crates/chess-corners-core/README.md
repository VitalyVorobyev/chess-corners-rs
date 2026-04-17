# chess-corners-core

Core primitives for computing ChESS responses and extracting subpixel chessboard corners.

This crate implements:

- 16-sample ChESS rings (`ring` module) at radii 5 and 10.
- Dense response computation on 8-bit grayscale images
  (`response` module). The score is the raw, unnormalized
  `R = SR − DR − 16·MR` from the paper — `R > 0` is the default
  corner-acceptance criterion; `threshold_rel` / `threshold_abs` are
  opt-in adaptive policies layered on top.
- Thresholding, non-maximum suppression, and pluggable subpixel
  refinement (`detect` + `refine` modules).
- Rich corner descriptors (`descriptor` module) built from a
  4-parameter Gauss–Newton fit of the two-axis tanh model
  `μ + A · tanh(β·sin(φ − θ₁)) · tanh(β·sin(φ − θ₂))` to the 16 ring
  samples. Each descriptor carries both local grid axes with
  per-axis 1σ angular uncertainty (from the Gauss–Newton covariance
  `σ̂² · (JᵀJ)⁻¹` with `σ̂² = SSR / 12`), the fitted bright/dark
  amplitude, and the RMS fit residual.

Feature flags:

- `std` *(default)* – use the Rust standard library; disabling this yields `no_std` + `alloc`.
- `rayon` – parallelize response computation over image rows.
- `simd` – enable portable-SIMD acceleration of the response kernel (nightly only).
- `tracing` – emit structured spans around response and detector code for profiling.

Basic usage:

```rust
use chess_corners_core::{detect::find_corners_u8, ChessParams, RefinerKind};

fn detect(img: &[u8], w: usize, h: usize) {
    let mut params = ChessParams::default();
    // Default = center-of-mass refinement on the response map.
    let corners = find_corners_u8(img, w, h, &params);
    println!("found {} corners", corners.len());

    // Opt into Förstner or saddle-point refiners on the image intensities:
    params.refiner = RefinerKind::Forstner(Default::default());
    let refined = find_corners_u8(img, w, h, &params);
    println!("found {} corners with Förstner", refined.len());
}
```

For a higher-level, image-friendly API (including multiscale detection and an optional CLI),
see the `chess-corners` crate in this workspace.
