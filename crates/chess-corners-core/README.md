# chess-corners-core

Core primitives for computing ChESS / Radon responses and extracting
subpixel chessboard corners.

The crate is organised along the three orthogonal axes the detector
pipeline composes:

- **Detection** (`detect` module) — two independent feature-detector
  families share a common output type:
  - [`detect::chess`] — ChESS response (16-sample ring) at radii 5 and 10,
    NMS, cluster filtering.
  - [`detect::radon`] — whole-image Duda-Frese localized Radon detector
    with summed-area-table ray sums.
- **Refinement** (`refine` module) — pluggable subpixel-refinement
  backends: center-of-mass, Förstner, saddle-point, and Radon-peak.
- **Orientation** (`orientation` module) — two-axis orientation fit at
  each detected corner: ring-fit (parametric tanh model with robust
  seeding and σ-LUT) and disk-sector full-disk crossing-line
  estimator.

Rich corner descriptors carry both local grid axes with per-axis 1σ
angular uncertainty (from the Gauss–Newton covariance
`σ̂² · (JᵀJ)⁻¹` with `σ̂² = SSR / 12`), the fitted bright/dark
amplitude, and the RMS fit residual.

Feature flags:

- `std` *(default)* – use the Rust standard library; disabling this
  yields `no_std` + `alloc`.
- `rayon` – parallelize response computation over image rows.
- `simd` – portable-SIMD acceleration of the response kernel
  (nightly only).
- `tracing` – emit structured spans around response and detector code
  for profiling.

Basic usage:

```rust
use chess_corners_core::{
    detect::find_corners_u8, ChessParams, RefinerKind,
};

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

For a higher-level, image-friendly API (including multiscale detection
through a `Detector` struct, an optional CLI, and bindings to Python
and WebAssembly), see the `chess-corners` crate in this workspace.
