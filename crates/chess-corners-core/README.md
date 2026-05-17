# chess-corners-core

Core primitives for computing ChESS / Radon responses and extracting
subpixel chessboard corners.

The crate exposes three orthogonal components via its crate root:

- **Detection** — two independent feature-detector families share a
  common output type:
  - ChESS response (16-sample ring) at radii 5 and 10, NMS, cluster
    filtering (`chess_response_u8`, `find_corners_u8`, and friends).
  - Whole-image Duda-Frese localized Radon detector with
    summed-area-table ray sums (`radon_response_u8`,
    `detect_peaks_from_radon`, and friends).
- **Refinement** — pluggable subpixel-refinement backends:
  center-of-mass, Förstner, saddle-point, and Radon-peak
  (`refine_corners_on_image`, `Refiner`, `RefinerKind`).
- **Orientation** — two-axis orientation fit at each detected corner:
  ring-fit (parametric tanh model with robust seeding and σ-LUT) and
  disk-sector full-disk crossing-line estimator (`describe_corners`,
  `OrientationMethod`).

Rich corner descriptors carry both local grid axes with per-axis 1σ
angular uncertainty (from the Gauss–Newton covariance
`σ̂² · (JᵀJ)⁻¹` with `σ̂² = SSR / 12`), the fitted bright/dark
amplitude, and the RMS fit residual.

Feature flags:

- `std` *(default)* – compatibility feature reserved for future use.
  The current detector implementation requires the Rust standard
  library.
- `rayon` – parallelize response computation over image rows.
- `simd` – portable-SIMD acceleration of the response kernel
  (nightly only).
- `tracing` – emit structured spans around response and detector code
  for profiling.

Basic usage:

```rust
use chess_corners_core::{
    find_corners_u8, ChessParams, RefinerKind,
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
