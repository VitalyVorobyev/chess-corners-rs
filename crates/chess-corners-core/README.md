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
- **Subpixel localization** — each detector's own localizer (the ChESS
  center of mass on the response map, configured by
  `ChessParams::center_of_mass`; the Radon peak fit inside peak
  detection) and the saddle-point image refiner for either detector
  (`refine_corners_on_image`, `Refiner`, `RefinerKind`).
- **Orientation** — two-axis orientation fit at each detected corner:
  ring-fit (parametric tanh model with robust seeding) and
  disk-sector full-disk crossing-line estimator (`describe_corners`,
  `OrientationMethod`).

Rich corner descriptors carry both local grid axes, each with
`fit_std_error`: the residual-based least-squares standard error of the
axis fit, `√([(JᵀJ)⁻¹]θθ · SSR / (N − 4))` over the fit's samples, in
radians (a fit-quality score, not a calibrated 1σ). `fit_axes_at_point`
also returns the fitted bright/dark amplitude and the RMS fit residual.

Feature flags:

- `std` *(default)* – compatibility feature reserved for future use.
  The current detector implementation requires the Rust standard
  library.
- `rayon` – parallelize response computation over image rows and the
  refinement of large corner sets.
- `simd` – portable-SIMD acceleration of the response kernel
  (nightly only).
- `tracing` – emit structured spans around response and detector code
  for profiling.

The default (stable) build requires Rust **1.91** or newer
(`rust-version` in `Cargo.toml`). The `simd` feature uses
`portable_simd` and needs a nightly toolchain; every other feature
builds on stable.

Basic usage:

```rust
use chess_corners_core::{
    chess_response_u8, detect_corners_from_response_with_refiner, find_corners_u8, ChessParams,
    ImageView, Refiner, RefinerKind,
};

fn detect(img: &[u8], w: usize, h: usize) {
    let params = ChessParams::default();
    // ChESS peaks placed by the center of mass of the response map.
    let corners = find_corners_u8(img, w, h, &params);
    println!("found {} corners", corners.len());

    // Refine the peaks on the image intensities with the saddle-point
    // image refiner instead:
    let resp = chess_response_u8(img, w, h, &params);
    let view = ImageView::from_u8_slice(w, h, img).unwrap();
    let mut saddle = Refiner::from_kind(RefinerKind::SaddlePoint(Default::default()));
    let refined = detect_corners_from_response_with_refiner(&resp, &params, Some(view), &mut saddle);
    println!("found {} corners with SaddlePoint", refined.len());
}
```

For a higher-level, image-friendly API (including multiscale detection
through a `Detector` struct, an optional CLI, and bindings to Python
and WebAssembly), see the `chess-corners` crate in this workspace.
