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
7. Build `CornerDescriptor` values, optionally with a two-axis
   orientation fit (skipped when orientation is disabled, leaving each
   descriptor's `axes` as `None`).

The ChESS path computes the Bennett-Lasenby ring response
`R = SR - DR - 16 * |mean_ring - mean_cross|` at radius 5 or 10. The
Radon path computes `(max_alpha S_alpha - min_alpha S_alpha)^2` from
four summed-area-table ray sums.

## Public Configuration

`DetectorConfig` is the compatibility surface. Cross-cutting fields sit
at the top level:

- `strategy`: `DetectionStrategy::Chess(ChessConfig)` or
  `DetectionStrategy::Radon(RadonConfig)`.
- `threshold`: a single `f32`. ChESS reads it as an absolute floor on
  the raw response (default `30`); Radon as a fraction in `[0, 1]` of
  the per-frame maximum (default `0.01`).
- `multiscale`: `SingleScale` or `Pyramid { levels, min_size,
  refinement_radius }`.
- `upscale`: disabled or fixed integer pre-upscale.
- `orientation_method`: `RingFit`, `DiskFit`, or disabled
  (`without_orientation()`), which skips the fit and leaves `axes` as
  `None`.
- `merge_radius`: duplicate-suppression distance in input pixels.
- `uncertainty`: `Off`, or `Propagated { noise }` with `noise` either
  `NoiseLevel::Estimate` (per-image estimate) or `NoiseLevel::Known(σ)`
  (gray levels); the noise level scales localizer covariances to px².
  Core refiners only take the on/off switch (`Refiner::with_covariance`);
  the facade lowers `DetectorConfig::uncertainty` in one place.

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

Each detector keeps the localizer that works on its own response map:
ChESS the center of mass (`ChessConfig::center_of_mass`, a
`CenterOfMassRefiner`), Radon the 3-point peak fit inside peak detection
(`RadonConfig::peak_fit`). One optional, detector-independent image
refiner — `DetectorConfig::refiner: Option<ImageRefiner>` with
`SaddlePoint` (core `RefinerKind`) or the facade's `Ml` — replaces the
ChESS center of mass or refines the Radon peak fit (keeping it where it
does not converge). The facade lowers it once (`core_image_refiner` in
`multiscale.rs`) into a crate-private `Localizer`; ML runs its own
traversal step generic over `DenseDetector`.

`RefineResult` reports the refined point, a refiner-specific score,
`RefineStatus` (`Accepted`, `Rejected`, `OutOfBounds`, or
`IllConditioned`), and `noise_covariance_per_unit_variance() ->
Option<PositionCovariance>` — the first-order position covariance per
unit image-noise variance (px² per gray-level²), set by every non-ML
refiner on accepted corners when it reports a covariance
(`Refiner::with_covariance(true)`, set under `Uncertainty::Propagated`). `Corner` carries it through
merging; the facade turns it into `CornerDescriptor::position_covariance`
in px² (`max(σₙ², 1/12)·C + f²I`, `f` = `POSITION_FLOOR_PX`) after `describe_corners`, with σₙ
from `noise_sigma` or `estimate_noise_sigma_u8` on the refinement image
(returned per frame by `Detector::last_noise_sigma`). The Radon peak fit's
own covariance is attached in core (`DenseDetector::attach_peak_covariance`)
to the peaks no image refiner moved.
Final refinement always runs on the detector's input image (pyramid seeds
are refined into the base level), so no rescaling is needed; on upscaled
input the noise is spatially correlated and no covariance is reported. Runtime refiners own their
scratch buffers and reuse them across seeds. The saddle-point refiner
refits on a patch re-centred at its previous estimate; since resampling
and the fit are linear, each refit is a bilinear blend of fits on
integer-centred patches, so it builds its fit operator once at
construction and per corner fits four to six integer patches.

## Feature Flags

Compile-time features control implementation paths and optional
dependencies:

| Feature | Effect |
|---------|--------|
| `image` | `image::GrayImage` integration |
| `rayon` | parallel response/refinement work |
| `simd` | optional high-performance path: portable `std::simd` inner loops (nightly only) |
| `par_pyramid` | SIMD/Rayon pyramid downsampling |
| `tracing` | structured diagnostic spans |
| `cli` | builds the CLI binary |
| `radon-sat-u32` | lower-memory Radon SATs with an input-size cap |

Feature combinations are expected to preserve numerical output. Parallel
paths sort final outputs by stable keys before returning.

## Bindings

The Python package exposes a Python-native `chess_corners.Detector` and
configuration classes that round-trip through the same JSON shape as the
Rust facade. `Detector.detect(image)` accepts a 2D C-contiguous `uint8`
NumPy array and returns a structure-of-arrays `Detections` object with
named fields — `.xy` (`(N, 2)`), `.response` (`(N,)`), `.angles`
(`(N, 2)`), `.fit_std_errors` (`(N, 2)`), `.position_covariance` (`(N, 2, 2)`,
`NaN` rows where absent) — where `.angles` / `.fit_std_errors` are `None`
when orientation is disabled (`without_orientation()`).

The WebAssembly package exposes the same detector/configuration concepts
for JavaScript and TypeScript, returning a flat stride-7 `Float32Array`
(`x, y, response, axis0_angle, axis0_fitStdError, axis1_angle, axis1_fitStdError`; the
four axis values are `NaN` when orientation is disabled), plus a stride-3
`positionCovariance()` accessor for the last detection. The C ABI
(`cc_abi_version` 7) carries the covariance in `cc_corner` and the frame's
noise σ in `cc_result`, each with a presence flag.

The exact per-binding shapes are the rustdoc/book reference.
