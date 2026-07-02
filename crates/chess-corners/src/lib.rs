//! Ergonomic ChESS / Radon corner-detector facade over
//! `chess-corners-core`.
//!
//! # Overview
//!
//! This crate is the high-level entry point for two chessboard-corner
//! detectors that share the same output surface:
//!
//! - **ChESS** (Chess-board Extraction by Subtraction and Summation)
//!   — a dense ring-difference response with NMS and a pluggable
//!   subpixel refiner. This is the default path and the fastest preset
//!   in the repository's clean-image benchmark.
//! - **Radon** — a whole-image Duda-Frese accumulator that scores
//!   corners by summing ray intensities through each pixel. It is useful
//!   when the ChESS ring does not produce enough seeds, especially in
//!   the small-cell, blur, and low-contrast fixtures covered by the
//!   tests.
//!
//! The [`Detector`] struct ties together the active strategy, the
//! orientation fit, and the multiscale / upscale scratch buffers
//! behind a single `detect` call. It returns subpixel
//! [`CornerDescriptor`] values in full-resolution input coordinates.
//! In most applications you construct a [`DetectorConfig`] (typically
//! via [`DetectorConfig::chess`], [`DetectorConfig::chess_multiscale`],
//! [`DetectorConfig::radon`], or [`DetectorConfig::radon_multiscale`]),
//! optionally tweak its fields, build a [`Detector`], and call
//! [`Detector::detect`].
//!
//! Building a [`Detector`] once and calling [`Detector::detect`] in a
//! loop reuses the pyramid, response, and upscale scratch buffers
//! across frames — no per-frame allocation.
//!
//! # Quick start
//!
//! ## Using `image` (default)
//!
//! The default feature set includes integration with the `image`
//! crate. This example reads from disk and is marked `no_run`:
//!
//! ```no_run
//! use chess_corners::{ChessRefiner, Detector, DetectorConfig};
//! use image::io::Reader as ImageReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let img = ImageReader::open("board.png")?
//!     .decode()?
//!     .to_luma8();
//!
//! let cfg = DetectorConfig::chess_multiscale()
//!     .with_threshold(120.0)
//!     .with_chess(|c| c.refiner = ChessRefiner::forstner());
//!
//! let mut detector = Detector::new(cfg)?;
//! let corners = detector.detect(&img)?;
//! println!("found {} corners", corners.len());
//!
//! for c in &corners {
//!     // `axes` is `None` when the orientation fit was skipped
//!     // (see `DetectorConfig::without_orientation`).
//!     if let Some(axes) = c.axes {
//!         println!(
//!             "corner at ({:.2}, {:.2}), response {:.1}, axes [{:.2}, {:.2}] rad",
//!             c.x, c.y, c.response, axes[0].angle, axes[1].angle,
//!         );
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! ## Raw grayscale buffer
//!
//! If you already have an 8-bit grayscale buffer, call
//! [`Detector::detect_u8`]:
//!
//! ```
//! use chess_corners::{Detector, DetectorConfig};
//!
//! // 8×8 black/white checkerboard of 16-pixel squares (128×128).
//! let mut img = vec![0u8; 128 * 128];
//! for y in 0..128 {
//!     for x in 0..128 {
//!         if ((x / 16) + (y / 16)) % 2 == 0 {
//!             img[y * 128 + x] = 255;
//!         }
//!     }
//! }
//!
//! let cfg = DetectorConfig::chess();
//! let mut detector = Detector::new(cfg)?;
//! let corners = detector.detect_u8(&img, 128, 128)?;
//! assert!(!corners.is_empty());
//! # Ok::<(), chess_corners::ChessError>(())
//! ```
//!
//! ## Radon strategy
//!
//! Switch to the whole-image Radon detector when ChESS misses corners on
//! the images you care about. The strategy lives inside
//! [`DetectorConfig::strategy`]; pick a Radon preset to get sensible
//! defaults:
//!
//! ```
//! use chess_corners::{Detector, DetectorConfig};
//!
//! let mut img = vec![0u8; 128 * 128];
//! for y in 0..128 {
//!     for x in 0..128 {
//!         if ((x / 16) + (y / 16)) % 2 == 0 {
//!             img[y * 128 + x] = 255;
//!         }
//!     }
//! }
//!
//! let cfg = DetectorConfig::radon();
//! let mut detector = Detector::new(cfg)?;
//! let corners = detector.detect_u8(&img, 128, 128)?;
//! assert!(!corners.is_empty());
//! # Ok::<(), chess_corners::ChessError>(())
//! ```
//!
//! ## ML refiner (feature `ml-refiner`)
//!
//! Pick the ML pipeline by selecting `ChessRefiner::Ml` inside the
//! ChESS strategy. The example is marked `no_run` because loading the
//! embedded ONNX model on first use is not appropriate for a doctest:
//!
//! ```no_run
//! # #[cfg(feature = "ml-refiner")]
//! # {
//! use chess_corners::{ChessRefiner, Detector, DetectorConfig};
//! use image::GrayImage;
//!
//! let cfg = DetectorConfig::chess()
//!     .with_chess(|c| c.refiner = ChessRefiner::Ml);
//!
//! let img: GrayImage = image::open("board.png").unwrap().to_luma8();
//! let mut detector = Detector::new(cfg).unwrap();
//! let corners = detector.detect(&img).unwrap();
//! # let _ = corners;
//! # }
//! ```
//!
//! The ML refiner runs a small ONNX model on normalized intensity
//! patches (uint8 / 255.0) centered at each candidate. The model
//! predicts `[dx, dy, conf_logit]`, but the confidence output is
//! currently ignored; the offsets are applied directly. Current
//! accuracy benchmarks are synthetic; real-world accuracy still needs
//! validation. Per-refiner cost is measured in Part VIII §7.6 of the
//! book. The ML path is slower than the hand-coded refiners and should
//! be chosen only after measuring that its behavior helps your data.
//!
//! ## Python and JavaScript bindings
//!
//! The workspace also ships bindings that wrap this facade:
//!
//! - `crates/chess-corners-py` (PyO3 / maturin) exposes a
//!   `chess_corners.Detector` class whose `detect(image)` method
//!   accepts a 2D `uint8` NumPy array and returns a `Detections`
//!   structure-of-arrays object with named fields: `.xy` (`(N, 2)`
//!   float32), `.response` (`(N,)` float32), `.angles` (`(N, 2)`
//!   float32, or `None` when orientation is disabled), and `.sigmas`
//!   (`(N, 2)` float32, or `None` when orientation is disabled). See
//!   its README for usage and configuration details.
//! - `crates/chess-corners-wasm` (wasm-bindgen / wasm-pack) exposes
//!   the same surface to JavaScript / TypeScript via the
//!   `@vitavision/chess-corners` npm package.
//!
//! # Configuration
//!
//! [`DetectorConfig`] is strategy-typed: the [`DetectorConfig::strategy`]
//! field is a [`DetectionStrategy`] enum carrying either a
//! [`ChessConfig`] (detector ring, refiner) or a [`RadonConfig`]
//! (whole-image Duda-Frese parameters). Acceptance is a single
//! [`threshold`](DetectorConfig::threshold) number — read as an absolute
//! response floor by ChESS and as a fraction of the per-frame maximum by
//! Radon. [`MultiscaleConfig`] and [`UpscaleConfig`] live at the top level
//! and apply to both strategies. The detector translates this into
//! lower-level parameter structs internally. To drive those stages
//! yourself, lower a config with [`DetectorConfig::chess_params`] or
//! [`DetectorConfig::radon_detector_params`] and call the stage
//! functions re-exported from `chess-corners-core`.
//!
//! Intermediate response maps and Radon heatmaps for debugging and
//! visualization live in the opt-in [`diagnostics`] module, which
//! carries a weaker stability promise than the facade root and is not
//! needed by typical consumers. For deeper internals (ring offsets, SAT
//! views, scalar reference paths) depend on `chess-corners-core`
//! directly.
//!
//! # Features
//!
//! - `image` *(default)* – enables [`Detector::detect`] and
//!   `image::GrayImage` integration.
//! - `rayon` – parallelizes response computation and multiscale
//!   refinement over image rows. Combine with `par_pyramid` to
//!   parallelize pyramid downsampling as well.
//! - `ml-refiner` – enables the ML-backed refiner entry points via the
//!   `chess-corners-ml` crate and embedded ONNX model.
//! - `simd` – enables portable-SIMD accelerated inner loops for the
//!   response kernel (requires a nightly compiler). Combine with
//!   `par_pyramid` to SIMD-accelerate pyramid downsampling.
//! - `par_pyramid` – opt-in gate for SIMD/`rayon` acceleration inside
//!   the pyramid builder.
//! - `tracing` – emits structured spans for multiscale detection,
//!   suitable for use with `tracing-subscriber` or JSON tracing from
//!   the CLI.
//! - `cli` – builds the `chess-corners` binary shipped with this
//!   crate; it is not required when using the library as a
//!   dependency.
//!
//! The library API is stable across feature combinations; features
//! only affect performance and observability, not numerical results.
//!
//! # Minimum supported Rust version
//!
//! The default (stable) build requires Rust **1.88** or newer, as
//! declared by `rust-version` in `Cargo.toml`. The optional `simd`
//! feature uses `portable_simd` and therefore requires a **nightly**
//! toolchain; every other feature builds on stable.
//!
//! # References
//!
//! - Bennett, Lasenby. *ChESS: A Fast and Accurate Chessboard Corner
//!   Detector*. CVIU 2014.
//! - Duda, Frese. *Accurate Detection and Localization of Checkerboard
//!   Corners for Calibration*. BMVC 2018.

mod config;
mod detector;
pub mod diagnostics;
mod error;
#[cfg(feature = "ml-refiner")]
mod ml_refiner;
mod multiscale;
mod radon;
mod upscale;

// The crate root surfaces the stable facade: the detector, its
// configuration and result types, errors, and the config-lowering and
// pipeline-stage primitives for callers that compose the stages
// themselves. Diagnostic outputs are reachable via [`diagnostics`];
// deeper internals (ring offsets, SAT views, scalar reference paths)
// via a direct `chess-corners-core` dependency.
pub use crate::config::{
    ChessConfig, ChessRefiner, ChessRing, DetectionParams, DetectionStrategy, DetectorConfig,
    MultiscaleConfig, RadonConfig,
};
pub use crate::error::ChessError;

/// Optional pre-pipeline integer bilinear upscaling stage. These are the
/// raw stage primitives behind [`UpscaleConfig`]; [`Detector`] applies
/// the stage automatically, so they are only needed when composing the
/// pipeline by hand.
pub use crate::upscale::{
    rescale_descriptors_to_input, upscale_bilinear_u8, UpscaleBuffers, UpscaleConfig, UpscaleError,
};
pub use chess_corners_core::{
    AxisEstimate, CenterOfMassConfig, CornerDescriptor, ForstnerConfig, OrientationMethod,
    PeakFitMode, SaddlePointConfig,
};

// Primary detector entry point.
pub use crate::detector::Detector;
