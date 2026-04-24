//! Ergonomic ChESS detector facade over `chess-corners-core`.
//!
//! # Overview
//!
//! This crate is the high-level entry point for the ChESS
//! (Chess-board Extraction by Subtraction and Summation) corner
//! detector. It exposes:
//!
//! - single-scale detection on raw grayscale buffers via
//!   [`find_chess_corners`],
//! - optional `image::GrayImage` helpers (see
//!   `find_chess_corners_image`) when the `image` feature is
//!   enabled,
//! - a flat user-facing [`ChessConfig`] with explicit modes for
//!   thresholding, ring selection, and multiscale tuning.
//!
//! The detector returns subpixel [`CornerDescriptor`] values in
//! full-resolution image coordinates. In most applications you
//! construct a [`ChessConfig`], optionally tweak its fields, and call
//! [`find_chess_corners`] or `find_chess_corners_image`.
//!
//! # Quick start
//!
//! ## Using `image` (default)
//!
//! The default feature set includes integration with the `image`
//! crate:
//!
//! ```no_run
//! use chess_corners::{ChessConfig, RefinementMethod, find_chess_corners_image};
//! use image::io::Reader as ImageReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a grayscale chessboard image.
//! let img = ImageReader::open("board.png")?
//!     .decode()?
//!     .to_luma8();
//!
//! // Start from the recommended coarse-to-fine preset.
//! let mut cfg = ChessConfig::multiscale();
//! cfg.threshold_value = 0.15;
//! cfg.refiner.kind = RefinementMethod::Forstner;
//!
//! let corners = find_chess_corners_image(&img, &cfg);
//! println!("found {} corners", corners.len());
//!
//! for c in &corners {
//!     println!(
//!         "corner at ({:.2}, {:.2}), response {:.1}, axes [{:.2}, {:.2}] rad",
//!         c.x, c.y, c.response, c.axes[0].angle, c.axes[1].angle,
//!     );
//! }
//! # Ok(()) }
//! ```
//!
//! ## Raw grayscale buffer
//!
//! If you already have an 8-bit grayscale buffer, you can call the
//! detector directly without depending on `image`:
//!
//! ```no_run
//! use chess_corners::{ChessConfig, find_chess_corners_u8};
//!
//! # fn detect(img: &[u8], width: u32, height: u32) {
//! // Single-scale convenience configuration.
//! let cfg = ChessConfig::single_scale();
//!
//! let corners = find_chess_corners_u8(img, width, height, &cfg);
//! println!("found {} corners", corners.len());
//! # let _ = corners;
//! # }
//! ```
//!
//! ## ML refiner (feature `ml-refiner`)
//!
//! ```no_run
//! # #[cfg(feature = "ml-refiner")]
//! # {
//! use chess_corners::{ChessConfig, find_chess_corners_image_with_ml};
//! use image::GrayImage;
//!
//! let img = GrayImage::new(1, 1);
//! let cfg = ChessConfig::single_scale();
//!
//! let corners = find_chess_corners_image_with_ml(&img, &cfg);
//! # let _ = corners;
//! # }
//! ```
//!
//! The ML refiner runs a small ONNX model on normalized intensity
//! patches (uint8 / 255.0) centered at each candidate. The model
//! predicts `[dx, dy, conf_logit]`, but the confidence output is
//! currently ignored; the offsets are applied directly. Current
//! benchmarks are synthetic; real-world accuracy still needs
//! validation. It is also slower (about 23.5 ms vs 0.6 ms for 77
//! corners on `testimages/mid.png`).
//!
//! ## Python bindings
//!
//! The workspace includes a PyO3-based Python extension crate at
//! `crates/chess-corners-py`. It exposes `chess_corners.find_chess_corners`,
//! which accepts a 2D `uint8` NumPy array and returns a float32 `(N, 9)` array
//! with columns `[x, y, response, contrast, fit_rms, axis0_angle,
//! axis0_sigma, axis1_angle, axis1_sigma]`. See
//! `crates/chess-corners-py/README.md` for usage and configuration details.
//!
//! For tight processing loops you can also reuse pyramid storage
//! explicitly via [`find_chess_corners_buff`] and the internal
//! `pyramid` module; this avoids reallocating intermediate pyramid
//! levels across frames. Most users should stick to
//! [`find_chess_corners`] / `find_chess_corners_image` unless they
//! need fine-grained control over allocations.
//!
//! # Configuration
//!
//! [`ChessConfig`] is intentionally flat. It exposes detector ring,
//! descriptor ring, threshold mode/value, NMS controls, refiner
//! choice, and multiscale settings directly. The detector translates
//! that high-level config into lower-level [`ChessParams`] and
//! [`CoarseToFineParams`] internally.
//!
//! If you need raw response maps or more control, depend directly on
//! `chess-corners-core` and use its [`chess_corners_core::response`]
//! and [`chess_corners_core::detect`] modules alongside the
//! re-exported [`ResponseMap`] and [`CornerDescriptor`] types.
//!
//! # Features
//!
//! - `image` *(default)* – enables `find_chess_corners_image` and
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
//! The ChESS idea was proposed in the papaer Bennett, Lasenby, *ChESS: A Fast and
//! Accurate Chessboard Corner Detector*, CVIU 2014

mod config;
#[cfg(feature = "ml-refiner")]
mod ml_refiner;
mod multiscale;
pub mod upscale;

// Re-export a focused subset of core types for convenience. Consumers that
// need lower-level primitives (rings, raw response functions, etc.) are
// encouraged to depend on `chess-corners-core` directly.
pub use crate::config::{
    ChessConfig, DescriptorMode, DetectorMode, RefinementMethod, RefinerConfig, ThresholdMode,
};
pub use crate::upscale::{UpscaleBuffers, UpscaleConfig, UpscaleError, UpscaleMode};
pub use chess_corners_core::{
    AxisEstimate, CenterOfMassConfig, ChessParams, CornerDescriptor, CornerRefiner, ForstnerConfig,
    ImageView, RadonPeakConfig, RefineResult, RefineStatus, Refiner, RefinerKind, ResponseMap,
    SaddlePointConfig,
};

// High-level helpers on `image::GrayImage`.
#[cfg(feature = "image")]
pub mod image;
#[cfg(all(feature = "image", feature = "ml-refiner"))]
pub use image::find_chess_corners_image_with_ml;
#[cfg(feature = "image")]
pub use image::{find_chess_corners_image, find_chess_corners_image_with_refiner};

// Multiscale/coarse-to-fine API types.
pub use crate::multiscale::{
    find_chess_corners, find_chess_corners_buff, find_chess_corners_buff_with_refiner,
    find_chess_corners_with_refiner, CoarseToFineParams,
};
#[cfg(feature = "ml-refiner")]
pub use crate::multiscale::{find_chess_corners_buff_with_ml, find_chess_corners_with_ml};
pub use box_image_pyramid::{ImageBuffer, PyramidBuffers, PyramidParams};

/// Detect chessboard corners from a raw grayscale image buffer.
///
/// The `img` slice must be `width * height` bytes in row-major order.
/// If `cfg.upscale` is enabled, the image is upscaled internally and
/// output corner coordinates are rescaled back to the original input
/// pixel frame.
///
/// # Panics
///
/// Panics if `img.len() != width * height` or if the upscale
/// configuration fails validation.
#[must_use]
pub fn find_chess_corners_u8(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
) -> Vec<CornerDescriptor> {
    run_with_upscale(img, width, height, cfg, |view, cfg| {
        multiscale::find_chess_corners(view, cfg)
    })
}

/// Detect corners from a raw grayscale buffer with an explicit refiner choice.
#[must_use]
pub fn find_chess_corners_u8_with_refiner(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    run_with_upscale(img, width, height, cfg, |view, cfg| {
        multiscale::find_chess_corners_with_refiner(view, cfg, refiner)
    })
}

/// Detect corners from a raw grayscale buffer using the ML refiner pipeline.
#[must_use]
#[cfg(feature = "ml-refiner")]
pub fn find_chess_corners_u8_with_ml(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
) -> Vec<CornerDescriptor> {
    run_with_upscale(img, width, height, cfg, |view, cfg| {
        multiscale::find_chess_corners_with_ml(view, cfg)
    })
}

/// Thread the optional upscaling stage around the detection closure.
/// Allocates a single-use `UpscaleBuffers`; callers with their own
/// buffer reuse pattern should drive the pipeline directly via the
/// `multiscale` module plus `upscale::upscale_bilinear_u8`.
fn run_with_upscale(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
    detect: impl FnOnce(ImageView<'_>, &ChessConfig) -> Vec<CornerDescriptor>,
) -> Vec<CornerDescriptor> {
    let src_w = width as usize;
    let src_h = height as usize;
    let view = ImageView::from_u8_slice(src_w, src_h, img)
        .expect("image dimensions must match buffer length");

    // Enforce the upscale invariants up-front so a misconfigured
    // `UpscaleMode::Fixed` with factor 0 or 1 fails loudly instead of
    // silently behaving like `Disabled`.
    cfg.upscale
        .validate()
        .expect("invalid upscale configuration");

    let factor = cfg.upscale.effective_factor();
    if factor <= 1 {
        return detect(view, cfg);
    }

    let mut buffers = UpscaleBuffers::new();
    let upscaled = upscale::upscale_bilinear_u8(img, src_w, src_h, factor, &mut buffers)
        .expect("invalid upscale configuration");
    let mut corners = detect(upscaled, cfg);
    upscale::rescale_descriptors_to_input(&mut corners, factor);
    corners
}
