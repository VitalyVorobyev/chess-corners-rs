#![cfg_attr(all(feature = "simd", feature = "par_pyramid"), feature(portable_simd))]
//! Ergonomic ChESS detector facade over `chess-corners-core`.
//!
//! # Overview
//!
//! This crate is the high-level entry point for the ChESS
//! (Chess-board Extraction by Subtraction and Summation) corner
//! detector. It re-exports the main configuration/result types
//! from [`chess_corners_core`] and adds:
//!
//! - single-scale detection on raw grayscale buffers via
//!   [`find_chess_corners`],
//! - optional `image::GrayImage` helpers (see
//!   `find_chess_corners_image`) when the `image` feature is
//!   enabled,
//! - a coarse-to-fine multiscale detector configured through
//!   [`ChessConfig`] and [`CoarseToFineParams`].
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
//! use chess_corners::{ChessConfig, ChessParams, find_chess_corners_image};
//! use image::io::Reader as ImageReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a grayscale chessboard image.
//! let img = ImageReader::open("board.png")?
//!     .decode()?
//!     .to_luma8();
//!
//! // Start from the recommended defaults.
//! let mut cfg = ChessConfig::default();
//! cfg.params = ChessParams::default();
//!
//! // For a single chessboard in the frame, 2–3 pyramid levels
//! // are often enough. Setting `num_levels` to 1 forces
//! // single-scale detection.
//! cfg.multiscale.pyramid.num_levels = 3;
//!
//! let corners = find_chess_corners_image(&img, &cfg);
//! println!("found {} corners", corners.len());
//!
//! for c in &corners {
//!     println!(
//!         "corner at ({:.2}, {:.2}), response {:.1}, theta {:.2} rad, phase {}",
//!         c.x, c.y, c.response, c.orientation, c.phase,
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
//! use chess_corners::{ChessConfig, ChessParams, find_chess_corners_u8};
//!
//! # fn detect(img: &[u8], width: u32, height: u32) {
//! // Single-scale convenience configuration.
//! let mut cfg = ChessConfig::single_scale();
//! cfg.params = ChessParams::default();
//!
//! let corners = find_chess_corners_u8(img, width, height, &cfg);
//! println!("found {} corners", corners.len());
//! # let _ = corners;
//! # }
//! ```
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
//! [`ChessConfig`] combines the low-level ChESS parameters with
//! multiscale tuning:
//!
//! - [`ChessParams`] (re-exported from `chess-corners-core`) controls
//!   the response kernel and detector behavior: ring radius, relative
//!   or absolute threshold, non-maximum suppression radius, and the
//!   minimum cluster size for accepting a corner.
//! - [`CoarseToFineParams`] describes how the multiscale detector
//!   behaves: number of pyramid levels, minimum level size, coarse
//!   ROI radius (at the smallest level) and merge radius for
//!   deduplicating refined corners.
//!
//! The shortcut [`ChessConfig::single_scale`] configures a
//! single-scale run by setting `multiscale.pyramid.num_levels = 1`.
//! Any `pyramid.num_levels > 1` triggers the coarse-to-fine path:
//! corners are first detected on the smallest pyramid level and then
//! refined inside base-image regions of interest.
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

mod multiscale;
mod pyramid;

// Re-export a focused subset of core types for convenience. Consumers that
// need lower-level primitives (rings, raw response functions, etc.) are
// encouraged to depend on `chess-corners-core` directly.
pub use chess_corners_core::{ChessParams, CornerDescriptor, ResponseMap};

// High-level helpers on `image::GrayImage`.
#[cfg(feature = "image")]
pub mod image;
#[cfg(feature = "image")]
pub use image::find_chess_corners_image;

// Multiscale/coarse-to-fine API types.
pub use crate::multiscale::{find_chess_corners, find_chess_corners_buff, CoarseToFineParams};
pub use crate::pyramid::{ImageView, PyramidBuffers};

/// Unified detector configuration combining response/detector params and
/// multiscale/pyramid tuning.
#[derive(Clone, Debug, Default)]
pub struct ChessConfig {
    /// Low-level ChESS response/detector parameters (ring radius, thresholds,
    /// NMS radius, minimum cluster size).
    pub params: ChessParams,
    /// Coarse-to-fine multiscale configuration (pyramid shape, ROI radius,
    /// merge radius).
    pub multiscale: CoarseToFineParams,
}

impl ChessConfig {
    /// Convenience helper for single-scale detection.
    pub fn single_scale() -> Self {
        let mut cfg = Self::default();
        cfg.multiscale.pyramid.num_levels = 1;
        cfg
    }
}

/// Detect chessboard corners from a raw grayscale image buffer.
///
/// The `img` slice must be `width * height` bytes in row-major order.
///
/// # Panics
///
/// Panics if `img.len() != width * height`.
#[must_use]
pub fn find_chess_corners_u8(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
) -> Vec<CornerDescriptor> {
    let view = pyramid::ImageView::from_u8_slice(width, height, img)
        .expect("image dimensions must match buffer length");
    multiscale::find_chess_corners(view, cfg)
}
