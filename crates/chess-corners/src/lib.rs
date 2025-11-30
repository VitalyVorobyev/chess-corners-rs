#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Ergonomic wrappers over `chess-corners-core` with optional `image` helpers.
//!
//! This crate is organized into a few focused modules:
//! - [`multiscale`] – unified single/multiscale corner finder.
//! - [`pyramid`] – minimal u8 buffers and downsampling for pyramids.
//! - optional `image` helpers for `image::GrayImage`.

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
pub use crate::multiscale::{find_chess_corners_buff, CoarseToFineParams, CoarseToFineResult};

/// Unified detector configuration combining response/detector params and
/// multiscale/pyramid tuning.
#[derive(Clone, Debug, Default)]
pub struct ChessConfig {
    pub params: ChessParams,
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
pub fn find_chess_corners(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
) -> CoarseToFineResult {
    let view = pyramid::ImageView::from_u8_slice(width, height, img)
        .expect("image dimensions must match buffer length");
    let mut buffers = pyramid::PyramidBuffers::with_capacity(cfg.multiscale.pyramid.num_levels);
    multiscale::find_chess_corners_buff(view, cfg, &mut buffers)
}
