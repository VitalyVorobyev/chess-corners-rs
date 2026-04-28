//! Optional `image::GrayImage` helpers for the unified corner detector.

use crate::{find_chess_corners_u8, find_chess_corners_u8_with_refiner};
use crate::{ChessConfig, ChessError, CornerDescriptor, RefinerKind};
use image::GrayImage;

/// Detect chessboard corners from a `GrayImage`.
///
/// This is a thin wrapper over the multiscale detector. It honours
/// `cfg.upscale` and returns [`CornerDescriptor`] values in the
/// original input-image pixel coordinates.
///
/// A dimension mismatch is not possible for `GrayImage` since the
/// `image` crate guarantees `as_raw().len() == width * height`.
///
/// # Errors
///
/// Returns [`ChessError::Upscale`] if the upscale configuration is invalid.
pub fn find_chess_corners_image(
    img: &GrayImage,
    cfg: &ChessConfig,
) -> Result<Vec<CornerDescriptor>, ChessError> {
    find_chess_corners_u8(img.as_raw(), img.width(), img.height(), cfg)
}

/// Detect chessboard corners from a `GrayImage` with an explicit refiner choice.
///
/// # Errors
///
/// Returns [`ChessError::Upscale`] if the upscale configuration is invalid.
pub fn find_chess_corners_image_with_refiner(
    img: &GrayImage,
    cfg: &ChessConfig,
    refiner: &RefinerKind,
) -> Result<Vec<CornerDescriptor>, ChessError> {
    find_chess_corners_u8_with_refiner(img.as_raw(), img.width(), img.height(), cfg, refiner)
}

/// Detect chessboard corners from a `GrayImage` using the ML refiner pipeline.
///
/// # Errors
///
/// Returns [`ChessError::Upscale`] if the upscale configuration is invalid.
#[cfg(feature = "ml-refiner")]
pub fn find_chess_corners_image_with_ml(
    img: &GrayImage,
    cfg: &ChessConfig,
) -> Result<Vec<CornerDescriptor>, ChessError> {
    crate::find_chess_corners_u8_with_ml(img.as_raw(), img.width(), img.height(), cfg)
}
