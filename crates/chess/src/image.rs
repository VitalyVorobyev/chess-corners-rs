//! Single-scale image helpers for the ChESS detector.
//!
//! These wrappers expose the core `chess-core` response and detection
//! primitives in terms of `image::GrayImage`, which is convenient for most
//! downstream consumers.

use chess_core::detect;
use chess_core::{ChessParams, ResponseMap};
use image::GrayImage;

/// Compute a dense ChESS response map for an `image::GrayImage`.
#[inline]
pub fn chess_response_image(img: &GrayImage, params: &ChessParams) -> ResponseMap {
    chess_core::response::chess_response_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Detect subpixel corners from an `image::GrayImage`.
#[inline]
pub fn find_corners_image(img: &GrayImage, params: &ChessParams) -> Vec<detect::Corner> {
    detect::find_corners_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Detect subpixel corners from an `image::GrayImage` and return timing stats.
#[inline]
pub fn find_corners_image_trace(img: &GrayImage, params: &ChessParams) -> detect::ChessResult {
    detect::find_corners_u8_with_trace(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}
