pub use chess_core::*;

use image::GrayImage;

/// Compute ChESS response map for an `image::GrayImage`.
pub fn chess_response_image(img: &GrayImage, params: &ChessParams) -> ResponseMap {
    chess_core::response::chess_response_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Detect subpixel corners from an `image::GrayImage`.
pub fn find_corners_image(img: &GrayImage, params: &ChessParams) -> Vec<chess_core::detect::Corner> {
    chess_core::detect::find_corners_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}
