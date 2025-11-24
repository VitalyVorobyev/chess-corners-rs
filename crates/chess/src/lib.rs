pub use chess_core::*;

use image::GrayImage;

pub fn chess_response_image(img: &GrayImage, params: &ChessParams) -> ResponseMap {
    chess_core::response::chess_response_u8(img.as_raw(), img.width() as usize, img.height() as usize, params)
}
