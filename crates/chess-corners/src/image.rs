//! Optional `image::GrayImage` helpers for the unified corner detector.

use crate::multiscale::{find_chess_corners};
use crate::pyramid::ImageView;
use crate::{ChessConfig, CornerDescriptor};
use image::GrayImage;

/// Detect chessboard corners from a GrayImage. Dispatches to single- or
/// multiscale based on `cf.pyramid.num_levels`.
pub fn find_chess_corners_image(img: &GrayImage, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    let view =
        ImageView::from_u8_slice(img.width(), img.height(), img.as_raw()).expect("valid view");
    find_chess_corners(view, cfg)
}
