//! Optional `image::GrayImage` helpers for the unified corner detector.

use crate::multiscale::{find_chess_corners, CoarseToFineParams, CoarseToFineResult};
use crate::pyramid::ImageView;
use chess_corners_core::ChessParams;
use image::GrayImage;

/// Detect chessboard corners from a GrayImage. Dispatches to single- or
/// multiscale based on `cf.pyramid.num_levels`.
pub fn find_chess_corners_image(
    img: &GrayImage,
    params: &ChessParams,
    cf: &CoarseToFineParams,
    buffers: &mut crate::pyramid::PyramidBuffers,
) -> CoarseToFineResult {
    let view =
        ImageView::from_u8_slice(img.width(), img.height(), img.as_raw()).expect("valid view");
    find_chess_corners(view, params, cf, buffers)
}
