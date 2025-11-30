//! Optional `image::GrayImage` helpers for the unified corner detector.

use crate::multiscale::{find_chess_corners_buff, CoarseToFineResult};
use crate::pyramid::{ImageView, PyramidBuffers};
use crate::ChessConfig;
use image::GrayImage;

/// Detect chessboard corners from a GrayImage. Dispatches to single- or
/// multiscale based on `cf.pyramid.num_levels`.
pub fn find_chess_corners_image(img: &GrayImage, cfg: &ChessConfig) -> CoarseToFineResult {
    let view =
        ImageView::from_u8_slice(img.width(), img.height(), img.as_raw()).expect("valid view");
    let mut buffers = PyramidBuffers::with_capacity(cfg.multiscale.pyramid.num_levels);
    find_chess_corners_buff(view, cfg, &mut buffers)
}
