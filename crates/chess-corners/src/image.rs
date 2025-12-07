//! Optional `image::GrayImage` helpers for the unified corner detector.

use crate::multiscale::find_chess_corners;
use crate::pyramid::ImageView;
use crate::{ChessConfig, CornerDescriptor};
use image::GrayImage;

/// Detect chessboard corners from a `GrayImage`.
///
/// This is a thin wrapper over the multiscale detector that builds an
/// [`ImageView`] from `img` and dispatches to single- or multiscale
/// mode based on `cfg.multiscale.pyramid.num_levels`, returning
/// [`CornerDescriptor`] values in full-resolution pixel coordinates.
#[must_use]
pub fn find_chess_corners_image(img: &GrayImage, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    let view =
        ImageView::from_u8_slice(img.width(), img.height(), img.as_raw()).expect("valid view");
    find_chess_corners(view, cfg)
}
