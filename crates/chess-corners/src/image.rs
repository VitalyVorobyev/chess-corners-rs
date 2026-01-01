//! Optional `image::GrayImage` helpers for the unified corner detector.

#[cfg(feature = "ml-refiner")]
use crate::multiscale::find_chess_corners_with_ml;
use crate::multiscale::{find_chess_corners, find_chess_corners_with_refiner};
use crate::{ChessConfig, CornerDescriptor, RefinerKind};
use chess_corners_core::ImageView;
use image::GrayImage;

/// Detect chessboard corners from a `GrayImage`.
///
/// This is a thin wrapper over the multiscale detector that builds an
/// [`ImageView`] from `img` and dispatches to single- or multiscale
/// mode based on `cfg.multiscale.pyramid.num_levels`, returning
/// [`CornerDescriptor`] values in full-resolution pixel coordinates.
#[must_use]
pub fn find_chess_corners_image(img: &GrayImage, cfg: &ChessConfig) -> Vec<CornerDescriptor> {
    let view = ImageView::from_u8_slice(img.width() as usize, img.height() as usize, img.as_raw())
        .expect("valid view");
    find_chess_corners(view, cfg)
}

/// Detect chessboard corners from a `GrayImage` with an explicit refiner choice.
#[must_use]
pub fn find_chess_corners_image_with_refiner(
    img: &GrayImage,
    cfg: &ChessConfig,
    refiner: &RefinerKind,
) -> Vec<CornerDescriptor> {
    let view = ImageView::from_u8_slice(img.width() as usize, img.height() as usize, img.as_raw())
        .expect("valid view");
    find_chess_corners_with_refiner(view, cfg, refiner)
}

/// Detect chessboard corners from a `GrayImage` using the ML refiner pipeline.
#[must_use]
#[cfg(feature = "ml-refiner")]
pub fn find_chess_corners_image_with_ml(
    img: &GrayImage,
    cfg: &ChessConfig,
) -> Vec<CornerDescriptor> {
    let view = ImageView::from_u8_slice(img.width() as usize, img.height() as usize, img.as_raw())
        .expect("valid view");
    find_chess_corners_with_ml(view, cfg)
}
