//! Public Radon-detector convenience functions.
//!
//! The whole-image Duda-Frese Radon detector lives in
//! [`chess_corners_core::radon_detector`]; the corner-detection path is
//! exposed via [`crate::find_chess_corners_u8`] when
//! [`ChessConfig::detector_mode`](crate::ChessConfig::detector_mode) is
//! [`DetectorMode::Radon`](crate::DetectorMode::Radon). This module
//! adds a thin wrapper that returns the dense Radon response heatmap
//! (the intermediate `(max_α S_α − min_α S_α)²` image) for
//! visualization and debugging.
//!
//! The heatmap is returned at *working resolution* — that is,
//! `width * upscale_factor * radon_image_upsample` by the same in `y`.
//! Use [`ResponseMap::width`] / [`ResponseMap::height`] for the actual
//! dimensions; the working-to-input scale factor is
//! `cfg.upscale.effective_factor() *
//! cfg.radon_detector.image_upsample.clamp(1, 2)`.

use chess_corners_core::{radon_response_u8, ImageView, RadonBuffers, ResponseMap};

use crate::config::ChessConfig;
use crate::error::ChessError;
use crate::upscale::{self, UpscaleBuffers};

/// Compute the whole-image Radon response heatmap from a raw
/// grayscale buffer.
///
/// `img` must be `width * height` bytes in row-major order. If
/// `cfg.upscale` is enabled, the input is upscaled first (same path as
/// [`crate::find_chess_corners_u8`]) and the heatmap is returned at the
/// working resolution of the upscaled + radon-supersampled image.
///
/// The heatmap data is row-major `f32`, length
/// `map.width() * map.height()`. Values are non-negative.
///
/// # Errors
///
/// Returns [`ChessError::DimensionMismatch`] if `img.len() != width * height`.
/// Returns [`ChessError::Upscale`] if the upscale configuration is invalid.
pub fn radon_heatmap_u8(
    img: &[u8],
    width: u32,
    height: u32,
    cfg: &ChessConfig,
) -> Result<ResponseMap, ChessError> {
    cfg.upscale.validate()?;

    let src_w = width as usize;
    let src_h = height as usize;
    let expected = src_w * src_h;
    if img.len() != expected {
        return Err(ChessError::DimensionMismatch {
            expected,
            actual: img.len(),
        });
    }
    let view = ImageView::from_u8_slice(src_w, src_h, img).expect("dimensions were checked above");

    let factor = cfg.upscale.effective_factor();
    let mut rb = RadonBuffers::new();

    if factor <= 1 {
        let resp = radon_response_u8(
            view.data,
            view.width,
            view.height,
            &cfg.radon_detector,
            &mut rb,
        );
        return Ok(resp.to_response_map());
    }

    let mut up_buffers = UpscaleBuffers::new();
    let upscaled = upscale::upscale_bilinear_u8(img, src_w, src_h, factor, &mut up_buffers)?;
    let resp = radon_response_u8(
        upscaled.data,
        upscaled.width,
        upscaled.height,
        &cfg.radon_detector,
        &mut rb,
    );
    Ok(resp.to_response_map())
}

/// Compute the Radon response heatmap from an `image::GrayImage`.
///
/// Convenience wrapper over [`radon_heatmap_u8`] when the `image`
/// feature is enabled.
///
/// # Errors
///
/// Returns [`ChessError::Upscale`] if the upscale configuration in `cfg` is invalid.
/// A dimension mismatch is not possible for `GrayImage` since the `image` crate
/// guarantees `as_raw().len() == width * height`.
#[cfg(feature = "image")]
pub fn radon_heatmap_image(
    img: &::image::GrayImage,
    cfg: &ChessConfig,
) -> Result<ResponseMap, ChessError> {
    let (w, h) = img.dimensions();
    radon_heatmap_u8(img.as_raw(), w, h, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChessConfig;
    use chess_corners_core::{radon_response_u8 as core_radon, RadonBuffers as CoreRadonBuffers};

    fn synthetic_board(w: usize, h: usize) -> Vec<u8> {
        // 8×8 alternating-square board scaled to (w, h). Generates real
        // saddle structure so the Radon response is not all zeros.
        let cell = (w.min(h) / 9).max(2);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let cx = x / cell;
                let cy = y / cell;
                out[y * w + x] = if (cx + cy) & 1 == 0 { 220 } else { 35 };
            }
        }
        out
    }

    #[test]
    fn heatmap_matches_core_path_no_upscale() {
        let (w, h) = (96usize, 72usize);
        let img = synthetic_board(w, h);
        let cfg = ChessConfig::radon();

        let map = radon_heatmap_u8(&img, w as u32, h as u32, &cfg).unwrap();

        let mut rb = CoreRadonBuffers::new();
        let view = core_radon(&img, w, h, &cfg.radon_detector, &mut rb);
        assert_eq!(map.width(), view.width());
        assert_eq!(map.height(), view.height());
        assert_eq!(map.data().len(), view.data().len());
        // Bitwise-identical: the facade just copies the borrowed slice.
        assert_eq!(map.data(), view.data());
    }

    #[test]
    fn heatmap_dimensions_match_working_resolution() {
        let (w, h) = (96usize, 72usize);
        let img = synthetic_board(w, h);
        let cfg = ChessConfig::radon();
        let upsample = cfg.radon_detector.image_upsample.clamp(1, 2) as usize;

        let map = radon_heatmap_u8(&img, w as u32, h as u32, &cfg).unwrap();
        assert_eq!(map.width(), w * upsample);
        assert_eq!(map.height(), h * upsample);
    }

    #[test]
    fn heatmap_is_non_zero_on_a_board() {
        let (w, h) = (96usize, 72usize);
        let img = synthetic_board(w, h);
        let cfg = ChessConfig::radon();

        let map = radon_heatmap_u8(&img, w as u32, h as u32, &cfg).unwrap();
        let max = map.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > 0.0, "expected positive Radon response on a board");
    }

    #[test]
    fn heatmap_honors_upscale_factor() {
        use crate::upscale::UpscaleConfig;

        let (w, h) = (48usize, 36usize);
        let img = synthetic_board(w, h);
        let mut cfg = ChessConfig::radon();
        cfg.upscale = UpscaleConfig::fixed(2);
        let radon_upsample = cfg.radon_detector.image_upsample.clamp(1, 2) as usize;

        let map = radon_heatmap_u8(&img, w as u32, h as u32, &cfg).unwrap();
        // Working resolution = input × upscale × radon_image_upsample.
        assert_eq!(map.width(), w * 2 * radon_upsample);
        assert_eq!(map.height(), h * 2 * radon_upsample);
    }
}
