//! Per-image pixel-noise estimate.

use crate::ChessError;

/// Estimate the standard deviation of an 8-bit grayscale image's pixel
/// noise, in gray levels.
///
/// This is the estimate [`Detector`](crate::Detector) uses to express
/// position covariances in px² when
/// [`DetectorConfig::noise_sigma`](crate::DetectorConfig::noise_sigma) is
/// `None`; calling it on the same image returns the same value. Strong
/// edges, corners, smooth shading and areas near black or white are
/// excluded, so it suits images dominated by a calibration board. It is not
/// reliable on demosaiced or JPEG images; supply
/// [`DetectorConfig::noise_sigma`](crate::DetectorConfig::noise_sigma)
/// for those. Cost is independent of the image size (well under a
/// millisecond).
///
/// `img` is `width * height` bytes, row-major. Returns `Ok(None)` when the
/// image is smaller than 4×4 or has too few structure-free areas for an
/// estimate. The full accuracy statement is on
/// [`chess_corners_core::estimate_noise_sigma_u8`].
///
/// ```
/// // A flat gray image with no noise: the estimate is the 8-bit
/// // rounding floor, near zero.
/// let img = vec![128u8; 64 * 64];
/// let sigma = chess_corners::estimate_noise_sigma_u8(&img, 64, 64)?;
/// assert!(sigma.is_none_or(|s| s < 0.5));
/// # Ok::<(), chess_corners::ChessError>(())
/// ```
///
/// # Errors
///
/// Returns [`ChessError::DimensionMismatch`] if `img.len() != width * height`.
pub fn estimate_noise_sigma_u8(
    img: &[u8],
    width: u32,
    height: u32,
) -> Result<Option<f32>, ChessError> {
    let (w, h) = (width as usize, height as usize);
    let expected = w * h;
    if img.len() != expected {
        return Err(ChessError::DimensionMismatch {
            expected,
            actual: img.len(),
        });
    }
    Ok(chess_corners_core::estimate_noise_sigma_u8(img, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_corners_testutil::aa_chessboard;

    #[test]
    fn rejects_dimension_mismatch() {
        let err = estimate_noise_sigma_u8(&[0u8; 10], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            ChessError::DimensionMismatch {
                expected: 16,
                actual: 10
            }
        ));
    }

    #[test]
    fn matches_the_core_estimate() {
        let size = 128usize;
        let mut img = aa_chessboard(size, 12, (0.3, 0.6), 40, 200);
        // Deterministic ±3 gray-level noise.
        let mut state = 11u32;
        for v in &mut img {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = v
                .saturating_add(((state >> 16) % 7) as u8)
                .saturating_sub(3);
        }
        let facade = estimate_noise_sigma_u8(&img, size as u32, size as u32).unwrap();
        let core = chess_corners_core::estimate_noise_sigma_u8(&img, size, size);
        assert_eq!(facade, core);
        assert!(facade.is_some_and(|s| s > 1.0));
    }
}
