//! Top-level error type for the `chess-corners` facade.
use crate::upscale::UpscaleError;
use std::fmt;

/// Errors returned by detection and heatmap entry points.
///
/// This type aggregates all failure modes reachable from the public
/// API. The [`From`] implementation for [`UpscaleError`] lets callers
/// propagate upscale failures with `?`.
#[derive(Debug)]
#[non_exhaustive]
pub enum ChessError {
    /// The supplied image slice length does not match `width * height`.
    DimensionMismatch {
        /// Expected length (`width * height`).
        expected: usize,
        /// Actual slice length.
        actual: usize,
    },
    /// An upscale configuration or execution error.
    Upscale(UpscaleError),
    /// The configured refiner is not available on the single-scale ROI
    /// detection path ([`Detector::detect_u8_roi`](crate::Detector::detect_u8_roi)
    /// / [`Detector::detect_roi`](crate::Detector::detect_roi)). The ML
    /// refiner runs a whole-frame model pipeline that the ROI path does
    /// not carry; the refiner selection is never silently downgraded, so
    /// this variant is returned instead. Select a built-in refiner
    /// (center-of-mass, Förstner, or saddle-point) for ROI detection, or
    /// use the whole-image [`Detector::detect_u8`](crate::Detector::detect_u8)
    /// / [`Detector::detect`](crate::Detector::detect) entry points for ML
    /// refinement.
    #[cfg(feature = "ml-refiner")]
    RoiRefinerUnsupported,
}

impl fmt::Display for ChessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "image buffer length mismatch: expected {expected} bytes (width*height), got {actual}"
            ),
            Self::Upscale(e) => write!(f, "upscale error: {e}"),
            #[cfg(feature = "ml-refiner")]
            Self::RoiRefinerUnsupported => write!(
                f,
                "configured refiner is not supported on the ROI detection path; \
                 select center-of-mass, Förstner, or saddle-point, or use the \
                 whole-image detect entry point"
            ),
        }
    }
}

impl std::error::Error for ChessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Upscale(e) => Some(e),
            _ => None,
        }
    }
}

impl From<UpscaleError> for ChessError {
    fn from(e: UpscaleError) -> Self {
        Self::Upscale(e)
    }
}
