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
}

impl fmt::Display for ChessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "image buffer length mismatch: expected {expected} bytes (width*height), got {actual}"
            ),
            Self::Upscale(e) => write!(f, "upscale error: {e}"),
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
