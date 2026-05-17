//! Opt-in diagnostic channel for inspecting how a detection was produced.
//!
//! The items in this module expose **intermediate evidence** from the
//! detector pipeline — the dense ChESS response map and the Radon
//! heatmap — rather than the detection result itself. They exist for
//! debugging, parameter tuning, and visualization: rendering a response
//! map as an overlay, checking why a corner was or was not seeded, or
//! sweeping a threshold against a heatmap.
//!
//! These are **not** part of the normal detection result contract. The
//! supported way to obtain corners is [`Detector`] plus
//! [`DetectorConfig`](crate::DetectorConfig); a typical consumer never
//! needs this module.
//!
//! # Stability
//!
//! This channel carries a **looser stability promise** than
//! [`Detector`] and [`DetectorConfig`](crate::DetectorConfig).
//! The shape of an intermediate response map or heatmap may change as
//! the detector internals evolve, even when the detection result
//! contract does not. Treat anything reachable here as advisory
//! diagnostic data, not a versioned API guarantee.

/// Storage for a dense detector response map (one score per pixel).
pub use chess_corners_core::ResponseMap;

/// Compute the dense ChESS response map for an 8-bit grayscale image.
pub use chess_corners_core::chess_response_u8;

/// Compute the ChESS response over a sub-rectangle of an 8-bit image.
pub use chess_corners_core::unstable::chess_response_u8_patch;

/// Compute the Radon corner heatmap for an 8-bit grayscale buffer.
pub use crate::radon::radon_heatmap_u8;

/// Compute the Radon corner heatmap from an `image::GrayImage`.
#[cfg(feature = "image")]
pub use crate::radon::radon_heatmap_image;

use crate::error::ChessError;
use crate::Detector;

/// Detector-bound accessor for the diagnostic outputs in this module.
///
/// This is the detector-bound half of the diagnostics channel: it
/// exposes the same intermediate response maps and heatmaps as the
/// free functions above, but sources its parameters from an existing,
/// already-configured [`Detector`] instead of asking the caller to
/// re-supply a [`DetectorConfig`](crate::DetectorConfig). It is the
/// convenient path when you already hold a configured [`Detector`].
///
/// Obtain one with [`Detector::diagnostics`]. The handle borrows the
/// detector; it neither clones the configuration nor mutates the
/// detector.
///
/// # Stability
///
/// Like the free functions in this module, the methods here carry a
/// **looser stability promise** than [`Detector::detect`] and the
/// detection result contract. Treat their output as advisory
/// diagnostic data, not a versioned API guarantee.
pub struct DetectorDiagnostics<'a> {
    detector: &'a Detector,
}

impl<'a> DetectorDiagnostics<'a> {
    /// Wrap a borrowed detector. Constructed via [`Detector::diagnostics`].
    pub(crate) fn new(detector: &'a Detector) -> Self {
        Self { detector }
    }

    /// Compute the dense ChESS response map for an 8-bit grayscale
    /// buffer, using the ring and threshold parameters of the bound
    /// detector's [`DetectorConfig`](crate::DetectorConfig).
    ///
    /// `img` must be `width * height` bytes in row-major order. The
    /// response is computed at input resolution (the detector's
    /// upscale stage is not applied here); the returned
    /// [`ResponseMap`] therefore has the same dimensions as the input.
    ///
    /// This mirrors the free function [`chess_response_u8`] but sources
    /// [`ChessParams`](chess_corners_core::ChessParams) from the bound
    /// detector via
    /// [`DetectorConfig::to_chess_params`](crate::DetectorConfig::to_chess_params).
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::DimensionMismatch`] if
    /// `img.len() != width * height`.
    pub fn chess_response_u8(
        &self,
        img: &[u8],
        width: u32,
        height: u32,
    ) -> Result<ResponseMap, ChessError> {
        let w = width as usize;
        let h = height as usize;
        let expected = w * h;
        if img.len() != expected {
            return Err(ChessError::DimensionMismatch {
                expected,
                actual: img.len(),
            });
        }
        let params = self.detector.config().to_chess_params();
        Ok(chess_response_u8(img, w, h, &params))
    }

    /// Compute the dense Radon corner heatmap for an 8-bit grayscale
    /// buffer, using the bound detector's
    /// [`DetectorConfig`](crate::DetectorConfig).
    ///
    /// `img` must be `width * height` bytes in row-major order. As with
    /// the free function [`radon_heatmap_u8`], if the detector's
    /// upscale configuration is enabled the input is upscaled first and
    /// the heatmap is returned at working resolution.
    ///
    /// This mirrors [`radon_heatmap_u8`] but sources the
    /// [`DetectorConfig`](crate::DetectorConfig) from the bound
    /// detector.
    ///
    /// # Errors
    ///
    /// Returns [`ChessError::DimensionMismatch`] if
    /// `img.len() != width * height`. Returns [`ChessError::Upscale`]
    /// if the detector's upscale configuration is invalid.
    pub fn radon_heatmap_u8(
        &self,
        img: &[u8],
        width: u32,
        height: u32,
    ) -> Result<ResponseMap, ChessError> {
        crate::radon::radon_heatmap_u8(img, width, height, self.detector.config())
    }

    /// Compute the dense Radon corner heatmap from an
    /// [`image::GrayImage`], using the bound detector's
    /// [`DetectorConfig`](crate::DetectorConfig). See
    /// [`Self::radon_heatmap_u8`].
    ///
    /// # Errors
    ///
    /// Inherits the error contract of [`Self::radon_heatmap_u8`].
    #[cfg(feature = "image")]
    pub fn radon_heatmap(&self, img: &image::GrayImage) -> Result<ResponseMap, ChessError> {
        self.radon_heatmap_u8(img.as_raw(), img.width(), img.height())
    }
}
