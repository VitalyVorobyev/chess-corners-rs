use chess_corners_core::SaddlePointConfig;
use serde::{Deserialize, Serialize};

/// The optional image-refinement stage
/// ([`DetectorConfig::refiner`](crate::DetectorConfig::refiner)).
///
/// Each detector first places its corners with its own subpixel localizer on
/// its response map: ChESS with the center of mass
/// ([`ChessConfig::center_of_mass`](crate::ChessConfig::center_of_mass)),
/// Radon with the peak fit ([`RadonConfig::peak_fit`](crate::RadonConfig::peak_fit)).
/// An image refiner reads only the input image around each corner, so every
/// variant works with both detectors. Where it starts and what happens when
/// it cannot refine a corner follow the detector:
///
/// - **ChESS**: the refiner replaces the center of mass. It starts from the
///   integer response peak, and a corner it rejects is dropped.
/// - **Radon**: the refiner runs on top of the peak fit. It starts from the
///   peak-fit position, and a corner it does not refine to convergence keeps
///   its peak-fit position. No corner is dropped.
///
/// Under [`Uncertainty::Propagated`](crate::Uncertainty::Propagated) a
/// corner's position covariance comes from whichever localizer produced its
/// final position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageRefiner {
    /// Quadratic surface fit to the image patch at the saddle point,
    /// re-centred until it converges. Usually more accurate than the
    /// detector's own localizer, most of all under lens blur and board
    /// rotation; on cells of about 4 px its fit window spans the
    /// neighbouring corners. Reports a position covariance on the corners it
    /// refines to convergence.
    SaddlePoint(SaddlePointConfig),
}

impl ImageRefiner {
    /// Saddle-point refinement with default tuning.
    pub fn saddle_point() -> Self {
        Self::SaddlePoint(SaddlePointConfig::default())
    }
}
