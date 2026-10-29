use serde::{Deserialize, Serialize};

use crate::ChessError;

/// Which position uncertainty each corner reports in
/// [`position_covariance`](crate::CornerDescriptor::position_covariance).
///
/// Detector-independent: it applies to every localizer (the ChESS
/// refiners and the Radon peak fit) the same way. Corner positions are the
/// same under every variant.
///
/// In JSON: `"off"`, `{"propagated": {"noise": "estimate"}}` or
/// `{"propagated": {"noise": {"known": 1.2}}}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Uncertainty {
    /// No position covariance (the default). Localizers skip the noise
    /// propagation and the systematic model, and the detector makes no
    /// noise estimate.
    #[default]
    Off,
    /// Image noise propagated through the localizer to first order, plus
    /// the localizer's systematic-error model (from the corner's offset to
    /// the localizer's fit window and the width of its edges). `noise` is
    /// the image noise level that expresses the propagated part in px².
    ///
    /// Every localizer reports a covariance. None is reported on upscaled
    /// input ([`DetectorConfig::upscale`](crate::DetectorConfig::upscale)):
    /// interpolation correlates the noise of neighbouring pixels.
    ///
    /// The systematic part includes a per-localizer floor measured on real
    /// calibration images (about 0.05 px per axis for the saddle-point fit
    /// and the Radon peak fit, 0.07 center of mass). On real, well-exposed
    /// images this floor is usually the largest part of the covariance, so
    /// the reported σ varies little from corner to corner there. Weighting
    /// a calibration by it has not been shown to beat uniform weighting on
    /// real data. Errors shared by all corners of an image (defocus,
    /// lens-model error) are outside it. The floor was fitted on one camera
    /// pair and is likely low for other optics. Within that pair the two
    /// cameras' own floors differed by up to 2×.
    Propagated {
        /// The image noise level.
        noise: NoiseLevel,
    },
}

/// The image noise level that turns a propagated position covariance into
/// px² (see [`Uncertainty::Propagated`]).
///
/// In JSON: `"estimate"` or `{"known": 1.2}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NoiseLevel {
    /// Estimate σ from each image with
    /// [`estimate_noise_sigma_u8`](crate::estimate_noise_sigma_u8) (the
    /// default), once per frame and only when a corner carries a
    /// covariance. The estimate is not reliable on demosaiced or JPEG
    /// images; use [`Known`](Self::Known) for those.
    #[default]
    Estimate,
    /// A known standard deviation of the image's pixel noise, in gray
    /// levels (finite and `>= 0`), for example the standard deviation of
    /// the difference of two frames of a static scene, divided by `√2`.
    /// It is the total noise as it appears in the 8-bit image, rounding
    /// included (rounding adds `1/12` gray level² to the sensor noise,
    /// which matters only below about 1 gray level). The covariance never
    /// uses a variance below that `1/12`.
    Known(
        #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
        #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "gray level")))]
        f32,
    ),
}

impl Uncertainty {
    /// Whether the localizers report a covariance.
    #[inline]
    pub(crate) fn propagates(self) -> bool {
        matches!(self, Self::Propagated { .. })
    }

    /// Reject a known noise level that is negative or not finite.
    pub(crate) fn validate(self) -> Result<(), ChessError> {
        match self {
            Self::Propagated {
                noise: NoiseLevel::Known(sigma),
            } if !(sigma.is_finite() && sigma >= 0.0) => Err(ChessError::InvalidNoiseSigma(sigma)),
            _ => Ok(()),
        }
    }
}
