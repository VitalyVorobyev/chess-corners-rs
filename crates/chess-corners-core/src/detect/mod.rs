//! Feature-detection pipelines.
//!
//! Two independent detector families share a common output type:
//!
//! - [`chess`] — the ChESS (Chess-board Extraction by Subtraction and
//!   Summation) detector: 16-sample ring kernel, NMS, cluster filtering,
//!   pluggable subpixel refinement.
//! - [`radon`] — the Duda-Frese (2018) localized Radon detector:
//!   integral-image (SAT) ray sums, peak detection, Gaussian peak fit.
//!
//! Both families produce the same [`Corner`] / [`CornerDescriptor`]
//! values, then go through the orthogonal subpixel-refinement and
//! orientation-fit stages defined in [`crate::refine`] and
//! [`crate::orientation`].

pub mod chess;
pub mod dense;
mod merge;
mod neighbors;
pub mod radon;

pub use chess::detect::{
    detect_corners_from_response, detect_corners_from_response_with_refiner,
    detect_peaks_from_response_with_refine_radius, find_corners_u8, refine_corners_on_image,
};
pub use merge::merge_corners_simple;
pub(crate) use neighbors::{count_positive_neighbors, is_local_max};

use crate::refine::PositionCovariance;
use crate::uncertainty::CovarianceModel;

/// A detected corner candidate (subpixel position with raw response strength).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Corner {
    /// Subpixel x coordinate in image pixels.
    pub x: f32,
    /// Subpixel y coordinate in image pixels.
    pub y: f32,
    /// Raw detector response at the integer peak (before refinement).
    pub strength: f32,
    /// The noise term of the position covariance per unit image-noise
    /// variance (px² per gray-level²); `None` when the corner carries no
    /// covariance. Read it with
    /// [`Corner::noise_covariance_per_unit_variance`]; the full covariance
    /// in px² is [`Corner::position_covariance`]. Changing this field drops
    /// the systematic term.
    #[deprecated(
        since = "1.7.0",
        note = "read `noise_covariance_per_unit_variance()`, or `position_covariance(noise_sigma)` for the full covariance in px²; writing this field drops the systematic term"
    )]
    pub covariance_per_noise_variance: Option<PositionCovariance>,
    /// The full covariance model; `covariance_per_noise_variance` mirrors
    /// its noise term.
    pub(crate) model: Option<CovarianceModel>,
    /// The position is the detector's own subpixel peak fit (a Radon peak
    /// no image refiner moved), so the peak fit's covariance applies to it
    /// (see [`DenseDetector::attach_peak_covariance`](crate::DenseDetector::attach_peak_covariance)).
    pub(crate) at_peak_fit: bool,
}

impl Corner {
    /// Construct a [`Corner`] without a covariance.
    #[inline]
    #[allow(deprecated)]
    pub fn new(x: f32, y: f32, strength: f32) -> Self {
        Self {
            x,
            y,
            strength,
            covariance_per_noise_variance: None,
            model: None,
            at_peak_fit: false,
        }
    }

    /// Attach `model`, mirroring its noise term in the public field.
    #[inline]
    #[allow(deprecated)]
    pub(crate) fn set_model(&mut self, model: Option<CovarianceModel>) {
        self.covariance_per_noise_variance = model.map(|m| m.noise);
        self.model = model;
    }

    /// The covariance model this corner carries (see
    /// [`CovarianceModel::effective`]).
    #[inline]
    #[allow(deprecated)]
    pub(crate) fn covariance_model(&self) -> Option<CovarianceModel> {
        CovarianceModel::effective(self.model, self.covariance_per_noise_variance)
    }

    /// The noise term of the position covariance per unit image-noise
    /// variance, px² per gray-level²: `C` in
    /// [`position_covariance`](Self::position_covariance)'s
    /// `Σ = max(σₙ², 1/12)·C + Σ_sys`, the first-order scatter that white
    /// pixel noise causes. `None` when the corner carries no covariance.
    #[inline]
    pub fn noise_covariance_per_unit_variance(&self) -> Option<PositionCovariance> {
        self.covariance_model().map(|m| m.noise)
    }

    /// Position covariance in px² for an image whose pixels carry white
    /// noise of standard deviation `noise_sigma` gray levels: the total
    /// noise of the 8-bit image, rounding included, as
    /// [`estimate_noise_sigma_u8`](crate::estimate_noise_sigma_u8) measures
    /// it (or the standard deviation of the difference of two frames of a
    /// static scene, divided by `√2`):
    ///
    /// `Σ = max(σₙ², 1/12)·C + Σ_sys`
    ///
    /// - `C` = [`noise_covariance_per_unit_variance`](Self::noise_covariance_per_unit_variance)
    ///   is the first-order scatter that white pixel noise causes, per unit
    ///   noise variance. `σₙ²` is at least `1/12` gray level², the variance
    ///   of rounding to 8 bits, so a smoothed or compressed image whose
    ///   measured noise is lower does not get an overconfident covariance.
    /// - `Σ_sys` is the localizer's systematic error. A single-shot
    ///   localizer (center of mass, Förstner, the Radon peak fit) is pulled
    ///   toward the centre of its fit window by an amount set by the
    ///   corner's offset from it and by the width of the corner's edges;
    ///   `Σ_sys` holds that expected pull and the spread around it, as
    ///   calibrated per localizer on synthetic camera-formed boards. The
    ///   re-centred saddle-point fit has no such pull; its `Σ_sys` keeps the
    ///   small error it has where the corner sits between pixels. Every
    ///   localizer's `Σ_sys` also carries a floor measured on real
    ///   calibration images, for the error real optics and targets add
    ///   (about 0.05 px per axis for the saddle-point fit and the Radon peak
    ///   fit, 0.07 for the center of mass, 0.11 for Förstner); on
    ///   ideal synthetic corners the covariance is therefore conservative.
    ///   On real, well-exposed images this floor is usually the largest
    ///   part of `Σ`, so the reported σ varies little from corner to
    ///   corner there, and weighting a calibration by it has not been
    ///   shown to beat uniform weighting on real data.
    ///
    /// Known limits: the systematic model cannot see the board's rotation,
    /// so the center of mass is overconfident on sharp boards rotated a few
    /// degrees, and the Radon peak fit on blurred corners near 45° (its
    /// flat-topped response lets noise move the peak by up to a pixel; the
    /// whitened error variance reaches about 4 on synthetic boards, more
    /// under asymmetric blur). Error that every corner of an image shares
    /// (defocus, depth, lens-model error) is outside any single-corner
    /// covariance; it shows up in a downstream calibration's variance
    /// factor. A corner whose `covariance_per_noise_variance` field was set
    /// or changed by the caller carries that noise term only.
    /// Valid only for the image the corner was refined on. With
    /// spatially correlated noise (demosaiced, compressed or resampled
    /// images) [`estimate_noise_sigma_u8`](crate::estimate_noise_sigma_u8)
    /// is not reliable: pass a `σₙ` measured on the delivered images, and
    /// the noise term is then approximate. `None` when the corner carries no
    /// covariance.
    #[inline]
    pub fn position_covariance(&self, noise_sigma: f32) -> Option<PositionCovariance> {
        self.covariance_model().map(|m| m.total(noise_sigma))
    }
}

/// Direction of one local grid axis with a fit-quality score.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct AxisEstimate {
    /// Axis direction, radians in `[0, 2π)`.
    ///
    /// See [`CornerDescriptor`] for the joint polarity convention.
    pub angle: f32,
    /// Fit-quality score in radians: larger means a less reliable axis
    /// estimate. It is not a calibrated 1σ.
    ///
    /// How it is derived depends on the orientation method. `RingFit`
    /// scales the Gauss-Newton covariance of the fit by the residual and
    /// an empirical factor; `DiskFit` reports a residual-based heuristic
    /// with a method-specific floor. Neither uses the image noise level or
    /// [`Uncertainty`](crate::Uncertainty), and it is computed whenever
    /// orientation is enabled. Use it for ranking and gating corners.
    pub sigma: f32,
}

impl AxisEstimate {
    /// Construct an [`AxisEstimate`].
    #[inline]
    pub fn new(angle: f32, sigma: f32) -> Self {
        Self { angle, sigma }
    }
}

/// Describes a detected chessboard corner in full-resolution image coordinates.
///
/// # Axis polarity convention
///
/// Local chessboard corner intensity patterns have exact 180° symmetry,
/// so assigning an absolute `[0, 2π)` direction to any single axis ray
/// is not possible from ring-local data. Instead the two axes are
/// reported jointly:
///
/// - `axes[0].angle` lies in `[0, π)` — the "line direction" of axis 1.
/// - `axes[1].angle` lies in `(axes[0].angle, axes[0].angle + π) ⊂ [0, 2π)`.
///
/// Together they satisfy: rotating CCW (in the usual `atan2(dy, dx)`
/// sense — note: in image pixel coordinates with y-axis pointing down,
/// this is a clockwise visual rotation) from `axes[0].angle` toward
/// `axes[1].angle` traverses a **dark** sector of the corner. The
/// second half-turn (`axes[0].angle + π → axes[1].angle + π`) crosses
/// the second dark sector; the two remaining sectors are bright.
///
/// Each axis direction is signed as a `f32` in `[0, 2π)`; the axes are
/// **not** assumed orthogonal (holds up under projective warp).
///
/// All [`crate::OrientationMethod`] variants emit axes
/// under this same convention, so consumers may compare `axes[0]`
/// (e.g. for slot-parity matching between cardinal grid neighbours)
/// across methods without method-aware translation.
///
/// # Orientation skipped
///
/// [`axes`](Self::axes) is `None` when the per-corner orientation fit was
/// skipped — the dominant per-corner cost. Consumers that derive board
/// geometry themselves can disable it (the orientation method is `None` in
/// the detector configuration); position and response are still produced.
///
/// # Position covariance
///
/// [`position_covariance`](Self::position_covariance) is the 2×2 covariance
/// of `(x, y)` in px², for whitening the corner's residuals in a
/// calibration. The `chess-corners` `Detector` fills it for every localizer
/// except the ML refiner when its uncertainty tier is
/// [`Uncertainty::Propagated`](crate::Uncertainty::Propagated) (the default
/// tier is off), and [`describe_corners`] leaves it `None` because it needs the image noise
/// level (see [`Corner::position_covariance`]).
///
/// [`describe_corners`]: crate::describe_corners
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    /// Subpixel y position in full-resolution image pixels.
    pub y: f32,

    /// Raw, **unnormalized** detector response at the detected peak. For
    /// the ChESS path this is `R = SR − DR − 16·MR`
    /// (see [`chess_response_u8`](crate::chess_response_u8)). Units are 8-bit
    /// pixel sums; data-dependent. Do not interpret it as a probability,
    /// a contrast, or a normalized strength.
    pub response: f32,

    /// The two local grid axis directions with per-axis fit-quality scores
    /// ([`AxisEstimate::sigma`], not a calibrated 1σ), or
    /// `None` when the orientation fit was skipped (orientation disabled in
    /// the detector configuration).
    pub axes: Option<[AxisEstimate; 2]>,

    /// Covariance of the position `(x, y)` in px² of this frame: the
    /// first-order scatter due to white pixel noise plus the localizer's
    /// systematic error ([`Corner::position_covariance`] gives the model).
    ///
    /// `None` when the localizer reports none (the uncertainty tier is off,
    /// the ML refiner, a saddle-point refinement on ChESS seeds that did not
    /// converge), when the image noise level is unknown (the estimate
    /// declined and none was supplied), and when the corner was refined on
    /// an upscaled copy of the input, whose interpolated noise is spatially
    /// correlated.
    pub position_covariance: Option<PositionCovariance>,
}

impl CornerDescriptor {
    /// Construct a [`CornerDescriptor`] with a fitted orientation.
    #[inline]
    #[allow(deprecated)]
    pub fn new(x: f32, y: f32, response: f32, axes: [AxisEstimate; 2]) -> Self {
        Self {
            x,
            y,
            response,
            axes: Some(axes),
            position_covariance: None,
        }
    }

    /// Construct a [`CornerDescriptor`] with the orientation fit skipped, so
    /// [`axes`](Self::axes) is `None`.
    #[inline]
    pub fn without_axes(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            response,
            axes: None,
            position_covariance: None,
        }
    }
}
