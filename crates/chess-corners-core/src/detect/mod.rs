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

/// Model-error floor of a refined position, in px per axis: added as `f²`
/// to each diagonal entry of [`Corner::position_covariance`]. It covers the
/// phase-dependent error the re-centred saddle-point fit keeps, largest on
/// boards rotated 30–45° from the pixel grid. Chosen to minimise the worst
/// |ln σ̂₀²| of the whitened error on synthetic camera-formed boards (lens
/// blur σ 0.5–2.5 px before pixel integration; 8–12 px cells, rotated 0–45°
/// or sheared; noise σ 1–8 gray levels): 0.56, against 1.40 at 0.002 px.
/// Grid-aligned boards at the lowest noise are then conservative
/// (σ̂₀² ≈ 0.6).
pub(crate) const POSITION_FLOOR_PX: f32 = 0.006;

/// Lower bound on the pixel-noise variance used for a covariance, in gray
/// level²: the variance of rounding to 8 bits.
pub(crate) const MIN_NOISE_VAR: f32 = 1.0 / 12.0;

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
    /// The refiner's position covariance per unit image-noise variance (px²
    /// per gray-level²), copied from
    /// [`RefineResult::covariance_per_noise_variance`](crate::RefineResult::covariance_per_noise_variance)
    /// by [`refine_corners_on_image`] and by the Radon detector's image
    /// refinement ([`DenseDetector::refine_peaks_on_image`](crate::DenseDetector::refine_peaks_on_image));
    /// `None` when the refiner reports none.
    /// [`Corner::position_covariance`] turns it into px².
    pub covariance_per_noise_variance: Option<PositionCovariance>,
}

impl Corner {
    /// Construct a [`Corner`] without a covariance.
    #[inline]
    pub fn new(x: f32, y: f32, strength: f32) -> Self {
        Self {
            x,
            y,
            strength,
            covariance_per_noise_variance: None,
        }
    }

    /// Position covariance in px² for an image whose pixels carry white
    /// noise of standard deviation `noise_sigma` gray levels (for example
    /// from [`estimate_noise_sigma_u8`](crate::estimate_noise_sigma_u8)):
    /// `σₙ²·C + f²·I`, with `C` =
    /// [`covariance_per_noise_variance`](Self::covariance_per_noise_variance)
    /// and `f = 0.006` px per axis. `σₙ²` is at least `1/12` gray level²,
    /// the variance of rounding to 8 bits, so a smoothed or compressed image
    /// whose measured noise is lower does not get an overconfident
    /// covariance.
    ///
    /// `σₙ²·C` is the first-order noise scatter; `f` is a model-error floor
    /// for the small error the refiner keeps where the corner sits between
    /// pixels. On synthetic boards formed with a lens blur of σ ≥ 0.5 px,
    /// 8–12 px cells and any rotation, whitened errors have variance
    /// 0.6–1.75 at noise σ 1 and 0.7–1.15 at σ ≥ 4 (conservative on
    /// grid-aligned boards, overconfident at 30–45° rotations when the noise
    /// is lowest). Ideally sharp optics (blur σ below about 0.5 px) leave up
    /// to 0.02 px of phase-dependent error on boards rotated 2–15° or 45°,
    /// and smaller, sheared or strongly blurred cells more, which this floor
    /// does not cover. Valid only for the image the corner was refined on. With spatially correlated noise
    /// (demosaiced, compressed or resampled images) `σₙ` from
    /// [`estimate_noise_sigma_u8`](crate::estimate_noise_sigma_u8) is the
    /// noise as that estimator's operator sees it, and the absolute scale of
    /// the covariance is approximate; covariances of corners in the same
    /// image remain meaningful relative to each other. `None` when the
    /// corner carries no covariance.
    #[inline]
    pub fn position_covariance(&self, noise_sigma: f32) -> Option<PositionCovariance> {
        let var = (noise_sigma * noise_sigma).max(MIN_NOISE_VAR);
        let floor2 = POSITION_FLOOR_PX * POSITION_FLOOR_PX;
        self.covariance_per_noise_variance
            .map(|c| PositionCovariance::new(var * c.xx + floor2, var * c.xy, var * c.yy + floor2))
    }
}

/// Direction of one local grid axis with its 1σ angular uncertainty.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct AxisEstimate {
    /// Axis direction, radians in `[0, 2π)`.
    ///
    /// See [`CornerDescriptor`] for the joint polarity convention.
    pub angle: f32,
    /// 1σ angular uncertainty estimate (radians).
    ///
    /// How it is derived depends on the orientation method. `RingFit`
    /// scales the Gauss-Newton covariance of the fit by the residual
    /// variance and an empirical calibration factor; `DiskFit` reports a
    /// residual-based estimate, never below a method-specific floor.
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
/// calibration. Only the saddle-point refiner reports one; the
/// `chess-corners` `Detector` fills it, and [`describe_corners`] leaves it
/// `None` because it needs the image noise level (see
/// [`Corner::position_covariance`]).
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

    /// The two local grid axis directions with per-axis 1σ precision, or
    /// `None` when the orientation fit was skipped (orientation disabled in
    /// the detector configuration).
    pub axes: Option<[AxisEstimate; 2]>,

    /// Covariance of the position `(x, y)` in px² of this frame: the
    /// first-order scatter due to white pixel noise plus a small model-error
    /// floor ([`Corner::position_covariance`] gives the model).
    ///
    /// `None` unless the corner was refined by the saddle-point refiner, its
    /// re-centred refinement converged, the image noise level is known
    /// (estimated or supplied), and the corner was refined on the input
    /// image itself — not on an upscaled copy, whose interpolated noise is
    /// spatially correlated.
    pub position_covariance: Option<PositionCovariance>,
}

impl CornerDescriptor {
    /// Construct a [`CornerDescriptor`] with a fitted orientation.
    #[inline]
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
