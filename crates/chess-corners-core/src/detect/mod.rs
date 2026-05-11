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
pub mod radon;

pub(crate) use chess::detect::{count_positive_neighbors, is_local_max};
pub use chess::detect::{
    detect_corners_from_response, detect_corners_from_response_with_refiner,
    detect_peaks_from_response, find_corners_u8, find_corners_u8_with_refiner,
    merge_corners_simple, refine_corners_on_image,
};
pub use dense::{ChessBuffers, ChessDetector, DenseDetector, RadonDetector};

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
}

impl Corner {
    /// Construct a [`Corner`].
    #[inline]
    pub fn new(x: f32, y: f32, strength: f32) -> Self {
        Self { x, y, strength }
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
    /// 1σ angular uncertainty (radians), from the fit's covariance.
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
/// All [`crate::orientation::OrientationMethod`] variants emit axes
/// under this same convention, so consumers may compare `axes[0]`
/// (e.g. for slot-parity matching between cardinal grid neighbours)
/// across methods without method-aware translation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    /// Subpixel y position in full-resolution image pixels.
    pub y: f32,

    /// Raw, **unnormalized** detector response at the detected peak. For
    /// the ChESS path this is `R = SR − DR − 16·MR`
    /// (see [`chess::response::chess_response_u8`]). Units are 8-bit
    /// pixel sums; data-dependent. Do not interpret it as a probability,
    /// a contrast, or a normalized strength.
    pub response: f32,

    /// Bright/dark amplitude (`|A|`, ≥ 0) recovered by the two-axis
    /// orientation fit (see [`crate::orientation`]). Units are gray
    /// levels. Larger means a stronger bright/dark separation at the
    /// ring radius. This is an independent quantity from
    /// [`Self::response`] — they are computed by different estimators
    /// and must not be compared against each other or against the same
    /// threshold.
    pub contrast: f32,

    /// RMS fit residual of the two-axis intensity model (gray levels).
    /// Smaller = tighter match to an ideal chessboard corner.
    pub fit_rms: f32,

    /// The two local grid axis directions with per-axis 1σ precision.
    pub axes: [AxisEstimate; 2],
}

impl CornerDescriptor {
    /// Construct a [`CornerDescriptor`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f32,
        y: f32,
        response: f32,
        contrast: f32,
        fit_rms: f32,
        axes: [AxisEstimate; 2],
    ) -> Self {
        Self {
            x,
            y,
            response,
            contrast,
            fit_rms,
            axes,
        }
    }
}
