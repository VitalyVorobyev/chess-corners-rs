//! Public API for sampling and fitting axes at a single point.
//!
//! Two entry points:
//!
//! - [`fit_axes_at_point`] samples the 16-point ChESS ring around an
//!   image point and dispatches to the chosen [`OrientationMethod`].
//!   This is the workhorse used by the descriptor pipeline and by the
//!   orientation benchmark.
//! - [`fit_axes_from_samples`] takes pre-sampled ring values directly,
//!   which is convenient for unit tests that don't want to construct a
//!   real image. Both routes converge on the same dispatcher.

use super::descriptor::{ring_angles, sample_ring};
use super::{disk_sector, ring_fit, ring_fit_for_image, OrientationMethod};
use crate::detect::chess::ring::ring_offsets;

/// Result of a two-axis orientation fit at a single corner.
///
/// This is the public mirror of the (crate-private) `TwoAxisFit` used
/// by the descriptor path. All [`OrientationMethod`] variants populate
/// the same fields.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct AxisFitResult {
    /// Bright/dark amplitude `|A|` (≥ 0) recovered by the fit. Units
    /// are gray levels.
    pub amp: f32,
    /// First axis direction, radians in `[0, π)` (line-direction
    /// representative — see [`crate::CornerDescriptor`]).
    pub theta1: f32,
    /// Second axis direction, radians in `(theta1, theta1 + π) ⊂ [0, 2π)`.
    pub theta2: f32,
    /// 1σ angular uncertainty for `theta1` (radians).
    pub sigma_theta1: f32,
    /// 1σ angular uncertainty for `theta2` (radians).
    pub sigma_theta2: f32,
    /// RMS fit residual of the two-axis intensity model (gray levels).
    pub rms: f32,
}

impl From<ring_fit::TwoAxisFit> for AxisFitResult {
    #[inline]
    fn from(v: ring_fit::TwoAxisFit) -> Self {
        Self {
            amp: v.amp,
            theta1: v.theta1,
            theta2: v.theta2,
            sigma_theta1: v.sigma_theta1,
            sigma_theta2: v.sigma_theta2,
            rms: v.rms,
        }
    }
}

/// Sample the 16-point ChESS ring at `(cx, cy)` with `radius` and run
/// the chosen orientation method.
///
/// Public so the orientation benchmark can drive the fit directly
/// without going through the detection pipeline.
pub fn fit_axes_at_point(
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
    method: OrientationMethod,
) -> AxisFitResult {
    let ring = ring_offsets(radius);
    let ring_phi = ring_angles(ring);
    let samples = sample_ring(img, w, h, cx, cy, ring);
    match method {
        OrientationMethod::RingFit => {
            ring_fit_for_image(img, w, h, cx, cy, radius, &samples, &ring_phi).into()
        }
        OrientationMethod::DiskFit => {
            disk_sector::fit(img, w, h, cx, cy, radius, &samples, &ring_phi).into()
        }
    }
}

/// Run the chosen orientation method on pre-sampled ring values.
///
/// Use this when you already have the 16 ring samples in hand (e.g.
/// from a custom sampler or a synthetic test) and only need the fit.
/// Image-dependent variants will fall back to the ring-only
/// [`OrientationMethod::RingFit`] result when invoked through this
/// helper.
pub fn fit_axes_from_samples(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    method: OrientationMethod,
) -> AxisFitResult {
    match method {
        OrientationMethod::RingFit => ring_fit::fit_ring(samples, ring_phi).into(),
        // DiskFit needs image data; fall back to RingFit when not available.
        OrientationMethod::DiskFit => ring_fit::fit_ring(samples, ring_phi).into(),
    }
}
