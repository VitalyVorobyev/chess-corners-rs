//! Two-axis projective orientation fit at a chessboard corner.
//!
//! The detector lifts each subpixel corner to an [`AxisFitResult`] that
//! carries the two local grid-axis directions, their per-axis 1σ angular
//! uncertainties, and a residual RMS. The actual fit is pluggable via
//! [`OrientationMethod`].
//!
//! Two public entry points are provided:
//! - [`fit_axes_at_point`] (image input) — samples the ring at
//!   `(cx, cy)` and dispatches to the chosen method.
//! - [`fit_axes_from_samples`] (presampled input) — accepts the 16
//!   ring samples directly; convenient for unit tests and for the
//!   orientation benchmark when ring sampling is decoupled.
//!
//! New algorithm variants plug into the [`OrientationMethod`] enum without
//! breaking SemVer thanks to the `#[non_exhaustive]` attribute.

mod api;
mod disk_sector;
mod ring_fit;

pub use api::{fit_axes_at_point, fit_axes_from_samples, AxisFitResult};

/// Method used to fit the two grid axes at a detected corner.
///
/// The default [`Self::RingFit`] covers the vast majority of use cases. Use
/// [`Self::DiskFit`] when you need improved axis accuracy on corners with
/// strong projective skew (axis separation far from 90°).
//
// `f32` payloads disqualify the enum from `Eq` derive; we keep
// `PartialEq` only and rely on the manual variants for matching.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum OrientationMethod {
    /// Fit the parametric two-axis chessboard intensity model
    /// `I(φ) = μ + A·tanh(β·sin(φ−θ₁))·tanh(β·sin(φ−θ₂))` to 16
    /// ring samples via Gauss-Newton seeded from the 2nd-harmonic
    /// orientation. Per-axis 1σ uncertainties are calibrated by a
    /// piecewise-linear lookup table keyed on the contrast-relative
    /// residual, bringing reported sigmas closer to the empirical RMSE.
    ///
    /// This is the **default** method. Suitable for the full range of
    /// standard chessboard images.
    #[default]
    RingFit,
    /// Full-disk crossing-line estimator. Samples all image pixels in a
    /// disk around the corner center and fits two possibly non-orthogonal
    /// axes from the resulting gradient field. When local evidence is
    /// weak or the ring fit already indicates a clean orthogonal corner,
    /// falls back to [`Self::RingFit`] output transparently.
    ///
    /// Use this when standard chessboards are imaged under strong
    /// projective warp (axis separation far from 90°). Higher per-corner
    /// cost than `RingFit` (~5–10× on typical hardware), but the lazy-gate
    /// short-circuits on clean inputs so the average cost is much lower.
    DiskFit,
}

// Re-exported to crate-internal callers (e.g. the descriptor module) so
// they can still use `TwoAxisFit` after the body moved.
pub(crate) use ring_fit::TwoAxisFit;

/// Crate-internal entry into the ring fit, used by the descriptor module.
#[inline]
pub(crate) fn ring_fit_for_descriptor(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    ring_fit::fit_ring(samples, ring_phi)
}

/// Crate-internal full-disk estimator entry, used by the descriptor's
/// `OrientationMethod` dispatcher.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn disk_fit_for_descriptor(
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> TwoAxisFit {
    disk_sector::fit(img, w, h, cx, cy, radius, samples, ring_phi)
}

/// Crate-internal hook that exposes `canonicalize` only to the
/// descriptor module's own test suite.
#[cfg(test)]
#[inline]
pub(crate) fn ring_fit_canonicalize_for_test(
    theta1: f32,
    theta2: f32,
    amp: f32,
) -> (f32, f32, f32) {
    ring_fit::canonicalize(theta1, theta2, amp)
}
