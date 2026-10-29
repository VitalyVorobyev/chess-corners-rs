//! Two-axis projective orientation fit at a chessboard corner.
//!
//! The detector lifts each subpixel corner to an [`AxisFitResult`] that
//! carries the two local grid-axis directions, the standard error of
//! each axis fit, and a residual RMS. The actual fit is pluggable via
//! [`OrientationMethod`].
//!
//! [`fit_axes_at_point`] is the public entry point: it samples the
//! ring at `(cx, cy)` and dispatches to the chosen method. A
//! crate-private `fit_axes_from_samples` variant accepts pre-sampled
//! ring values directly and backs this module's own unit tests.
//!
//! New algorithm variants plug into the [`OrientationMethod`] enum without
//! breaking SemVer thanks to the `#[non_exhaustive]` attribute.

mod api;
mod descriptor;
mod disk_sector;
mod ring_fit;

use crate::detect::chess::ring::ring_offsets;
use crate::imageview::ImageView;
use descriptor::{ring_angles, sample_ring};

pub use api::{fit_axes_at_point, AxisFitResult};
pub use descriptor::describe_corners;

/// Method used to fit the two grid axes at a detected corner.
///
/// [`Self::RingFit`] is the default. [`Self::DiskFit`] samples a larger
/// local support region and is intended for corners with strong
/// projective skew (axis separation far from 90°).
///
/// All variants emit `axes[0]` and `axes[1]` under the same canonical
/// convention documented on [`crate::CornerDescriptor`]:
/// `axes[0].angle ∈ [0, π)`, `axes[1].angle ∈ (axes[0].angle,
/// axes[0].angle + π)`, and the CCW arc `(axes[0], axes[1])` is a *dark*
/// sector of the corner. Downstream consumers can therefore compare
/// `axes[0]` slot parity between corners regardless of which method
/// produced them.
//
// `f32` payloads disqualify the enum from `Eq` derive; we keep
// `PartialEq` only and rely on the manual variants for matching.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum OrientationMethod {
    /// Fit the parametric two-axis chessboard intensity model
    /// `I(φ) = μ + A·tanh(β·sin(φ−θ₁))·tanh(β·sin(φ−θ₂))` to 16
    /// ring samples via Gauss-Newton seeded from the 2nd-harmonic
    /// orientation. Suspicious local minima are retried from a small
    /// deterministic seed grid over the same 16 samples. When a
    /// radius-10 detector ring is used and the fit still looks
    /// suspicious, the canonical radius-5 ring is sampled as a cheap
    /// safety check and used when it produces a valid contrast/residual
    /// fit. Each axis reports the residual-scaled least-squares standard
    /// error of this fit ([`crate::AxisEstimate::fit_std_error`]).
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
    /// projective warp (axis separation far from 90°). It has higher
    /// per-corner cost than `RingFit`; the lazy gate keeps the full disk
    /// fit off clean orthogonal corners.
    ///
    /// Output axes use the same canonical convention as [`Self::RingFit`]
    /// — see the type-level doc comment above and
    /// [`crate::CornerDescriptor`]. When the disk fit is used, each axis
    /// reports its residual-scaled least-squares standard error over the
    /// disk pixels; a ring-fit fallback reports the ring fit's, at least
    /// 10° when the two fits disagree.
    DiskFit,
}

// Re-exported to crate-internal callers (e.g. the descriptor module) so
// they can still use `TwoAxisFit` after the body moved.
pub(crate) use ring_fit::TwoAxisFit;

/// Crate-internal entry into the ring fit, used by the descriptor module.
#[cfg(test)]
#[inline]
pub(crate) fn ring_fit_for_descriptor(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    ring_fit::fit_ring(samples, ring_phi)
}

/// Image-side RingFit entry point. Radius-10 detector rings sample
/// farther from the candidate center. On very small extreme-skew corners
/// that outer trace can cross the wrong sectors. If the outer fit already
/// looks suspicious, retry the canonical radius-5 trace and use it as a
/// cheap safety fallback.
pub(crate) fn ring_fit_for_image(
    view: ImageView<'_>,
    cx: f32,
    cy: f32,
    radius: u32,
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> TwoAxisFit {
    let outer = ring_fit::fit_ring(samples, ring_phi);
    if radius != 10 || !ring_fit::fit_is_suspicious(&outer) {
        return outer;
    }

    let inner_ring = ring_offsets(5);
    let inner_phi = ring_angles(inner_ring);
    let inner_samples = sample_ring(view.data, view.width, view.height, cx, cy, inner_ring);
    let inner = ring_fit::fit_ring(&inner_samples, &inner_phi);

    if inner.amp >= 1.0 && inner.rms.is_finite() {
        inner
    } else {
        outer
    }
}

/// Crate-internal full-disk estimator entry, used by the descriptor's
/// `OrientationMethod` dispatcher.
#[inline]
pub(crate) fn disk_fit_for_descriptor(
    view: ImageView<'_>,
    cx: f32,
    cy: f32,
    radius: u32,
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> TwoAxisFit {
    disk_sector::fit(view, cx, cy, radius, samples, ring_phi)
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

/// Test reference for the axis standard error, in f64: for the model
/// `y = μ + A·q(θ₁, θ₂)`, `model[i] = [q, ∂q/∂θ₁, ∂q/∂θ₂]` at the reported
/// axes. Solves `μ` and `A` by linear least squares at those axes and
/// returns `sqrt([(JᵀJ)⁻¹]θθ · SSR / (N − 4))` for both axes.
#[cfg(test)]
pub(crate) fn reference_std_errors(model: &[[f64; 3]], y: &[f64]) -> [f64; 2] {
    let n = y.len() as f64;
    let (sq, sy) = (
        model.iter().map(|m| m[0]).sum::<f64>(),
        y.iter().sum::<f64>(),
    );
    let sqq: f64 = model.iter().map(|m| m[0] * m[0]).sum();
    let sqy: f64 = model.iter().zip(y).map(|(m, v)| m[0] * v).sum();
    let amp = (sqy - sq * sy / n) / (sqq - sq * sq / n);
    let mu = (sy - amp * sq) / n;
    let mut jtj = [[0.0f64; 4]; 4];
    let mut ssr = 0.0;
    for (m, v) in model.iter().zip(y) {
        let j = [1.0, m[0], amp * m[1], amp * m[2]];
        for k in 0..4 {
            for l in 0..4 {
                jtj[k][l] += j[k] * j[l];
            }
        }
        ssr += (v - mu - amp * m[0]).powi(2);
    }
    // Gauss-Jordan inverse of the 4×4 normal matrix.
    let mut a = [[0.0f64; 8]; 4];
    for i in 0..4 {
        a[i][..4].copy_from_slice(&jtj[i]);
        a[i][4 + i] = 1.0;
    }
    for i in 0..4 {
        let p = (i..4)
            .max_by(|&r, &s| a[r][i].abs().total_cmp(&a[s][i].abs()))
            .expect("non-empty range");
        a.swap(i, p);
        let piv = a[i][i];
        for v in a[i].iter_mut() {
            *v /= piv;
        }
        let row = a[i];
        for (k, r) in a.iter_mut().enumerate() {
            if k != i {
                let f = r[i];
                for (x, y) in r.iter_mut().zip(row.iter()) {
                    *x -= f * y;
                }
            }
        }
    }
    let s2 = ssr / (n - 4.0);
    [(a[2][6] * s2).sqrt(), (a[3][7] * s2).sqrt()]
}
