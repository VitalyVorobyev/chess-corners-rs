//! Two-axis projective orientation fit at a chessboard corner.
//!
//! The detector lifts each subpixel corner to a [`AxisFitResult`] that
//! carries the two local grid-axis directions, their per-axis 1σ
//! angular uncertainties, and a residual RMS. The actual fit is
//! pluggable via [`OrientationMethod`]; the [`OrientationMethod::Baseline`]
//! variant reproduces the legacy 16-sample Gauss-Newton fit
//! bit-identically.
//!
//! Two public entry points are provided:
//! - [`fit_axes_at_point`] (image input) — samples the ring at
//!   `(cx, cy)` and dispatches to the chosen method.
//! - [`fit_axes_from_samples`] (presampled input) — accepts the 16
//!   ring samples directly; convenient for unit tests and for the
//!   orientation benchmark when ring sampling is decoupled.
//!
//! Future algorithm variants (adaptive β, LM damping, structure-tensor
//! seeding, full-disk fits, two-ring fits, σ correction, …) plug into
//! the [`OrientationMethod`] enum without breaking SemVer thanks to the
//! `#[non_exhaustive]` attribute.

mod adaptive_beta;
mod api;
mod baseline;
mod disk_sector;
mod sigma_correction;

pub use api::{fit_axes_at_point, fit_axes_from_samples, AxisFitResult};

/// Method used to fit the two grid axes at a detected corner.
///
/// `Baseline` is the legacy 2nd-harmonic-seeded Gauss-Newton fit on a
/// 16-sample ring. Other variants are added in subsequent versions.
///
/// V6 (`SigmaCorrectionConst` / `SigmaCorrectionLut`) is a pure
/// post-processing pass: the underlying fit is always
/// [`OrientationMethod::Baseline`], and only the per-axis 1σ
/// uncertainties are scaled afterwards. Recovered angles, amplitude,
/// and `fit_rms` are bit-identical to baseline.
//
// `f32` payloads disqualify the enum from `Eq` derive; we keep
// `PartialEq` only and rely on the manual variants for matching.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum OrientationMethod {
    /// Legacy 16-sample Gauss-Newton fit. Bit-identical to the
    /// pre-refactor `fit_two_axes` behaviour.
    #[default]
    Baseline,
    /// V6a — apply a constant scalar `multiplier` to both axis sigmas
    /// after running the baseline fit. The fit itself is unchanged;
    /// only `sigma_theta{1,2}` are scaled (and clamped at π).
    ///
    /// Useful as a sanity baseline against the LUT variant. The
    /// recommended starting value is the weighted-mean `z_std` from
    /// the Phase 2 baseline sweep (≈ 1.644).
    SigmaCorrectionConst {
        /// Multiplier applied to both axis sigmas.
        multiplier: f32,
    },
    /// V6b — apply a piecewise-linear multiplier keyed on the fit
    /// residual (`fit_rms`) to both axis sigmas after running the
    /// baseline fit. Better than the constant variant on
    /// heterogeneous bench data because heavy-noise cells (large
    /// `fit_rms`) need a much smaller correction than clean cells.
    SigmaCorrectionLut,
    /// V1 — promote the tanh slope `β` from a fixed constant to a 5th
    /// Gauss-Newton parameter so the fit adapts to the local edge
    /// sharpness implied by the upstream blur. Phase 2's blur sweep
    /// exposed a U-shape in baseline RMSE because the hardcoded
    /// `β = 4` is only optimal near `σ_blur ≈ 2`. The recovered β is
    /// not surfaced through the public result; it only feeds the σ
    /// extraction.
    AdaptiveBeta,
    /// Full-disk crossing-line estimator. Uses the existing detector
    /// only for the corner center, then fits two possibly
    /// non-orthogonal axes from all image pixels in a disk around that
    /// center. If the disk model is weak or invalid, it falls back to
    /// [`OrientationMethod::SigmaCorrectionLut`].
    FullDiskSector,
}

/// Shared Gauss-Newton solver settings for orientation variants.
///
/// The current default reproduces the constants previously hard-coded
/// in the baseline fit (`max_iters = 6`, `tol = 1e-4`,
/// `max_step = 0.5`). Future variants that share the same outer GN
/// loop can read these values rather than re-declaring them.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct OrientationSolver {
    /// Maximum Gauss-Newton iterations.
    pub max_iters: u32,
    /// Convergence threshold on angular updates (radians).
    pub tol: f32,
    /// Maximum angular step per iteration (radians).
    pub max_step: f32,
}

impl Default for OrientationSolver {
    fn default() -> Self {
        // Mirror the current `descriptor.rs` constants exactly.
        Self {
            max_iters: 6,
            tol: 1e-4,
            max_step: 0.5,
        }
    }
}

// Re-exported to crate-internal callers (e.g. the descriptor shim) so
// they can still talk about a `TwoAxisFit` after the body moved.
pub(crate) use baseline::TwoAxisFit;

/// Crate-internal entry into the baseline fit, used by the descriptor
/// shim so that `corners_to_descriptors` keeps producing
/// bit-identical results after the structural move.
#[inline]
pub(crate) fn baseline_fit_for_descriptor(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    baseline::fit(samples, ring_phi)
}

/// Crate-internal V6a (constant-multiplier) entry, used by the
/// descriptor's `OrientationMethod` dispatcher.
#[inline]
pub(crate) fn sigma_correction_const_fit_for_descriptor(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    multiplier: f32,
) -> TwoAxisFit {
    sigma_correction::fit_const(samples, ring_phi, multiplier)
}

/// Crate-internal V6b (LUT-based) entry, used by the descriptor's
/// `OrientationMethod` dispatcher.
#[inline]
pub(crate) fn sigma_correction_lut_fit_for_descriptor(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> TwoAxisFit {
    sigma_correction::fit_lut(samples, ring_phi)
}

/// Crate-internal V1 (adaptive-β) entry, used by the descriptor's
/// `OrientationMethod` dispatcher.
#[inline]
pub(crate) fn adaptive_beta_fit_for_descriptor(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> TwoAxisFit {
    adaptive_beta::fit(samples, ring_phi)
}

/// Crate-internal full-disk estimator entry, used by the descriptor's
/// `OrientationMethod` dispatcher. This method needs both the ring
/// samples (for its sigma-LUT fallback) and the original image disk.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn full_disk_sector_fit_for_descriptor(
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
/// descriptor module's own test suite. The function lives here, in
/// the orientation module, so the math has a single home; this
/// indirection keeps the existing tests untouched after the move.
#[cfg(test)]
#[inline]
pub(crate) fn baseline_canonicalize_for_test(
    theta1: f32,
    theta2: f32,
    amp: f32,
) -> (f32, f32, f32) {
    baseline::canonicalize(theta1, theta2, amp)
}

/// Test-only export of the V1 5×5 linear solver, so the integration
/// test in `tests/orientation_adaptive_beta.rs` can exercise it on a
/// hand-constructed PSD system without going through a full ring
/// fit. Production code routes through [`adaptive_beta::fit`] only.
#[doc(hidden)]
#[inline]
pub fn __test_solve_5x5(a: &[[f32; 5]; 5], b: &[f32; 5]) -> Option<[f32; 5]> {
    adaptive_beta::solve_5x5(a, b)
}
