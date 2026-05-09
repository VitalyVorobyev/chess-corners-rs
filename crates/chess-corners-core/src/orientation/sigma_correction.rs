//! V6 — post-fit σ correction for the two-axis orientation fit.
//!
//! The Phase 2 orientation benchmark showed that the per-axis 1σ
//! uncertainties returned by [`super::baseline::fit`] (derived from
//! `(JᵀJ)⁻¹`) systematically underestimate the empirical RMSE of the
//! recovered angles. The weighted-mean `z_std` across all sweeps was
//! ≈ 1.644 and the `|z|>2` outlier fraction was ≈ 12% (Gaussian target
//! ≈ 4.55%). Downstream calibration code that weights observations by
//! `1/σ²` is therefore overconfident.
//!
//! This module implements two pure post-processing variants. Both run
//! [`super::baseline::fit`] verbatim, then multiply the two reported
//! axis sigmas by a correction factor. The fit itself is unchanged:
//! `theta1`, `theta2`, `amp`, and `rms` are bit-identical to the
//! baseline output, and there is no per-corner allocation overhead.
//!
//! - [`fit_const`] applies a single user-supplied multiplier (V6a).
//! - [`fit_lut`] looks up the multiplier in a piecewise-linear LUT
//!   keyed on the contrast-relative residual `fit_rms / max(|amp|, ε)`
//!   (V6b). Using the ratio rather than the absolute `fit_rms` keeps
//!   the LUT invariant to image contrast — the same calibration table
//!   serves a clean synthetic patch (amp ≈ 80, rms ≈ 3 → ratio ≈ 0.04)
//!   and a low-contrast real image (amp ≈ 20, rms ≈ 1 → ratio ≈ 0.05).
//!   Cells with a small relative residual (clean, sharp inputs) need a
//!   larger correction; cells with a large relative residual (heavy
//!   noise / model mismatch) need very little.
//!
//! Multiplied sigmas are clamped at π so a "well-calibrated"
//! degenerate fit (`σ = π`) cannot exceed the legitimate range; a
//! truly degenerate fit (`σ ≈ π` already) stays degenerate.

use super::baseline::{self, TwoAxisFit};
use core::f32::consts::PI;

/// Apply a constant multiplier `multiplier` to both axis sigmas after
/// running the baseline fit (V6a). Negative or non-finite multipliers
/// are floored at 0 so we never invert the sign convention.
#[inline]
pub(crate) fn fit_const(samples: &[f32; 16], ring_phi: &[f32; 16], multiplier: f32) -> TwoAxisFit {
    let mut fit = baseline::fit(samples, ring_phi);
    let m = if multiplier.is_finite() {
        multiplier.max(0.0)
    } else {
        0.0
    };
    fit.sigma_theta1 = (fit.sigma_theta1 * m).min(PI);
    fit.sigma_theta2 = (fit.sigma_theta2 * m).min(PI);
    fit
}

/// Apply a piecewise-linear multiplier keyed on the contrast-relative
/// residual `fit_rms / max(|amp|, ε)` to both axis sigmas after running
/// the baseline fit (V6b). The LUT breakpoints are calibrated against
/// the Phase 2 `bench_default` sweep: clean cells (small relative
/// residual) need a ~1.7× correction; heavy-noise / model-mismatch
/// cells (large relative residual) only need ~1.0–1.1×.
#[inline]
pub(crate) fn fit_lut(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    let mut fit = baseline::fit(samples, ring_phi);
    let m = lut_multiplier(fit.rms, fit.amp);
    fit.sigma_theta1 = (fit.sigma_theta1 * m).min(PI);
    fit.sigma_theta2 = (fit.sigma_theta2 * m).min(PI);
    fit
}

/// Piecewise-linear LUT of `(rel_rms_threshold, multiplier)` breakpoints,
/// where `rel_rms = fit_rms / max(|amp|, AMP_FLOOR)`.
///
/// Calibrated against the Phase 2 `bench_default` sweep: the non-skew
/// per-cell `z_std` baseline averages ≈ 1.25, so the correction factor
/// for typical cells (rel_rms ≈ 0.05–0.20) sits near 1.20–1.25. The
/// tail tapers to 1.0 for high-residual / model-mismatched fits where
/// the CRLB σ is already roughly correct. (Axis-skew cells with
/// z_std ≫ 5 are H3 territory, not addressable via post-fit σ
/// scaling — V6 deliberately under-corrects them.) The first and last
/// entries pin the function for clamping; intermediate entries
/// linearly interpolate.
const LUT: &[(f32, f32)] = &[
    (0.0, 1.25),
    (0.05, 1.25),
    (0.20, 1.20),
    (1.0, 1.10),
    (5.0, 1.00),
    (f32::INFINITY, 1.00),
];

/// Floor on `|amp|` used when forming the relative residual. Below this
/// the fit is essentially degenerate (σ already π); the relative
/// residual jumps to the saturated end of the LUT and the multiplier
/// becomes 1.0 — i.e. the σ correction is a no-op for degenerate fits.
const AMP_FLOOR: f32 = 1.0;

/// Return the multiplier for a given fit by piecewise-linear
/// interpolation of [`LUT`] keyed on `fit_rms / max(|amp|, AMP_FLOOR)`.
/// Non-finite inputs clamp to the first LUT entry (i.e. the "clean fit"
/// multiplier).
///
/// The trailing `(INFINITY, 1.0)` entry signals "step to 1.0 above the
/// last finite breakpoint": values beyond the last finite x return
/// the entry's multiplier directly rather than a (degenerate) linear
/// interpolation across an infinite segment.
pub(crate) fn lut_multiplier(rms: f32, amp: f32) -> f32 {
    let denom = if amp.is_finite() {
        amp.abs().max(AMP_FLOOR)
    } else {
        AMP_FLOOR
    };
    let raw = if rms.is_finite() {
        rms.max(0.0) / denom
    } else {
        0.0
    };
    let r = if raw.is_finite() { raw } else { 0.0 };
    for window in LUT.windows(2) {
        let (a, b) = (window[0], window[1]);
        if r <= b.0 {
            // Same-x degenerate breakpoint: take the left endpoint.
            if a.0 == b.0 {
                return a.1;
            }
            // Trailing `(INFINITY, m)` entry: return `m` for every
            // value past the previous finite breakpoint. This is a
            // step, not a slope.
            if !b.0.is_finite() {
                return b.1;
            }
            let t = ((r - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
            return a.1 + (b.1 - a.1) * t;
        }
    }
    LUT.last().expect("LUT is non-empty by construction").1
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests use `amp = 1.0` so the relative residual equals the raw
    // `rms` argument and the existing LUT-vs-rel_rms expectations hold.

    #[test]
    fn lut_breakpoint_values() {
        // Each table entry's `multiplier` must be returned exactly when
        // we evaluate the LUT at its threshold (the first breakpoint
        // always wins for that x).
        assert!((lut_multiplier(0.0, 1.0) - 1.25).abs() < 1e-6);
        assert!((lut_multiplier(0.05, 1.0) - 1.25).abs() < 1e-6);
        assert!((lut_multiplier(0.20, 1.0) - 1.20).abs() < 1e-6);
        assert!((lut_multiplier(1.0, 1.0) - 1.10).abs() < 1e-6);
        assert!((lut_multiplier(5.0, 1.0) - 1.00).abs() < 1e-6);
    }

    #[test]
    fn lut_linear_interpolation_midpoint() {
        // rel_rms = 0.125 is halfway between 0.05 (1.25) and 0.20 (1.20) → 1.225.
        let m = lut_multiplier(0.125, 1.0);
        assert!((m - 1.225).abs() < 1e-6, "got {m}");
    }

    #[test]
    fn lut_clamps_below_first_breakpoint() {
        assert!((lut_multiplier(-0.1, 1.0) - 1.25).abs() < 1e-6);
        assert!((lut_multiplier(-1e6, 1.0) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn lut_plateaus_above_last_finite_breakpoint() {
        // Beyond rel_rms = 5.0 the LUT plateaus at 1.0 (the trailing
        // INFINITY entry signals "flat to the right").
        assert!((lut_multiplier(100.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((lut_multiplier(1e9, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lut_handles_nan_as_zero() {
        // NaN rms degrades to 0.0 (safest assumption: treat as "clean").
        assert!((lut_multiplier(f32::NAN, 1.0) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn lut_uses_relative_residual_with_amp() {
        // Bench scale: amp ~ 80, rms ~ 16 → rel_rms = 0.20 → multiplier 1.20.
        assert!((lut_multiplier(16.0, 80.0) - 1.20).abs() < 1e-6);
        // Same rms, half the amp → rel_rms = 0.40 → linear interp
        // between (0.20, 1.20) and (1.0, 1.10): t = 0.25 → 1.175.
        assert!((lut_multiplier(16.0, 40.0) - 1.175).abs() < 1e-6);
    }

    #[test]
    fn lut_amp_floor_for_degenerate_fits() {
        // amp = 0 (degenerate) — denominator floors at AMP_FLOOR = 1.0
        // so we don't divide by zero. With rms = 50 the relative
        // residual saturates the LUT at 1.0.
        assert!((lut_multiplier(50.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((lut_multiplier(50.0, f32::NAN) - 1.0).abs() < 1e-6);
    }
}
