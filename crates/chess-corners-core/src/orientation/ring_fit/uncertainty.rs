//! σ-LUT post-processing for the ring fit.
//!
//! Runs the robust-seeded Gauss-Newton ring fit (in [`super::robust`])
//! and applies a piecewise-linear σ-correction LUT to the raw CRLB
//! per-axis uncertainties. Recovered angles, amplitude, and `fit_rms`
//! are left unchanged; only `sigma_theta1` and `sigma_theta2` are
//! rescaled.

use super::robust::fit;
use super::{TwoAxisFit, AMP_FLOOR};
use core::f32::consts::PI;

/// Top-level ring-fit entry. Implementation of
/// [`super::super::OrientationMethod::RingFit`].
#[inline]
pub(crate) fn fit_ring(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    let mut result = fit(samples, ring_phi);
    let m = lut_multiplier(result.rms, result.amp);
    result.sigma_theta1 = (result.sigma_theta1 * m).min(PI);
    result.sigma_theta2 = (result.sigma_theta2 * m).min(PI);
    result
}

/// Piecewise-linear LUT of `(rel_rms_threshold, multiplier)` breakpoints,
/// where `rel_rms = fit_rms / max(|amp|, AMP_FLOOR)`.
///
/// Cells with small relative residual (clean, sharp inputs) need a
/// larger correction than cells with large relative residual (heavy
/// noise / model mismatch) where the CRLB σ is already approximately
/// correct. The first and last entries pin the function for clamping;
/// intermediate entries linearly interpolate.
const LUT: &[(f32, f32)] = &[
    (0.0, 1.25),
    (0.05, 1.25),
    (0.20, 1.20),
    (1.0, 1.10),
    (5.0, 1.00),
    (f32::INFINITY, 1.00),
];

/// Return the σ-correction multiplier for a given fit by piecewise-linear
/// interpolation of [`LUT`] keyed on `fit_rms / max(|amp|, AMP_FLOOR)`.
/// Non-finite inputs clamp to the first LUT entry (the "clean fit"
/// multiplier). The trailing `(INFINITY, 1.0)` entry signals "flat to the
/// right" past the last finite breakpoint.
fn lut_multiplier(rms: f32, amp: f32) -> f32 {
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
            if a.0 == b.0 {
                return a.1;
            }
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
        assert!((lut_multiplier(0.0, 1.0) - 1.25).abs() < 1e-6);
        assert!((lut_multiplier(0.05, 1.0) - 1.25).abs() < 1e-6);
        assert!((lut_multiplier(0.20, 1.0) - 1.20).abs() < 1e-6);
        assert!((lut_multiplier(1.0, 1.0) - 1.10).abs() < 1e-6);
        assert!((lut_multiplier(5.0, 1.0) - 1.00).abs() < 1e-6);
    }

    #[test]
    fn lut_linear_interpolation_midpoint() {
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
        assert!((lut_multiplier(100.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((lut_multiplier(1e9, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lut_handles_nan_as_zero() {
        assert!((lut_multiplier(f32::NAN, 1.0) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn lut_uses_relative_residual_with_amp() {
        assert!((lut_multiplier(16.0, 80.0) - 1.20).abs() < 1e-6);
        assert!((lut_multiplier(16.0, 40.0) - 1.175).abs() < 1e-6);
    }

    #[test]
    fn lut_amp_floor_for_degenerate_fits() {
        assert!((lut_multiplier(50.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((lut_multiplier(50.0, f32::NAN) - 1.0).abs() < 1e-6);
    }
}
