//! Tests for the V6 [`OrientationMethod::SigmaCorrectionConst`] /
//! [`OrientationMethod::SigmaCorrectionLut`] post-fit σ-correction
//! variants.
//!
//! These tests verify three contracts:
//! - **Pure post-processing**: angles, amplitude, and `fit_rms` must
//!   match the baseline output bit-for-bit.
//! - **Sigma scaling**: the constant variant must multiply both axis
//!   sigmas by the requested factor (clamped at π); the LUT variant
//!   must follow the documented breakpoints.
//! - **Degenerate-fit propagation**: a flat-ring input still returns a
//!   degenerate fit (`σ ≈ π`) under either variant; the multiplier
//!   must not rescue it.

use chess_corners_core::orientation::{fit_axes_from_samples, AxisFitResult, OrientationMethod};
use chess_corners_core::ring::ring_offsets;
use core::f32::consts::{FRAC_PI_4, PI};

const TANH_BETA: f32 = 4.0;

fn eval_model(phi: f32, mu: f32, amp: f32, theta1: f32, theta2: f32) -> f32 {
    let h1 = (TANH_BETA * (phi - theta1).sin()).tanh();
    let h2 = (TANH_BETA * (phi - theta2).sin()).tanh();
    mu + amp * h1 * h2
}

fn ring_angles(ring: &[(i32, i32); 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        out[i] = (dy as f32).atan2(dx as f32);
    }
    out
}

fn synthetic_ring(mu: f32, amp: f32, theta1: f32, theta2: f32) -> ([f32; 16], [f32; 16]) {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let mut samples = [0.0f32; 16];
    for i in 0..16 {
        samples[i] = eval_model(phi[i], mu, amp, theta1, theta2);
    }
    (samples, phi)
}

/// Compare two `AxisFitResult`s for bit-exact equality on the
/// fit-related fields (angles, amplitude, residual). Sigma fields are
/// expected to differ.
fn assert_fit_fields_equal(a: &AxisFitResult, b: &AxisFitResult) {
    assert_eq!(a.theta1.to_bits(), b.theta1.to_bits(), "theta1");
    assert_eq!(a.theta2.to_bits(), b.theta2.to_bits(), "theta2");
    assert_eq!(a.amp.to_bits(), b.amp.to_bits(), "amp");
    assert_eq!(a.rms.to_bits(), b.rms.to_bits(), "rms");
}

#[test]
fn const_multiplier_scales_sigmas_only() {
    // Clean axis-aligned corner: baseline returns small (well below π)
    // sigmas. V6a with multiplier 1.5 must scale both sigmas exactly
    // 1.5× while leaving the rest of the fit untouched.
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let base = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    let v6a = fit_axes_from_samples(
        &samples,
        &phi,
        OrientationMethod::SigmaCorrectionConst { multiplier: 1.5 },
    );

    // Angles, amp, rms: bit-identical.
    assert_fit_fields_equal(&base, &v6a);

    // The clean fit's sigmas are ≪ π, so no clamp.
    assert!(base.sigma_theta1 * 1.5 < PI);
    assert!(base.sigma_theta2 * 1.5 < PI);
    let expected1 = (base.sigma_theta1 * 1.5).min(PI);
    let expected2 = (base.sigma_theta2 * 1.5).min(PI);
    assert!(
        (v6a.sigma_theta1 - expected1).abs() < 1e-7,
        "got {} expected {}",
        v6a.sigma_theta1,
        expected1
    );
    assert!(
        (v6a.sigma_theta2 - expected2).abs() < 1e-7,
        "got {} expected {}",
        v6a.sigma_theta2,
        expected2
    );
}

#[test]
fn const_multiplier_clamps_at_pi() {
    // Build a clean fit that yields some finite sigma, then apply a
    // multiplier large enough to push past π. Result must clamp.
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let base = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    assert!(base.sigma_theta1 > 0.0 && base.sigma_theta1.is_finite());

    let big = (PI / base.sigma_theta1) * 10.0; // 10× the value that hits π
    let fit = fit_axes_from_samples(
        &samples,
        &phi,
        OrientationMethod::SigmaCorrectionConst { multiplier: big },
    );
    assert!((fit.sigma_theta1 - PI).abs() < 1e-6);
    assert!((fit.sigma_theta2 - PI).abs() < 1e-6);
    // Angles, amp, rms unchanged.
    assert_fit_fields_equal(&base, &fit);
}

#[test]
fn lut_variant_uses_lut_for_clean_cells() {
    // Clean fit → fit_rms ≈ 0 → LUT multiplier 1.25 (first breakpoint).
    // We verify by comparing against the const-multiplier variant set
    // to the same value: their sigmas should match (both apply the
    // same scaling on top of the same baseline).
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let base = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    let lut = fit_axes_from_samples(&samples, &phi, OrientationMethod::SigmaCorrectionLut);
    let const_clean = fit_axes_from_samples(
        &samples,
        &phi,
        OrientationMethod::SigmaCorrectionConst { multiplier: 1.25 },
    );

    assert_fit_fields_equal(&base, &lut);
    assert!(
        base.rms < 0.05,
        "baseline rms {} not in clean regime",
        base.rms
    );
    assert!(
        (lut.sigma_theta1 - const_clean.sigma_theta1).abs() < 1e-6,
        "lut sigma1={}, const(1.25) sigma1={}",
        lut.sigma_theta1,
        const_clean.sigma_theta1
    );
    assert!(
        (lut.sigma_theta2 - const_clean.sigma_theta2).abs() < 1e-6,
        "lut sigma2={}, const(1.25) sigma2={}",
        lut.sigma_theta2,
        const_clean.sigma_theta2
    );
    // And the LUT variant clearly inflated the baseline sigmas.
    assert!(lut.sigma_theta1 > base.sigma_theta1);
    assert!(lut.sigma_theta2 > base.sigma_theta2);
}

#[test]
fn degenerate_fit_remains_degenerate_under_const() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0f32; 16];
    let base = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    // Baseline degenerate sentinel: sigmas pinned at π.
    assert!((base.sigma_theta1 - PI).abs() < 1e-6);
    assert!((base.sigma_theta2 - PI).abs() < 1e-6);

    let v6a = fit_axes_from_samples(
        &samples,
        &phi,
        OrientationMethod::SigmaCorrectionConst { multiplier: 1.644 },
    );
    let v6b = fit_axes_from_samples(&samples, &phi, OrientationMethod::SigmaCorrectionLut);

    // Multiplier ≥ 1 on a value already at π must clamp back at π.
    // The variants must not turn a degenerate fit into a
    // "non-degenerate" one with a smaller σ.
    assert!((v6a.sigma_theta1 - PI).abs() < 1e-6);
    assert!((v6a.sigma_theta2 - PI).abs() < 1e-6);
    assert!((v6b.sigma_theta1 - PI).abs() < 1e-6);
    assert!((v6b.sigma_theta2 - PI).abs() < 1e-6);

    // Angles + amp + rms still match the baseline output.
    assert_fit_fields_equal(&base, &v6a);
    assert_fit_fields_equal(&base, &v6b);
}

#[test]
fn const_multiplier_negative_floors_at_zero() {
    // Negative multipliers are clamped to 0 (we never want to invert
    // sigma's sign convention or reproduce a NaN through the math).
    let (samples, phi) = synthetic_ring(128.0, 80.0, FRAC_PI_4, 3.0 * FRAC_PI_4);
    let fit = fit_axes_from_samples(
        &samples,
        &phi,
        OrientationMethod::SigmaCorrectionConst { multiplier: -2.5 },
    );
    assert!(fit.sigma_theta1.abs() < 1e-7);
    assert!(fit.sigma_theta2.abs() < 1e-7);
}
