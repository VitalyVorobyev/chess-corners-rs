//! Tests for the V1 [`OrientationMethod::AdaptiveBeta`] variant.
//!
//! Verifies four contracts:
//! - **Sharp / soft edge recovery**: synthetic rings rendered with
//!   non-default `β_truth` should be recovered to within an angle
//!   tolerance, with the fit's β landing in the expected envelope.
//! - **β=4 consistency with baseline**: when the input matches the
//!   baseline's hardcoded `β = 4`, V1 should converge to angles
//!   essentially indistinguishable from the baseline output.
//! - **β clamp**: a saturated input (`β_truth = 20`) should not produce
//!   NaN; the fit clamps β at `BETA_MAX = 16`.
//! - **Degenerate ring**: a flat ring still yields a degenerate fit
//!   (`σ ≈ π`), matching the baseline contract.
//!
//! Plus a unit test for the 5×5 solver using a hand-constructed
//! positive-definite system.

use chess_corners_core::orientation::{fit_axes_from_samples, OrientationMethod};
use chess_corners_core::ring::ring_offsets;
use core::f32::consts::PI;

fn eval_model(phi: f32, beta: f32, mu: f32, amp: f32, theta1: f32, theta2: f32) -> f32 {
    let h1 = (beta * (phi - theta1).sin()).tanh();
    let h2 = (beta * (phi - theta2).sin()).tanh();
    mu + amp * h1 * h2
}

fn ring_angles(ring: &[(i32, i32); 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        out[i] = (dy as f32).atan2(dx as f32);
    }
    out
}

fn synthetic_ring(
    beta: f32,
    mu: f32,
    amp: f32,
    theta1: f32,
    theta2: f32,
) -> ([f32; 16], [f32; 16]) {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let mut samples = [0.0f32; 16];
    for i in 0..16 {
        samples[i] = eval_model(phi[i], beta, mu, amp, theta1, theta2);
    }
    (samples, phi)
}

/// Smallest distance between two angles modulo π (line-direction
/// metric — the model itself is π-symmetric in each axis).
fn angle_err_mod_pi(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(PI);
    d.min(PI - d)
}

/// Recover the fitted `β` from a converged AdaptiveBeta result by
/// solving a tiny inverse problem: given the recovered (θ₁, θ₂, A, μ)
/// and the input samples, find the β that minimises the SSR via a
/// short golden-section search. We do this rather than surface β
/// through the public API because the plan keeps the recovered β
/// internal to the fit (it only feeds σ extraction).
fn estimate_beta_from_fit(
    samples: &[f32; 16],
    phi: &[f32; 16],
    mu: f32,
    amp: f32,
    theta1: f32,
    theta2: f32,
) -> f32 {
    let ssr = |beta: f32| -> f32 {
        let mut s = 0.0f32;
        for i in 0..16 {
            let pred = eval_model(phi[i], beta, mu, amp, theta1, theta2);
            let r = samples[i] - pred;
            s += r * r;
        }
        s
    };
    // Golden-section search on [0.5, 16].
    let mut lo = 0.5f32;
    let mut hi = 16.0f32;
    let phi_g = (5.0f32.sqrt() - 1.0) / 2.0;
    let mut x1 = hi - phi_g * (hi - lo);
    let mut x2 = lo + phi_g * (hi - lo);
    let mut f1 = ssr(x1);
    let mut f2 = ssr(x2);
    for _ in 0..60 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi_g * (hi - lo);
            f1 = ssr(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi_g * (hi - lo);
            f2 = ssr(x2);
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn sharp_edge_recovery() {
    // β_truth = 10 — very sharp tanh, well above the baseline's β = 4.
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(10.0, 128.0, 80.0, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::AdaptiveBeta);

    let err1 = angle_err_mod_pi(fit.theta1, t1).to_degrees();
    let err2 = angle_err_mod_pi(fit.theta2, t2).to_degrees();
    assert!(err1 < 0.5, "theta1 err {err1}° (fit theta1 {})", fit.theta1);
    assert!(err2 < 0.5, "theta2 err {err2}° (fit theta2 {})", fit.theta2);

    let beta_est = estimate_beta_from_fit(&samples, &phi, 128.0, fit.amp, fit.theta1, fit.theta2);
    assert!(
        (7.0..=14.0).contains(&beta_est),
        "recovered β {beta_est} not in [7, 14] (sharp edge)"
    );
}

#[test]
fn soft_edge_recovery() {
    // β_truth = 2 — soft tanh, below the baseline's β = 4.
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(2.0, 128.0, 80.0, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::AdaptiveBeta);

    let err1 = angle_err_mod_pi(fit.theta1, t1).to_degrees();
    let err2 = angle_err_mod_pi(fit.theta2, t2).to_degrees();
    assert!(err1 < 1.0, "theta1 err {err1}° (fit theta1 {})", fit.theta1);
    assert!(err2 < 1.0, "theta2 err {err2}° (fit theta2 {})", fit.theta2);

    let beta_est = estimate_beta_from_fit(&samples, &phi, 128.0, fit.amp, fit.theta1, fit.theta2);
    assert!(
        (1.0..=3.5).contains(&beta_est),
        "recovered β {beta_est} not in [1.0, 3.5] (soft edge)"
    );
}

#[test]
fn beta_default_consistent_with_baseline() {
    // β_truth = 4 matches baseline's hardcoded slope, so V1 should
    // converge to roughly the same angles. Allow 0.05° slop.
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(4.0, 128.0, 80.0, t1, t2);
    let base = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    let v1 = fit_axes_from_samples(&samples, &phi, OrientationMethod::AdaptiveBeta);

    let dt1 = angle_err_mod_pi(v1.theta1, base.theta1).to_degrees();
    let dt2 = angle_err_mod_pi(v1.theta2, base.theta2).to_degrees();
    assert!(
        dt1 < 0.05,
        "theta1 mismatch {dt1}° (v1 {} baseline {})",
        v1.theta1,
        base.theta1
    );
    assert!(
        dt2 < 0.05,
        "theta2 mismatch {dt2}° (v1 {} baseline {})",
        v1.theta2,
        base.theta2
    );

    let beta_est = estimate_beta_from_fit(&samples, &phi, 128.0, v1.amp, v1.theta1, v1.theta2);
    assert!(
        (3.5..=4.5).contains(&beta_est),
        "recovered β {beta_est} not in [3.5, 4.5] for β=4 input"
    );
}

#[test]
fn beta_clamp_and_no_nan() {
    // β_truth = 20 saturates beyond BETA_MAX = 16; the fit must clamp
    // (rather than diverge to NaN) and still report finite angles.
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(20.0, 128.0, 80.0, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::AdaptiveBeta);

    assert!(fit.theta1.is_finite(), "theta1 not finite: {}", fit.theta1);
    assert!(fit.theta2.is_finite(), "theta2 not finite: {}", fit.theta2);
    assert!(fit.amp.is_finite(), "amp not finite: {}", fit.amp);
    assert!(fit.rms.is_finite(), "rms not finite: {}", fit.rms);
    assert!(
        fit.sigma_theta1.is_finite(),
        "sigma1 not finite: {}",
        fit.sigma_theta1
    );
    assert!(
        fit.sigma_theta2.is_finite(),
        "sigma2 not finite: {}",
        fit.sigma_theta2
    );

    let beta_est = estimate_beta_from_fit(&samples, &phi, 128.0, fit.amp, fit.theta1, fit.theta2);
    // The fit clamps β at 16; the SSR-minimising β recovered from the
    // post-fit parameters must therefore sit at or just below the
    // clamp boundary (the golden search returns the upper bound when
    // SSR is monotonically decreasing through the interval).
    assert!(
        (14.0..=16.0).contains(&beta_est),
        "recovered β {beta_est} should sit at/near BETA_MAX (16)"
    );
}

#[test]
fn degenerate_flat_ring() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0f32; 16];
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::AdaptiveBeta);

    // Same degenerate sentinel as baseline: σ ≈ π on both axes.
    assert!(
        (fit.sigma_theta1 - PI).abs() < 1e-6,
        "sigma1 {} ≠ π",
        fit.sigma_theta1
    );
    assert!(
        (fit.sigma_theta2 - PI).abs() < 1e-6,
        "sigma2 {} ≠ π",
        fit.sigma_theta2
    );
    // amp = 0 by the degenerate path.
    assert!(
        fit.amp.abs() < 1e-6,
        "amp {} ≠ 0 in degenerate fit",
        fit.amp
    );
}

#[test]
fn solve_5x5_matches_known_solution() {
    use chess_corners_core::orientation::__test_solve_5x5;
    // PSD (Hermitian, positive-definite) test matrix: A = LᵀL where
    // L is a hand-picked lower-triangular f32 matrix.
    let l: [[f32; 5]; 5] = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 1.2, 0.0, 0.0, 0.0],
        [0.2, -0.3, 0.9, 0.0, 0.0],
        [-0.4, 0.1, 0.6, 1.3, 0.0],
        [0.7, 0.0, -0.2, 0.4, 1.1],
    ];
    let mut a = [[0f32; 5]; 5];
    for (i, a_row) in a.iter_mut().enumerate() {
        for (j, a_ij) in a_row.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for l_row in &l {
                // A = LᵀL → a[i][j] = Σₖ L[k][i]·L[k][j]
                s += l_row[i] * l_row[j];
            }
            *a_ij = s;
        }
    }
    let x_known: [f32; 5] = [1.5, -2.0, 0.75, 3.25, -0.5];
    let mut b = [0f32; 5];
    for (i, b_i) in b.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (j, &xj) in x_known.iter().enumerate() {
            s += a[i][j] * xj;
        }
        *b_i = s;
    }
    let x = __test_solve_5x5(&a, &b).expect("solver returned None");
    for (i, (&xi, &xki)) in x.iter().zip(x_known.iter()).enumerate() {
        let err = (xi - xki).abs();
        assert!(err < 1e-4, "x[{i}] = {xi} expected {xki} err {err}");
    }
}
