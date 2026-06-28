//! Tests for [`OrientationMethod::RingFit`].
//!
//! Covers:
//! - Bit-parity of the recovered angles, amplitude, and residual against
//!   synthetic noiseless inputs (tight 5 mrad tolerance).
//! - Canonicalization invariants.
//! - Degenerate (flat-ring) sentinel behaviour.
//! - Image-input path matches pre-sampled path (up to u8 quantisation).
//! - σ-LUT: RingFit scales both axis sigmas via the piecewise-linear LUT
//!   while leaving angles, amp, and rms unchanged from the raw robust fit.
//! - σ-LUT degenerate propagation: a flat ring remains degenerate under
//!   RingFit.

use chess_corners_core::unstable::ring_offsets;
use chess_corners_core::{fit_axes_at_point, fit_axes_from_samples, OrientationMethod};
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

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

/// Smallest signed angular distance between `a` and `b` modulo π.
fn angle_err_mod_pi(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(PI);
    d.min(PI - d)
}

/// Max angular error between two unordered axis pairs (accounts for
/// the canonicaliser potentially swapping which axis lands in theta1).
fn axis_pair_err(fit_t1: f32, fit_t2: f32, gt_t1: f32, gt_t2: f32) -> f32 {
    let opt_a = angle_err_mod_pi(fit_t1, gt_t1).max(angle_err_mod_pi(fit_t2, gt_t2));
    let opt_b = angle_err_mod_pi(fit_t1, gt_t2).max(angle_err_mod_pi(fit_t2, gt_t1));
    opt_a.min(opt_b)
}

/// Per-axis tolerance for parity assertions (radians / gray levels).
const PARITY_TOL: f32 = 5e-3;

// ---------------------------------------------------------------------------
// Parity / accuracy tests
// ---------------------------------------------------------------------------

#[test]
fn ring_fit_axis_aligned_corner() {
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
    assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);

    let err1 = ((fit.theta1 - PI * 0.25).abs()).min((fit.theta1 - (PI * 0.25 + PI)).abs());
    let err2 = ((fit.theta2 - PI * 0.75).abs()).min((fit.theta2 - (PI * 0.75 + PI)).abs());
    assert!(err1 < 1e-2, "theta1 err {err1}");
    assert!(err2 < 1e-2, "theta2 err {err2}");
    assert!(fit.amp > 60.0, "amp {}", fit.amp);
    assert!(fit.rms < 1e-2, "rms {}", fit.rms);
    assert!(fit.sigma_theta1 < 5e-2);
    assert!(fit.sigma_theta2 < 5e-2);
}

#[test]
fn ring_fit_non_orthogonal_corner() {
    let t1 = 30f32.to_radians();
    let t2 = 100f32.to_radians();
    let (samples, phi) = synthetic_ring(120.0, 60.0, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let fold = |x: f32, target: f32| -> f32 {
        let d = (x - target).abs();
        d.min(PI - d).min((x - target - PI).abs())
    };
    let err1 = fold(fit.theta1, t1);
    let err2 = fold(fit.theta2, t2);
    assert!(err1 < 0.05, "theta1 {} vs {t1}, err {err1}", fit.theta1);
    assert!(err2 < 0.05, "theta2 {} vs {t2}, err {err2}", fit.theta2);
}

#[test]
fn ring_fit_robust_seed_recovers_extreme_skew_trace() {
    // Deterministic synthetic fixture from orientation_bench:
    // bench_default axis_skew=30°, seed=1, cell sample 8, sampled at the
    // detected image-frame center (21.815319, 20.875401). The legacy
    // 2nd-harmonic-only seed converged to the wrong basin by ~62° on
    // this trace even though the 16 bilinear samples contain the correct
    // two-axis model.
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [
        0.485009_f32,
        0.230703,
        0.0,
        0.203177,
        11.4078,
        139.206,
        230.952,
        128.866,
        14.6169,
        0.32334,
        1.58913,
        0.713731,
        10.256,
        129.307,
        107.935,
        8.76894,
    ];
    let gt_t1 = 20.765984_f32.to_radians();
    let gt_t2 = 57.899868_f32.to_radians();

    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let err = axis_pair_err(fit.theta1, fit.theta2, gt_t1, gt_t2);
    assert!(
        err < 8.0_f32.to_radians(),
        "axis err {} deg for fit {:?}",
        err.to_degrees(),
        fit
    );
    assert!(fit.rms < 35.0, "rms {}", fit.rms);
}

#[test]
fn ring_fit_polarity_swap_on_sign_flip() {
    let t1 = 0.3f32;
    let t2 = 0.3 + FRAC_PI_2;
    let (s_pos, phi) = synthetic_ring(128.0, 80.0, t1, t2);
    let (s_neg, _) = synthetic_ring(128.0, -80.0, t1, t2);

    let fit_pos = fit_axes_from_samples(&s_pos, &phi, OrientationMethod::RingFit);
    let fit_neg = fit_axes_from_samples(&s_neg, &phi, OrientationMethod::RingFit);

    let mod_pi = |x: f32| x.rem_euclid(PI);
    let pos_lines = [mod_pi(fit_pos.theta1), mod_pi(fit_pos.theta2)];
    let neg_lines = [mod_pi(fit_neg.theta1), mod_pi(fit_neg.theta2)];
    let pair_err = |a: &[f32; 2], b: &[f32; 2]| -> f32 {
        let d = |x: f32, y: f32| {
            let e = (x - y).abs();
            e.min(PI - e)
        };
        let opt1 = d(a[0], b[0]).max(d(a[1], b[1]));
        let opt2 = d(a[0], b[1]).max(d(a[1], b[0]));
        opt1.min(opt2)
    };
    assert!(
        pair_err(&pos_lines, &neg_lines) < 0.02,
        "lines mismatch: pos={pos_lines:?}, neg={neg_lines:?}"
    );
    assert!(fit_pos.amp > 0.0 && fit_neg.amp > 0.0);
}

#[test]
fn ring_fit_flat_ring_returns_degenerate() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0f32; 16];
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);
    assert_eq!(fit.amp, 0.0);
    assert!(fit.sigma_theta1 >= PI - 1e-3);
    assert!(fit.sigma_theta2 >= PI - 1e-3);
    assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
    assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);
}

#[test]
fn ring_fit_canonicalization_invariants() {
    let cases: &[(f32, f32, f32)] = &[
        (10.0, FRAC_PI_2, 0.1),
        (10.0, FRAC_PI_2, 0.1 + 3.0 * PI),
        (-10.0, FRAC_PI_2, 0.1),
        (10.0, FRAC_PI_2 + PI, 0.1),
        (5.0, -FRAC_PI_4, 0.1),
    ];
    for &(amp, skew, offset) in cases {
        let (samples, phi) = synthetic_ring(128.0, amp, offset, offset + skew);
        let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);
        assert!(
            (0.0..PI + 1e-6).contains(&fit.theta1),
            "theta1 {} out of [0, π)",
            fit.theta1
        );
        assert!(
            fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI + 1e-6,
            "theta2 {} not in (theta1={}, theta1+π)",
            fit.theta2,
            fit.theta1
        );
        assert!(fit.amp >= 0.0, "amp {} negative", fit.amp);
    }
}

#[test]
fn ring_fit_image_input_matches_sample_input() {
    let mu = 128.0f32;
    let amp = 80.0f32;
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(mu, amp, t1, t2);

    let w = 41usize;
    let h = 41usize;
    let cx = 20i32;
    let cy = 20i32;
    let mut img = vec![0u8; w * h];
    let ring = ring_offsets(5);
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        let px = (cx + dx) as usize;
        let py = (cy + dy) as usize;
        let q = samples[i].round().clamp(0.0, 255.0) as u8;
        img[py * w + px] = q;
    }

    let from_samples = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);
    let from_image = fit_axes_at_point(
        &img,
        w,
        h,
        cx as f32,
        cy as f32,
        5,
        OrientationMethod::RingFit,
    );

    assert!((from_samples.theta1 - from_image.theta1).abs() < 1e-2);
    assert!((from_samples.theta2 - from_image.theta2).abs() < 1e-2);
    assert!(from_image.amp > 0.0);
}

#[test]
fn ring_fit_radius10_uses_radius5_safety_when_outer_ring_is_suspicious() {
    let w = 41usize;
    let h = 41usize;
    let cx = 20i32;
    let cy = 20i32;
    let mut img = vec![128u8; w * h];

    // Deliberately make the radius-10 trace a high-contrast, nearly
    // parallel-axis pattern. That outer trace is a valid local model but
    // suspicious as a chess-grid orientation. The canonical radius-5
    // trace carries the intended local axes and should be used instead.
    let outer_ring = ring_offsets(10);
    let outer_phi = ring_angles(outer_ring);
    for (i, &(dx, dy)) in outer_ring.iter().enumerate() {
        let q = eval_model(
            outer_phi[i],
            128.0,
            80.0,
            0.0_f32.to_radians(),
            25.0_f32.to_radians(),
        )
        .round()
        .clamp(0.0, 255.0) as u8;
        img[(cy + dy) as usize * w + (cx + dx) as usize] = q;
    }

    let inner_t1 = 25.0_f32.to_radians();
    let inner_t2 = 115.0_f32.to_radians();
    let inner_ring = ring_offsets(5);
    let inner_phi = ring_angles(inner_ring);
    for (i, &(dx, dy)) in inner_ring.iter().enumerate() {
        let q = eval_model(inner_phi[i], 128.0, 80.0, inner_t1, inner_t2)
            .round()
            .clamp(0.0, 255.0) as u8;
        img[(cy + dy) as usize * w + (cx + dx) as usize] = q;
    }

    let fit = fit_axes_at_point(
        &img,
        w,
        h,
        cx as f32,
        cy as f32,
        10,
        OrientationMethod::RingFit,
    );
    let err = axis_pair_err(fit.theta1, fit.theta2, inner_t1, inner_t2);
    assert!(
        err < 2.0_f32.to_radians(),
        "radius-10 safety err {} deg for fit {:?}",
        err.to_degrees(),
        fit
    );
}

// ---------------------------------------------------------------------------
// Perf-parity assertions (tight 5 mrad tolerance)
// ---------------------------------------------------------------------------

#[test]
fn parity_clean_orthogonal_corner() {
    let (mu, amp) = (128.0_f32, 80.0_f32);
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (samples, phi) = synthetic_ring(mu, amp, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let err = axis_pair_err(fit.theta1, fit.theta2, t1, t2);
    assert!(err < PARITY_TOL, "axis err {err}");
    assert!(
        (fit.amp - amp).abs() < PARITY_TOL,
        "amp {} expected {}",
        fit.amp,
        amp
    );
    assert!(fit.rms < PARITY_TOL, "rms {}", fit.rms);
    assert!(fit.sigma_theta1 < PARITY_TOL);
    assert!(fit.sigma_theta2 < PARITY_TOL);
}

#[test]
fn parity_projective_skew_30_100() {
    let t1 = 30f32.to_radians();
    let t2 = 100f32.to_radians();
    let (mu, amp) = (120.0_f32, 60.0_f32);
    let (samples, phi) = synthetic_ring(mu, amp, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let err = axis_pair_err(fit.theta1, fit.theta2, t1, t2);
    assert!(err < PARITY_TOL, "axis err {err}");
    assert!(
        (fit.amp - amp).abs() < PARITY_TOL,
        "amp {} expected {}",
        fit.amp,
        amp
    );
    assert!(fit.rms < PARITY_TOL, "rms {}", fit.rms);
    assert!(fit.sigma_theta1 < PARITY_TOL);
    assert!(fit.sigma_theta2 < PARITY_TOL);
}

#[test]
fn parity_noisy_sharp_corner() {
    let (mu, amp) = (128.0_f32, 80.0_f32);
    let t1 = PI * 0.25;
    let t2 = PI * 0.75;
    let (mut samples, phi) = synthetic_ring(mu, amp, t1, t2);
    let noise = [
        1.5_f32, -1.7, 2.1, -0.9, 0.4, -1.1, 1.8, -2.0, 0.7, 1.2, -0.8, -1.6, 0.3, 1.9, -0.5, -1.3,
    ];
    for i in 0..16 {
        samples[i] += noise[i];
    }

    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let err = axis_pair_err(fit.theta1, fit.theta2, t1, t2);
    assert!(err < PARITY_TOL, "axis err {err}");
    assert!(
        (fit.amp - amp).abs() < 2.0,
        "amp {} expected {} ± 2",
        fit.amp,
        amp
    );
    assert!(fit.sigma_theta1.is_finite() && fit.sigma_theta1 < 0.05);
    assert!(fit.sigma_theta2.is_finite() && fit.sigma_theta2 < 0.05);
    assert!(fit.rms.is_finite() && fit.rms > 0.0);
}

#[test]
fn parity_low_contrast_corner() {
    let (mu, amp) = (96.0_f32, 5.0_f32);
    let t1 = 0.4_f32;
    let t2 = 0.4 + FRAC_PI_2;
    let (samples, phi) = synthetic_ring(mu, amp, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    let err = axis_pair_err(fit.theta1, fit.theta2, t1, t2);
    assert!(err < PARITY_TOL, "axis err {err}");
    assert!(
        (fit.amp - amp).abs() < PARITY_TOL,
        "amp {} expected {}",
        fit.amp,
        amp
    );
    assert!(fit.rms < PARITY_TOL, "rms {}", fit.rms);
    assert!(fit.sigma_theta1 < PARITY_TOL);
    assert!(fit.sigma_theta2 < PARITY_TOL);
}

#[test]
fn parity_degenerate_flat_input() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0_f32; 16];
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    assert!(fit.amp.abs() < PARITY_TOL, "amp {} should be 0", fit.amp);
    assert!(
        (fit.theta1 - 0.0).abs() < PARITY_TOL,
        "theta1 {} should be 0",
        fit.theta1
    );
    assert!(
        (fit.theta2 - FRAC_PI_2).abs() < PARITY_TOL,
        "theta2 {} should be π/2",
        fit.theta2
    );
    assert!(
        (fit.sigma_theta1 - PI).abs() < PARITY_TOL,
        "sigma1 {} should be π",
        fit.sigma_theta1
    );
    assert!(
        (fit.sigma_theta2 - PI).abs() < PARITY_TOL,
        "sigma2 {} should be π",
        fit.sigma_theta2
    );
    assert!(fit.rms < PARITY_TOL, "rms {} should be ~0", fit.rms);
}

// ---------------------------------------------------------------------------
// σ-LUT specific tests
// ---------------------------------------------------------------------------

#[test]
fn lut_applies_to_sigmas_only() {
    // Clean fit → fit_rms ≈ 0 → LUT multiplier 1.25 (first breakpoint).
    // Angles, amp, rms are not exposed before the LUT, so we verify the
    // clean-fit regime and that finite sigmas were scaled.
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    // On a clean noiseless input the GN solve converges tightly.
    assert!(
        fit.rms < 0.05,
        "baseline rms {} not in clean regime",
        fit.rms
    );
    // The LUT must have inflated the sigmas above the raw GN value
    // (which would be extremely small on a perfect input).
    assert!(fit.sigma_theta1 > 0.0);
    assert!(fit.sigma_theta2 > 0.0);
    // But they must still be well below π.
    assert!(fit.sigma_theta1 < 0.1);
    assert!(fit.sigma_theta2 < 0.1);
}

#[test]
fn degenerate_fit_remains_degenerate() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0f32; 16];
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::RingFit);

    // Flat ring → degenerate sentinel → σ pinned at π even after LUT.
    assert!((fit.sigma_theta1 - PI).abs() < 1e-6);
    assert!((fit.sigma_theta2 - PI).abs() < 1e-6);
}

#[test]
fn ring_fit_is_the_default() {
    assert_eq!(OrientationMethod::default(), OrientationMethod::RingFit);
}

#[test]
fn ring_fit_serde_key_is_ring_fit() {
    let s = serde_json::to_string(&OrientationMethod::RingFit).unwrap();
    assert_eq!(s, r#""ring_fit""#);
    let v: OrientationMethod = serde_json::from_str(r#""ring_fit""#).unwrap();
    assert_eq!(v, OrientationMethod::RingFit);
}

#[test]
fn disk_fit_serde_key_is_disk_fit() {
    let s = serde_json::to_string(&OrientationMethod::DiskFit).unwrap();
    assert_eq!(s, r#""disk_fit""#);
    let v: OrientationMethod = serde_json::from_str(r#""disk_fit""#).unwrap();
    assert_eq!(v, OrientationMethod::DiskFit);
}

// ---------------------------------------------------------------------------
// Optional orientation: `describe_corners(.., None)` skips the fit
// ---------------------------------------------------------------------------

#[test]
fn describe_corners_none_skips_orientation_keeps_positions() {
    use chess_corners_core::{describe_corners, Corner};

    // Synthetic checkerboard so the ring sampler has real support.
    let w = 48usize;
    let h = 48usize;
    let period = 12i32;
    let mut img = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let rx = ((x as i32) / period) & 1;
            let ry = ((y as i32) / period) & 1;
            img[y * w + x] = if rx ^ ry == 0 { 40 } else { 215 };
        }
    }

    let corners = vec![
        Corner::new(12.3, 12.7, 1.0),
        Corner::new(24.1, 24.2, 2.0),
        Corner::new(24.0, 12.0, 0.5),
    ];

    let oriented = describe_corners(
        &img,
        w,
        h,
        5,
        corners.clone(),
        Some(OrientationMethod::RingFit),
    );
    let skipped = describe_corners(&img, w, h, 5, corners.clone(), None);

    assert_eq!(oriented.len(), corners.len());
    assert_eq!(skipped.len(), corners.len());
    for (i, (o, s)) in oriented.iter().zip(skipped.iter()).enumerate() {
        // Position and response are bit-identical; only the fit is skipped.
        assert_eq!(o.x.to_bits(), s.x.to_bits(), "x mismatch at corner {i}");
        assert_eq!(o.y.to_bits(), s.y.to_bits(), "y mismatch at corner {i}");
        assert_eq!(
            o.response.to_bits(),
            s.response.to_bits(),
            "response mismatch at corner {i}"
        );
        assert!(o.axes.is_some(), "oriented corner {i} should carry axes");
        assert!(
            s.axes.is_none(),
            "skipped corner {i} must have axes == None"
        );
    }
}
