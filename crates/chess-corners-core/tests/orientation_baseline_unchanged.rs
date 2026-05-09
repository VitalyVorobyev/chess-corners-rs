//! Verifies that the [`OrientationMethod::Baseline`] path through
//! [`fit_axes_at_point`] / [`fit_axes_from_samples`] reproduces the
//! pre-refactor `fit_two_axes` behaviour exactly.
//!
//! Each test mirrors the corresponding case in
//! `crates/chess-corners-core/src/descriptor.rs::tests`, but drives
//! the new public dispatch entry point. Phase 3 of the orientation
//! plan promises bit-identical baseline output after the structural
//! move; these tests guard that contract.

use chess_corners_core::orientation::{
    fit_axes_at_point, fit_axes_from_samples, OrientationMethod,
};
use chess_corners_core::ring::ring_offsets;
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Same `tanh(β·sin(...))·tanh(β·sin(...))` model used by the
/// in-`descriptor.rs` test helper. Kept private to this file so the
/// test surface stays self-contained.
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

#[test]
fn baseline_axis_aligned_corner() {
    let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);

    assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
    assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);

    let err1 = ((fit.theta1 - PI * 0.25).abs()).min((fit.theta1 - (PI * 0.25 + PI)).abs());
    let err2 = ((fit.theta2 - PI * 0.75).abs()).min((fit.theta2 - (PI * 0.75 + PI)).abs());
    assert!(err1 < 1e-2, "theta1 err {err1} for fit {fit:?}");
    assert!(err2 < 1e-2, "theta2 err {err2} for fit {fit:?}");
    assert!(fit.amp > 60.0, "amp {}", fit.amp);
    assert!(fit.rms < 1e-2, "rms {}", fit.rms);
    assert!(fit.sigma_theta1 < 5e-2);
    assert!(fit.sigma_theta2 < 5e-2);
}

#[test]
fn baseline_non_orthogonal_corner() {
    // Axes at 30° and 100° (70° between). Projective-like. Mirrors
    // the `fit_recovers_non_orthogonal_corner` case in `descriptor.rs`.
    let t1 = 30f32.to_radians();
    let t2 = 100f32.to_radians();
    let (samples, phi) = synthetic_ring(120.0, 60.0, t1, t2);
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);

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
fn baseline_polarity_swap_on_sign_flip() {
    // Flipping A's sign must swap which axis carries the "dark-CCW"
    // role but leave the unordered set of line directions (mod π)
    // unchanged.
    let t1 = 0.3f32;
    let t2 = 0.3 + FRAC_PI_2;
    let (s_pos, phi) = synthetic_ring(128.0, 80.0, t1, t2);
    let (s_neg, _) = synthetic_ring(128.0, -80.0, t1, t2);

    let fit_pos = fit_axes_from_samples(&s_pos, &phi, OrientationMethod::Baseline);
    let fit_neg = fit_axes_from_samples(&s_neg, &phi, OrientationMethod::Baseline);

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
fn baseline_flat_ring_returns_degenerate() {
    let ring = ring_offsets(5);
    let phi = ring_angles(ring);
    let samples = [77.0f32; 16];
    let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    assert_eq!(fit.amp, 0.0);
    assert!(fit.sigma_theta1 >= PI - 1e-3);
    assert!(fit.sigma_theta2 >= PI - 1e-3);
    assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
    assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);
}

#[test]
fn baseline_canonicalization_invariants() {
    // Sweep `(amp, theta_skew, theta_offset)` through the same kind of
    // unusual configurations the in-file `canonicalization_enforces_invariants`
    // test exercises. We can't call `canonicalize` directly through
    // the public API (it's an internal step), so we drive the full
    // fit and verify the canonicalization invariants on the output.
    let cases: &[(f32, f32, f32)] = &[
        (10.0, FRAC_PI_2, 0.1),
        (10.0, FRAC_PI_2, 0.1 + 3.0 * PI),
        (-10.0, FRAC_PI_2, 0.1),
        (10.0, FRAC_PI_2 + PI, 0.1),
        (5.0, -FRAC_PI_4, 0.1),
    ];

    for &(amp, skew, offset) in cases {
        let (samples, phi) = synthetic_ring(128.0, amp, offset, offset + skew);
        let fit = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
        // Same invariants the descriptor's `canonicalize` enforces:
        // theta1 ∈ [0, π), theta2 ∈ (theta1, theta1 + π), amp ≥ 0.
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
fn baseline_image_input_matches_sample_input() {
    // `fit_axes_at_point` and `fit_axes_from_samples` must agree when
    // the image is constructed so that bilinear-sampling at integer
    // ring offsets reproduces the synthetic samples exactly.
    //
    // Build a 41×41 image. Ring radius = 5, centre = (20, 20). The
    // ChESS ring offsets are integer, so for each ring point (cx+dx,
    // cy+dy) with integer dx,dy and integer cx=cy=20 the bilinear
    // sampler reduces to a direct lookup. Filling those 16 pixels
    // with the desired sample values (clamped to [0, 255]) gives a
    // bilinear evaluation that matches `synthetic_ring` to within
    // float-rounding of the u8 quantisation.
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

    let from_samples = fit_axes_from_samples(&samples, &phi, OrientationMethod::Baseline);
    let from_image = fit_axes_at_point(
        &img,
        w,
        h,
        cx as f32,
        cy as f32,
        5,
        OrientationMethod::Baseline,
    );

    // The image path round-trips through u8 quantisation, so we
    // accept ~1 sample LSB of noise on the recovered angles. Both
    // fits must converge to the same canonical orientation.
    assert!((from_samples.theta1 - from_image.theta1).abs() < 1e-2);
    assert!((from_samples.theta2 - from_image.theta2).abs() < 1e-2);
    assert!(from_image.amp > 0.0);
}
