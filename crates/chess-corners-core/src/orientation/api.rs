//! Public API for sampling and fitting axes at a single point.
//!
//! [`fit_axes_at_point`] is the public entry point: it samples the
//! 16-point ChESS ring around an image point and dispatches to the
//! chosen [`OrientationMethod`]. This is the workhorse used by the
//! descriptor pipeline and by the orientation benchmark.
//!
//! A crate-private `fit_axes_from_samples` helper runs the same
//! dispatch directly on pre-sampled ring values, for use by this
//! module's own unit tests.

use super::descriptor::{ring_angles, sample_ring};
use super::{disk_sector, ring_fit, ring_fit_for_image, OrientationMethod};
use crate::detect::chess::ring::ring_offsets;
use crate::imageview::ImageView;

/// Result of a two-axis orientation fit at a single corner.
///
/// All [`OrientationMethod`] variants populate the same fields.
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
/// Useful for orientation estimates at externally-chosen points without
/// running detection.
pub fn fit_axes_at_point(
    view: ImageView<'_>,
    cx: f32,
    cy: f32,
    radius: u32,
    method: OrientationMethod,
) -> AxisFitResult {
    let ring = ring_offsets(radius);
    let ring_phi = ring_angles(ring);
    let samples = sample_ring(view.data, view.width, view.height, cx, cy, ring);
    match method {
        OrientationMethod::RingFit => {
            ring_fit_for_image(view, cx, cy, radius, &samples, &ring_phi).into()
        }
        OrientationMethod::DiskFit => {
            disk_sector::fit(view, cx, cy, radius, &samples, &ring_phi).into()
        }
    }
}

/// Run the chosen orientation method on pre-sampled ring values.
///
/// Use this when you already have the 16 ring samples in hand and only
/// need the fit. Image-dependent variants will fall back to the
/// ring-only [`OrientationMethod::RingFit`] result when invoked through
/// this helper.
#[cfg(test)]
pub(crate) fn fit_axes_from_samples(
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

// White-box ring-fit parity / accuracy tests. These build synthetic ring
// samples from the crate-private ring-offset geometry, so they live in
// the crate rather than in `tests/`. Cases that exercise only the public
// API stay in `tests/orientation_ring_fit.rs`.
#[cfg(test)]
mod tests {
    use super::{fit_axes_at_point, fit_axes_from_samples};
    use crate::detect::chess::ring::ring_offsets;
    use crate::imageview::ImageView;
    use crate::OrientationMethod;
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

    // -----------------------------------------------------------------------
    // Parity / accuracy tests
    // -----------------------------------------------------------------------

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
        let view = ImageView::from_u8_slice(w, h, &img).expect("view dims match buffer");
        let from_image =
            fit_axes_at_point(view, cx as f32, cy as f32, 5, OrientationMethod::RingFit);

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

        let view = ImageView::from_u8_slice(w, h, &img).expect("view dims match buffer");
        let fit = fit_axes_at_point(view, cx as f32, cy as f32, 10, OrientationMethod::RingFit);
        let err = axis_pair_err(fit.theta1, fit.theta2, inner_t1, inner_t2);
        assert!(
            err < 2.0_f32.to_radians(),
            "radius-10 safety err {} deg for fit {:?}",
            err.to_degrees(),
            fit
        );
    }

    // -----------------------------------------------------------------------
    // Perf-parity assertions (tight 5 mrad tolerance)
    // -----------------------------------------------------------------------

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
            1.5_f32, -1.7, 2.1, -0.9, 0.4, -1.1, 1.8, -2.0, 0.7, 1.2, -0.8, -1.6, 0.3, 1.9, -0.5,
            -1.3,
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

    // -----------------------------------------------------------------------
    // σ-LUT specific tests
    // -----------------------------------------------------------------------

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
}
