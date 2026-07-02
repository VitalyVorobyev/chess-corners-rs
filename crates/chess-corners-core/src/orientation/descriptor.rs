//! Lift detector candidates into [`CornerDescriptor`]s by sampling the
//! source image and dispatching to an [`OrientationMethod`].
//!
//! The detector path produces intermediate [`Corner`] values;
//! [`describe_corners`] then samples the original image on a ChESS
//! ring around each corner and fits a parametric two-axis intensity
//! model to estimate the local grid geometry. The fit body itself
//! lives in the sibling `ring_fit` and `disk_sector` submodules;
//! this file is a sampling and assembly shim.

use super::{disk_fit_for_descriptor, ring_fit_for_image, OrientationMethod, TwoAxisFit};
use crate::detect::chess::ring::ring_offsets;
use crate::detect::{AxisEstimate, Corner, CornerDescriptor};
use crate::imageview::ImageView;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Smoothing slope of the tanh sign approximation used in the
/// intensity model. Re-exposed at module scope so the in-file tests
/// keep compiling unchanged after the fit body moved.
#[cfg(test)]
const TANH_BETA: f32 = 4.0;

/// Full-frame ChESS runs can produce thousands of positive-response
/// candidates under permissive thresholds. The full-disk estimator is
/// intentionally local and expensive, so only the strongest candidates
/// get the disk pass; the rest keep the ring-fit fallback.
const FULL_DISK_MAX_FULL_IMAGE_CORNERS: usize = 80;

/// Convert raw corner candidates into full descriptors by sampling the
/// source image and running the chosen [`OrientationMethod`] at each
/// corner.
///
/// Orientation and polarity follow the conventions documented on
/// [`CornerDescriptor`]. When `method` is
/// `Some(`[`OrientationMethod::DiskFit`]`)` and the candidate count exceeds
/// an internal cap, only the strongest candidates run the full-disk
/// estimator; the rest fall back to ring-fit transparently.
///
/// When `method` is `None` the orientation fit is skipped entirely — no
/// ring is sampled and no model is fit — and every descriptor carries
/// `axes: None`. This is the cheaper fast path: it does strictly less work
/// (no per-corner ring buffer, no full-disk candidate mask) than any fitted
/// method.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "info",
        skip(img, corners),
        fields(corners = corners.len(), method = ?method)
    )
)]
pub fn describe_corners(
    img: &[u8],
    w: usize,
    h: usize,
    radius: u32,
    corners: Vec<Corner>,
    method: Option<OrientationMethod>,
) -> Vec<CornerDescriptor> {
    // Fast path: orientation skipped. No ring sampling, no fit, no mask
    // allocation — only the output vector is allocated. Parallelism and
    // output order match the fitted path (index-stable `par` collect).
    let Some(method) = method else {
        #[cfg(feature = "rayon")]
        {
            return corners
                .into_par_iter()
                .map(|c| CornerDescriptor::without_axes(c.x, c.y, c.strength))
                .collect();
        }
        #[cfg(not(feature = "rayon"))]
        {
            return corners
                .into_iter()
                .map(|c| CornerDescriptor::without_axes(c.x, c.y, c.strength))
                .collect();
        }
    };

    let ring = ring_offsets(radius);
    let ring_phi = ring_angles(ring);
    let full_disk_mask = full_disk_top_response_mask(&corners, method);
    // Bundle the source buffer into a zero-origin view so the per-corner
    // fit takes one argument instead of an `(img, w, h)` triple. Field
    // access (not `ImageView::sample_*`) keeps the downstream sampling
    // bit-identical.
    let view = ImageView {
        data: img,
        width: w,
        height: h,
        origin: [0, 0],
    };

    let describe = |(idx, c): (usize, Corner)| -> CornerDescriptor {
        let samples = sample_ring(view.data, view.width, view.height, c.x, c.y, ring);
        let orientation_method = match &full_disk_mask {
            Some(mask) if !mask[idx] => OrientationMethod::RingFit,
            _ => method,
        };
        let fit = fit_two_axes_with_method(
            &samples,
            &ring_phi,
            view,
            c.x,
            c.y,
            radius,
            orientation_method,
        );
        CornerDescriptor {
            x: c.x,
            y: c.y,
            response: c.strength,
            axes: Some([
                AxisEstimate {
                    angle: fit.theta1,
                    sigma: fit.sigma_theta1,
                },
                AxisEstimate {
                    angle: fit.theta2,
                    sigma: fit.sigma_theta2,
                },
            ]),
        }
    };

    #[cfg(feature = "rayon")]
    {
        corners.into_par_iter().enumerate().map(describe).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        corners.into_iter().enumerate().map(describe).collect()
    }
}

fn full_disk_top_response_mask(corners: &[Corner], method: OrientationMethod) -> Option<Vec<bool>> {
    if !matches!(method, OrientationMethod::DiskFit)
        || corners.len() <= FULL_DISK_MAX_FULL_IMAGE_CORNERS
    {
        return None;
    }

    let mut indices: Vec<usize> = (0..corners.len()).collect();
    indices.sort_by(|&a, &b| {
        corners[b]
            .strength
            .total_cmp(&corners[a].strength)
            .then_with(|| a.cmp(&b))
    });

    let mut mask = vec![false; corners.len()];
    for &idx in indices.iter().take(FULL_DISK_MAX_FULL_IMAGE_CORNERS) {
        mask[idx] = true;
    }
    Some(mask)
}

/// Sample the 16-point ChESS ring around `(x, y)` using bilinear interpolation.
pub(crate) fn sample_ring(
    img: &[u8],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    ring: &[(i32, i32); 16],
) -> [f32; 16] {
    let mut samples = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        let sx = x + dx as f32;
        let sy = y + dy as f32;
        samples[i] = sample_bilinear(img, w, h, sx, sy);
    }
    samples
}

/// Angles of each ring sample in image coordinates, via `atan2(dy, dx)`.
#[inline]
pub(crate) fn ring_angles(ring: &[(i32, i32); 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        out[i] = (dy as f32).atan2(dx as f32);
    }
    out
}

#[inline]
fn fit_two_axes_with_method(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    view: ImageView<'_>,
    cx: f32,
    cy: f32,
    radius: u32,
    method: OrientationMethod,
) -> TwoAxisFit {
    match method {
        OrientationMethod::RingFit => ring_fit_for_image(view, cx, cy, radius, samples, ring_phi),
        OrientationMethod::DiskFit => {
            disk_fit_for_descriptor(view, cx, cy, radius, samples, ring_phi)
        }
    }
}

fn sample_bilinear(img: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }

    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    let xf = x.clamp(0.0, max_x);
    let yf = y.clamp(0.0, max_y);

    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let wx = xf - x0 as f32;
    let wy = yf - y0 as f32;

    let i00 = img[y0 * w + x0] as f32;
    let i10 = img[y0 * w + x1] as f32;
    let i01 = img[y1 * w + x0] as f32;
    let i11 = img[y1 * w + x1] as f32;

    let i0 = i00 * (1.0 - wx) + i10 * wx;
    let i1 = i01 * (1.0 - wx) + i11 * wx;
    i0 * (1.0 - wy) + i1 * wy
}

#[cfg(test)]
mod tests {
    use super::super::{ring_fit_canonicalize_for_test, ring_fit_for_descriptor};
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[inline]
    fn fit_two_axes(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
        ring_fit_for_descriptor(samples, ring_phi)
    }

    #[inline]
    fn canonicalize(theta1: f32, theta2: f32, amp: f32) -> (f32, f32, f32) {
        ring_fit_canonicalize_for_test(theta1, theta2, amp)
    }

    /// Evaluate the intensity model at angle φ for given parameters.
    fn eval_model(phi: f32, mu: f32, amp: f32, theta1: f32, theta2: f32) -> f32 {
        let h1 = (TANH_BETA * (phi - theta1).sin()).tanh();
        let h2 = (TANH_BETA * (phi - theta2).sin()).tanh();
        mu + amp * h1 * h2
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
    fn fit_recovers_axis_aligned_corner() {
        let (samples, phi) = synthetic_ring(128.0, 80.0, PI * 0.25, PI * 0.75);
        let fit = fit_two_axes(&samples, &phi);

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
    fn fit_recovers_non_orthogonal_corner() {
        let t1 = 30f32.to_radians();
        let t2 = 100f32.to_radians();
        let (samples, phi) = synthetic_ring(120.0, 60.0, t1, t2);
        let fit = fit_two_axes(&samples, &phi);

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
    fn fit_polarity_swap_on_sign_flip() {
        let t1 = 0.3f32;
        let t2 = 0.3 + FRAC_PI_2;
        let (s_pos, phi) = synthetic_ring(128.0, 80.0, t1, t2);
        let (s_neg, _) = synthetic_ring(128.0, -80.0, t1, t2);

        let fit_pos = fit_two_axes(&s_pos, &phi);
        let fit_neg = fit_two_axes(&s_neg, &phi);

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
    fn fit_flat_ring_returns_degenerate() {
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = [77.0f32; 16];
        let fit = fit_two_axes(&samples, &phi);
        assert_eq!(fit.amp, 0.0);
        assert!(fit.sigma_theta1 >= PI - 1e-3);
        assert!(fit.sigma_theta2 >= PI - 1e-3);
        assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
        assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);
    }

    #[test]
    fn canonicalization_enforces_invariants() {
        let cases = [
            (0.1f32, 0.1 + FRAC_PI_2, 10.0),
            (0.1 + 3.0 * PI, 0.1 + 3.0 * PI + FRAC_PI_2, 10.0),
            (0.1, 0.1 + FRAC_PI_2, -10.0),
            (0.1, 0.1 + FRAC_PI_2 + PI, 10.0),
            (0.1, 0.1 - FRAC_PI_4, 5.0),
        ];
        for (t1, t2, a) in cases {
            let (c1, c2, ca) = canonicalize(t1, t2, a);
            assert!((0.0..PI + 1e-6).contains(&c1), "theta1 {c1}");
            assert!(c2 > c1 && c2 < c1 + PI + 1e-6, "theta2 {c2} vs {c1}");
            assert!(ca >= 0.0, "amp {ca}");
        }
    }

    /// Simple LCG-based deterministic pseudo-noise generator.
    fn lcg_noise(seed: u64, n: usize, sigma: f32) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut acc = 0.0f32;
            for _ in 0..6 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 33) as f32) / ((1u64 << 31) as f32) - 1.0;
                acc += u;
            }
            out.push(sigma * acc / 6f32.sqrt());
        }
        out
    }

    #[test]
    fn fit_sigma_grows_with_noise() {
        let t1 = FRAC_PI_4;
        let t2 = 3.0 * FRAC_PI_4;

        let (clean_samples, phi) = synthetic_ring(128.0, 80.0, t1, t2);
        let clean_fit = fit_two_axes(&clean_samples, &phi);

        let mut noisy_samples = clean_samples;
        for (i, n) in lcg_noise(42, 16, 5.0).iter().enumerate() {
            noisy_samples[i] += *n;
        }
        let noisy_fit = fit_two_axes(&noisy_samples, &phi);

        assert!(clean_fit.sigma_theta1 < 5e-2);
        assert!(clean_fit.sigma_theta2 < 5e-2);
        assert!(noisy_fit.sigma_theta1 > clean_fit.sigma_theta1);
        assert!(noisy_fit.sigma_theta2 > clean_fit.sigma_theta2);
        assert!(noisy_fit.sigma_theta1.is_finite());
        assert!(noisy_fit.sigma_theta2.is_finite());
        assert!(noisy_fit.rms > clean_fit.rms);
    }
}
