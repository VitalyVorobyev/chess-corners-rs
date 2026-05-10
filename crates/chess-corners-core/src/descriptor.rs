//! Corner descriptors and helpers for chessboard detection.
//!
//! This module turns raw ChESS corner candidates into richer
//! [`CornerDescriptor`] values that carry subpixel position, response,
//! contrast, and two-axis orientation with per-axis precision.
//!
//! The detector in [`crate::detect`] produces intermediate [`Corner`]
//! values; [`corners_to_descriptors_with_method`] then samples the
//! original image on a ChESS ring around each corner and fits a
//! parametric two-axis intensity model to estimate the local grid
//! geometry.
//!
//! The orientation fit body lives in [`crate::orientation`]; this
//! module is a thin sampling and assembly shim. Future algorithm
//! variants are added through [`crate::orientation::OrientationMethod`]
//! without touching this file.
use crate::orientation::{self, OrientationMethod, TwoAxisFit};
use crate::ring::ring_offsets;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Smoothing slope of the tanh sign approximation used in the
/// intensity model. Re-exposed at module scope (with the same value as
/// the canonical constant in [`crate::orientation`]) so the
/// existing tests in this file keep compiling unchanged after the
/// fit body moved to its own module.
#[cfg(test)]
const TANH_BETA: f32 = 4.0;

/// Full-frame ChESS runs can produce thousands of positive-response
/// candidates under permissive thresholds. The full-disk estimator is
/// intentionally local and expensive, so only the strongest candidates
/// get the disk pass; the rest keep the ring-fit fallback.
const FULL_DISK_MAX_FULL_IMAGE_CORNERS: usize = 80;

/// A detected ChESS corner (subpixel).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Corner {
    /// Subpixel x coordinate in image pixels.
    pub x: f32,
    /// Subpixel y coordinate in image pixels.
    pub y: f32,
    /// Raw ChESS response at the integer peak (before refinement).
    pub strength: f32,
}

impl Corner {
    /// Construct a [`Corner`].
    #[inline]
    pub fn new(x: f32, y: f32, strength: f32) -> Self {
        Self { x, y, strength }
    }
}

/// Direction of one local grid axis with its 1σ angular uncertainty.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct AxisEstimate {
    /// Axis direction, radians in [0, 2π).
    ///
    /// See [`CornerDescriptor`] for the joint polarity convention.
    pub angle: f32,
    /// 1σ angular uncertainty (radians), from the fit's covariance.
    pub sigma: f32,
}

impl AxisEstimate {
    /// Construct an [`AxisEstimate`].
    #[inline]
    pub fn new(angle: f32, sigma: f32) -> Self {
        Self { angle, sigma }
    }
}

/// Describes a detected chessboard corner in full-resolution image coordinates.
///
/// # Axis polarity convention
///
/// Local chessboard corner intensity patterns have exact 180° symmetry,
/// so assigning an absolute [0, 2π) direction to any single axis ray
/// is not possible from ring-local data. Instead the two axes are
/// reported jointly:
///
/// - `axes[0].angle` lies in `[0, π)` — the "line direction" of axis 1.
/// - `axes[1].angle` lies in `(axes[0].angle, axes[0].angle + π) ⊂ [0, 2π)`.
///
/// Together they satisfy: rotating CCW (in the usual `atan2(dy, dx)`
/// sense — note: in image pixel coordinates with y-axis pointing down,
/// this is a clockwise visual rotation) from `axes[0].angle` toward
/// `axes[1].angle` traverses a **dark** sector of the corner. The
/// second half-turn (`axes[0].angle + π → axes[1].angle + π`) crosses
/// the second dark sector; the two remaining sectors are bright.
///
/// Each axis direction is signed as a f32 in `[0, 2π)`; the axes are
/// **not** assumed orthogonal (holds up under projective warp).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    /// Subpixel y position in full-resolution image pixels.
    pub y: f32,

    /// Raw, **unnormalized** ChESS response `R = SR − DR − 16·MR` at
    /// the detected peak (see [`crate::response::chess_response_u8`]).
    /// Units are 8-bit pixel sums; data-dependent. `R > 0` is the
    /// paper's corner-acceptance criterion and the only contract this
    /// field satisfies — do not interpret it as a probability, a
    /// contrast, or a normalized strength.
    pub response: f32,

    /// Bright/dark amplitude (`|A|`, ≥ 0) recovered by the two-axis
    /// tanh fit (see [`crate::orientation`]). Units are gray levels.
    /// Larger means a stronger bright/dark separation at the ring
    /// radius. This is an independent quantity from [`Self::response`]
    /// — they are computed by different estimators and must not be
    /// compared against each other or against the same threshold.
    pub contrast: f32,

    /// RMS fit residual of the two-axis intensity model (in gray levels).
    /// Smaller = tighter match to an ideal chessboard corner.
    pub fit_rms: f32,

    /// The two local grid axis directions with per-axis 1σ precision.
    pub axes: [AxisEstimate; 2],
}

impl CornerDescriptor {
    /// Construct a [`CornerDescriptor`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f32,
        y: f32,
        response: f32,
        contrast: f32,
        fit_rms: f32,
        axes: [AxisEstimate; 2],
    ) -> Self {
        Self {
            x,
            y,
            response,
            contrast,
            fit_rms,
            axes,
        }
    }
}

/// Convert raw corner candidates into full descriptors by sampling the source image.
///
/// Orientation and polarity follow the conventions documented on [`CornerDescriptor`].
/// The default [`OrientationMethod::RingFit`] is used for all corners. When
/// [`OrientationMethod::DiskFit`] is requested for more than 80 candidates,
/// only the 80 strongest candidates run the full-disk pass; lower-ranked
/// candidates use the cheaper [`OrientationMethod::RingFit`] fallback.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "info",
        skip(img, corners),
        fields(corners = corners.len(), method = ?method)
    )
)]
pub fn corners_to_descriptors_with_method(
    img: &[u8],
    w: usize,
    h: usize,
    radius: u32,
    corners: Vec<Corner>,
    method: OrientationMethod,
) -> Vec<CornerDescriptor> {
    let ring = ring_offsets(radius);
    let ring_phi = ring_angles(ring);
    let full_disk_mask = full_disk_top_response_mask(&corners, method);

    // Closure that computes one CornerDescriptor given its index and Corner.
    // Captures all shared read-only state; img/ring/ring_phi/full_disk_mask are
    // all Sync, so both the rayon and serial paths compile cleanly.
    let describe = |(idx, c): (usize, Corner)| -> CornerDescriptor {
        let samples = sample_ring(img, w, h, c.x, c.y, ring);
        let orientation_method = match &full_disk_mask {
            Some(mask) if !mask[idx] => OrientationMethod::RingFit,
            _ => method,
        };
        let fit = fit_two_axes_with_method(
            &samples,
            &ring_phi,
            img,
            w,
            h,
            c.x,
            c.y,
            radius,
            orientation_method,
        );
        CornerDescriptor {
            x: c.x,
            y: c.y,
            response: c.strength,
            contrast: fit.amp.max(0.0),
            fit_rms: fit.rms,
            axes: [
                AxisEstimate {
                    angle: fit.theta1,
                    sigma: fit.sigma_theta1,
                },
                AxisEstimate {
                    angle: fit.theta2,
                    sigma: fit.sigma_theta2,
                },
            ],
        }
    };

    // Rayon path: into_par_iter().enumerate().map().collect() preserves order.
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

/// Sample the 16-point ChESS ring around (x, y) using bilinear interpolation.
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

/// Crate-internal entry point preserved for the existing in-file
/// test suite. Delegates to the ring-fit orientation in
/// [`crate::orientation`].
#[cfg(test)]
#[inline]
pub(crate) fn fit_two_axes(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    orientation::ring_fit_for_descriptor(samples, ring_phi)
}

/// Crate-internal dispatcher used by
/// [`corners_to_descriptors_with_method`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn fit_two_axes_with_method(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
    method: OrientationMethod,
) -> TwoAxisFit {
    match method {
        OrientationMethod::RingFit => orientation::ring_fit_for_descriptor(samples, ring_phi),
        OrientationMethod::DiskFit => {
            orientation::disk_fit_for_descriptor(img, w, h, cx, cy, radius, samples, ring_phi)
        }
    }
}

/// Test-only canonicalization shim. Re-exposes the implementation
/// that lives in [`crate::orientation`] under the name the
/// in-file tests use.
#[cfg(test)]
#[inline]
fn canonicalize(theta1: f32, theta2: f32, amp: f32) -> (f32, f32, f32) {
    orientation::ring_fit_canonicalize_for_test(theta1, theta2, amp)
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
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

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

        // axes[0].angle ∈ [0, π), axes[1].angle ∈ (θ1, θ1 + π)
        assert!(fit.theta1 >= 0.0 && fit.theta1 < PI);
        assert!(fit.theta2 > fit.theta1 && fit.theta2 < fit.theta1 + PI);
        // The two axes should match the input (possibly reduced by π).
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
        // Axes at 30° and 100° (70° between). Projective-like.
        let t1 = 30f32.to_radians();
        let t2 = 100f32.to_radians();
        let (samples, phi) = synthetic_ring(120.0, 60.0, t1, t2);
        let fit = fit_two_axes(&samples, &phi);

        // Acceptance: both axes recovered within 0.05 rad after folding mod π.
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
        // Same axis configuration with bright/dark swapped (A negated).
        // The two underlying *lines* must match between the two fits
        // (unordered, mod π). The polar assignment of axes[0] vs axes[1]
        // may differ because axes[0] is always the "dark-CCW" axis,
        // and that role flips with A's sign.
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
        // Feed unusual arguments and verify invariants hold.
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
        // Gaussian via sum of 6 uniforms (Irwin-Hall approximation, std=1 after rescale).
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

    #[test]
    fn disk_fit_mask_selects_only_top_response_budget() {
        let corners: Vec<Corner> = (0..(FULL_DISK_MAX_FULL_IMAGE_CORNERS + 3))
            .map(|i| Corner {
                x: i as f32,
                y: i as f32,
                strength: i as f32,
            })
            .collect();

        let mask = full_disk_top_response_mask(&corners, OrientationMethod::DiskFit)
            .expect("large DiskFit descriptor batch should be budgeted");

        assert_eq!(
            mask.iter().filter(|&&enabled| enabled).count(),
            FULL_DISK_MAX_FULL_IMAGE_CORNERS
        );
        assert!(!mask[0], "weakest candidate should fall back to RingFit");
        assert!(
            !mask[1],
            "second-weakest candidate should fall back to RingFit"
        );
        assert!(
            !mask[2],
            "third-weakest candidate should fall back to RingFit"
        );
        assert!(
            mask[FULL_DISK_MAX_FULL_IMAGE_CORNERS + 2],
            "strongest candidate should run DiskFit"
        );
    }
}
