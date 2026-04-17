//! Corner descriptors and helpers for chessboard detection.
//!
//! This module turns raw ChESS corner candidates into richer
//! [`CornerDescriptor`] values that carry subpixel position, response,
//! contrast, and two-axis orientation with per-axis precision.
//!
//! The detector in [`crate::detect`] produces intermediate [`Corner`]
//! values; [`corners_to_descriptors`] then samples the original image
//! on a ChESS ring around each corner and fits a parametric two-axis
//! intensity model to estimate the local grid geometry.
use crate::ring::ring_offsets;
#[cfg(feature = "tracing")]
use tracing::instrument;

use core::f32::consts::{PI, TAU};

/// Smoothing slope of the tanh sign approximation used in the intensity
/// model. Fixed constant — not a fit parameter. Reflects the effective
/// ring-integration blur at the sampled radius.
const TANH_BETA: f32 = 4.0;

/// Maximum Gauss-Newton iterations in the two-axis fit.
const GN_MAX_ITERS: usize = 6;

/// Cap on angular step per Gauss-Newton iteration (radians). Prevents
/// runaway updates near pathological configurations.
const GN_MAX_ANGLE_STEP: f32 = 0.5;

/// Convergence threshold on angular updates (radians).
const GN_TOL: f32 = 1e-4;

/// A detected ChESS corner (subpixel).
#[derive(Clone, Debug)]
pub struct Corner {
    /// Subpixel x coordinate in image pixels.
    pub x: f32,
    /// Subpixel y coordinate in image pixels.
    pub y: f32,
    /// Raw ChESS response at the integer peak (before refinement).
    pub strength: f32,
}

/// Direction of one local grid axis with its 1σ angular uncertainty.
#[derive(Clone, Copy, Debug)]
pub struct AxisEstimate {
    /// Axis direction, radians in [0, 2π).
    ///
    /// See [`CornerDescriptor`] for the joint polarity convention.
    pub angle: f32,
    /// 1σ angular uncertainty (radians), from the fit's covariance.
    pub sigma: f32,
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
pub struct CornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    pub y: f32,

    /// Raw, **unnormalized** ChESS response `R = SR − DR − 16·MR` at
    /// the detected peak (see [`crate::response::chess_response_u8`]).
    /// Units are 8-bit pixel sums; data-dependent. `R > 0` is the
    /// paper's corner-acceptance criterion and the only contract this
    /// field satisfies — do not interpret it as a probability, a
    /// contrast, or a normalized strength.
    pub response: f32,

    /// Bright/dark amplitude (`|A|`, ≥ 0) recovered by the two-axis
    /// tanh fit in [`fit_two_axes`]. Units are gray levels. Larger
    /// means a stronger bright/dark separation at the ring radius.
    /// This is an independent quantity from [`Self::response`] — they
    /// are computed by different estimators and must not be compared
    /// against each other or against the same threshold.
    pub contrast: f32,

    /// RMS fit residual of the two-axis intensity model (in gray levels).
    /// Smaller = tighter match to an ideal chessboard corner.
    pub fit_rms: f32,

    /// The two local grid axis directions with per-axis 1σ precision.
    pub axes: [AxisEstimate; 2],
}

/// Convert raw corner candidates into full descriptors by sampling the source image.
///
/// Orientation and polarity follow the conventions documented on [`CornerDescriptor`].
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "info",
        skip(img, corners),
        fields(corners = corners.len())
    )
)]
pub fn corners_to_descriptors(
    img: &[u8],
    w: usize,
    h: usize,
    radius: u32,
    corners: Vec<Corner>,
) -> Vec<CornerDescriptor> {
    let ring = ring_offsets(radius);
    let ring_phi = ring_angles(ring);
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let samples = sample_ring(img, w, h, c.x, c.y, ring);
        let fit = fit_two_axes(&samples, &ring_phi);

        out.push(CornerDescriptor {
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
        });
    }
    out
}

/// Sample the 16-point ChESS ring around (x, y) using bilinear interpolation.
fn sample_ring(
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
fn ring_angles(ring: &[(i32, i32); 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        out[i] = (dy as f32).atan2(dx as f32);
    }
    out
}

/// Result of the parametric two-axis intensity fit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwoAxisFit {
    pub amp: f32,
    pub theta1: f32,
    pub theta2: f32,
    pub sigma_theta1: f32,
    pub sigma_theta2: f32,
    pub rms: f32,
}

/// Fit a two-axis chessboard-corner intensity model to 16 ring samples.
///
/// Model: `I(φ) = μ + A · tanh(β · sin(φ − θ₁)) · tanh(β · sin(φ − θ₂))`,
/// with fixed `β = TANH_BETA`. Four free parameters: `μ, A, θ₁, θ₂`.
/// Solved via Gauss-Newton seeded from the 2nd-harmonic orientation.
/// Returns canonicalized angles plus per-axis 1σ uncertainties.
pub(crate) fn fit_two_axes(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    let mean = samples.iter().copied().sum::<f32>() / 16.0;
    let centered_var = samples
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / 16.0;

    // 2nd-harmonic seed: theta_seed ∈ [-π/2, π/2] is a SECTOR MIDPOINT
    // direction; the two grid axes bracket it at ±π/4.
    let (theta_seed, harmonic_mag) = second_harmonic(samples, ring_phi, mean);
    let mut mu = mean;
    let mut amp = harmonic_mag; // positive magnitude seed
    let mut theta1 = theta_seed + core::f32::consts::FRAC_PI_4;
    let mut theta2 = theta_seed + 3.0 * core::f32::consts::FRAC_PI_4;

    // If the 2nd-harmonic "signal" is negligible, return a degenerate fit.
    if !amp.is_finite() || amp.abs() < 1e-4 || centered_var < 1e-6 {
        return degenerate_fit(centered_var);
    }

    let mut jtj: [[f32; 4]; 4];
    let mut ssr: f32;
    let mut iter = 0usize;

    loop {
        // Evaluate residual/Jacobian at current state.
        jtj = [[0f32; 4]; 4];
        let mut jtr = [0f32; 4];
        ssr = 0.0;

        for i in 0..16 {
            let phi = ring_phi[i];
            let s1a = (phi - theta1).sin();
            let s2a = (phi - theta2).sin();
            let c1a = (phi - theta1).cos();
            let c2a = (phi - theta2).cos();

            let h1 = (TANH_BETA * s1a).tanh();
            let h2 = (TANH_BETA * s2a).tanh();
            let sech2_1 = 1.0 - h1 * h1;
            let sech2_2 = 1.0 - h2 * h2;

            let pred = mu + amp * h1 * h2;
            let r = samples[i] - pred;
            ssr += r * r;

            let j = [
                1.0,
                h1 * h2,
                -amp * sech2_1 * TANH_BETA * c1a * h2,
                -amp * h1 * sech2_2 * TANH_BETA * c2a,
            ];

            for k in 0..4 {
                jtr[k] += j[k] * r;
                for l in 0..4 {
                    jtj[k][l] += j[k] * j[l];
                }
            }
        }

        if iter >= GN_MAX_ITERS {
            break;
        }

        let delta = match solve_4x4(&jtj, &jtr) {
            Some(d) => d,
            None => break,
        };

        // Clamp angular steps to avoid runaway.
        let mut d = delta;
        if d[2].abs() > GN_MAX_ANGLE_STEP {
            d[2] = d[2].signum() * GN_MAX_ANGLE_STEP;
        }
        if d[3].abs() > GN_MAX_ANGLE_STEP {
            d[3] = d[3].signum() * GN_MAX_ANGLE_STEP;
        }

        // If the proposed step is negligible we are already at the
        // optimum of the current linearisation; keep the evaluated
        // jtj/ssr and stop.
        if d[2].abs() < GN_TOL && d[3].abs() < GN_TOL && d[1].abs() < 1e-4 {
            break;
        }

        mu += d[0];
        amp += d[1];
        theta1 += d[2];
        theta2 += d[3];
        iter += 1;
    }

    // Covariance from the final Hessian. Degrees of freedom = N − params = 12.
    let residual_var = ssr / 12.0;
    let cov = invert_4x4(&jtj);
    let (sigma_t1, sigma_t2) = match cov {
        Some(c) => (
            (residual_var * c[2][2].max(0.0)).sqrt(),
            (residual_var * c[3][3].max(0.0)).sqrt(),
        ),
        None => (PI, PI),
    };

    let rms = (ssr / 16.0).sqrt();

    let (theta1_c, theta2_c, amp_c) = canonicalize(theta1, theta2, amp);
    let _ = mu;

    TwoAxisFit {
        amp: amp_c,
        theta1: theta1_c,
        theta2: theta2_c,
        sigma_theta1: sigma_t1.min(PI),
        sigma_theta2: sigma_t2.min(PI),
        rms,
    }
}

/// Fallback result for flat / no-contrast inputs.
fn degenerate_fit(centered_var: f32) -> TwoAxisFit {
    TwoAxisFit {
        amp: 0.0,
        theta1: 0.0,
        theta2: core::f32::consts::FRAC_PI_2,
        sigma_theta1: PI,
        sigma_theta2: PI,
        rms: centered_var.max(0.0).sqrt(),
    }
}

/// Canonicalize (θ₁, θ₂, A) into the convention documented on
/// [`CornerDescriptor`]. Returns `(θ₁, θ₂, |A|)` with:
/// * θ₁ ∈ [0, π)
/// * θ₂ ∈ (θ₁, θ₁ + π) ⊂ [0, 2π)
/// * the sector (θ₁, θ₂) (CCW) is a dark sector of the corner
fn canonicalize(theta1: f32, theta2: f32, amp: f32) -> (f32, f32, f32) {
    let mut t1 = wrap_to_tau(theta1);
    let mut t2 = wrap_to_tau(theta2);
    let mut a = amp;

    // Force A > 0 by flipping one axis's polarity.
    if a < 0.0 {
        t1 = wrap_to_tau(t1 + PI);
        a = -a;
    }

    // With A > 0, dark sectors sit where sin(φ − θ₁)·sin(φ − θ₂) is
    // negative — two antipodal sectors. Pick θ₁'s representative ray
    // so that the CCW-adjacent dark sector ends at θ₂ within a
    // rotation of less than π.
    let mut diff = (t2 - t1).rem_euclid(TAU);
    if diff >= PI {
        t1 = wrap_to_tau(t1 + PI);
        diff = (t2 - t1).rem_euclid(TAU);
    }

    // Reduce θ₁ into [0, π) via π-wraparound (antipode ray of the same
    // line axis). Rebuild θ₂ so it still marks the dark-sector end.
    t1 = t1.rem_euclid(PI);
    t2 = t1 + diff;

    (t1, t2, a)
}

#[inline]
fn wrap_to_tau(mut x: f32) -> f32 {
    x = x.rem_euclid(TAU);
    if !x.is_finite() {
        return 0.0;
    }
    x
}

/// 2nd-harmonic estimate: returns `(theta_seed, magnitude)` where
/// `theta_seed` is a sector-midpoint direction (mod π) and magnitude
/// is a non-negative estimate of the A parameter amplitude.
fn second_harmonic(samples: &[f32; 16], ring_phi: &[f32; 16], mean: f32) -> (f32, f32) {
    let mut c2 = 0.0f32;
    let mut s2 = 0.0f32;
    for i in 0..16 {
        let v = samples[i] - mean;
        c2 += v * (2.0 * ring_phi[i]).cos();
        s2 += v * (2.0 * ring_phi[i]).sin();
    }
    let mut theta = 0.5 * s2.atan2(c2);
    if !theta.is_finite() {
        theta = 0.0;
    }
    // Magnitude of the complex Fourier coefficient, normalised by
    // Σ cos²(2φᵢ) ≈ 8 for 16 samples. Always non-negative.
    let mag = (c2 * c2 + s2 * s2).sqrt() / 8.0;
    (theta, mag)
}

/// Solve a 4x4 linear system A x = b with partial pivoting.
fn solve_4x4(a: &[[f32; 4]; 4], b: &[f32; 4]) -> Option<[f32; 4]> {
    let mut m = [[0f32; 5]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[..4].copy_from_slice(&a[i]);
        row[4] = b[i];
    }
    for i in 0..4 {
        let mut max_row = i;
        for k in (i + 1)..4 {
            if m[k][i].abs() > m[max_row][i].abs() {
                max_row = k;
            }
        }
        m.swap(i, max_row);
        let pivot = m[i][i];
        if !pivot.is_finite() || pivot.abs() < 1e-20 {
            return None;
        }
        let inv_pivot = 1.0 / pivot;
        for v in &mut m[i] {
            *v *= inv_pivot;
        }
        let pivot_row = m[i];
        for (k, row) in m.iter_mut().enumerate() {
            if k == i {
                continue;
            }
            let f = row[i];
            if f == 0.0 {
                continue;
            }
            for (rv, &pv) in row.iter_mut().zip(pivot_row.iter()) {
                *rv -= f * pv;
            }
        }
    }
    Some([m[0][4], m[1][4], m[2][4], m[3][4]])
}

/// Invert a 4x4 matrix by solving `A X = I` column-wise. Returns `None` on singularity.
fn invert_4x4(a: &[[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0f32; 4]; 4];
    for col in 0..4 {
        let mut b = [0f32; 4];
        b[col] = 1.0;
        let x = solve_4x4(a, &b)?;
        for (row_idx, inv_row) in inv.iter_mut().enumerate() {
            inv_row[col] = x[row_idx];
        }
    }
    Some(inv)
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
}
