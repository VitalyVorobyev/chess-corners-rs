//! Ring-fit two-axis orientation fit for [`super::OrientationMethod::RingFit`].
//!
//! Fits the parametric two-axis chessboard-corner intensity model
//! `I(φ) = μ + A·tanh(β·sin(φ−θ₁))·tanh(β·sin(φ−θ₂))` to the 16
//! ring samples around a detected corner, solving for μ, A, θ₁, θ₂
//! via Gauss-Newton seeded from the 2nd-harmonic orientation.
//!
//! After the Gauss-Newton solve, the raw CRLB per-axis 1σ uncertainties
//! are rescaled by a piecewise-linear LUT keyed on the contrast-relative
//! residual `fit_rms / max(|amp|, ε)`, calibrating the reported sigmas
//! closer to the empirical RMSE. Recovered angles, amplitude, and
//! `fit_rms` are bit-identical to the raw Gauss-Newton output.
//!
//! The Gauss-Newton body is factored into [`fit_with_seed`] so the disk
//! estimator can supply its own initial parameters and reuse the same
//! numerical machinery.

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
pub(crate) fn fit(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    let mean = samples.iter().copied().sum::<f32>() / 16.0;
    let centered_var = samples
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / 16.0;

    // 2nd-harmonic seed: theta_seed ∈ [-π/2, π/2] is a SECTOR MIDPOINT
    // direction; the two grid axes bracket it at ±π/4.
    let (theta_seed, harmonic_mag) = second_harmonic(samples, ring_phi, mean);
    let mu_seed = mean;
    let amp_seed = harmonic_mag;
    let theta1_seed = theta_seed + core::f32::consts::FRAC_PI_4;
    let theta2_seed = theta_seed + 3.0 * core::f32::consts::FRAC_PI_4;

    fit_with_seed(
        samples,
        ring_phi,
        theta1_seed,
        theta2_seed,
        amp_seed,
        mu_seed,
        centered_var,
    )
}

/// Run the Gauss-Newton solver from the supplied initial parameters.
///
/// This is the body of the baseline fit, factored out so other
/// variants (structure-tensor seeding, etc.) can reuse the exact same
/// numerical machinery while supplying their own seed. Behaviour for
/// the baseline path is preserved bit-identically: [`fit`] simply
/// computes the 2nd-harmonic seed and delegates here.
///
/// `centered_var` is the sample variance, used only to size the
/// fallback `degenerate_fit` return when `amp_seed` is too small.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_with_seed(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    theta1_seed: f32,
    theta2_seed: f32,
    amp_seed: f32,
    mu_seed: f32,
    centered_var: f32,
) -> TwoAxisFit {
    let mut mu = mu_seed;
    let mut amp = amp_seed;
    let mut theta1 = theta1_seed;
    let mut theta2 = theta2_seed;

    // If the seed "signal" is negligible, return a degenerate fit.
    if !amp.is_finite() || amp.abs() < 1e-4 || centered_var < 1e-6 {
        return degenerate_fit(centered_var);
    }

    // Trig hoist: ring-phi sin/cos depend only on the (fixed) ring
    // geometry, not on the iterating GN parameters. Compute once here
    // and use the angle-difference identities inside the GN loop:
    //   sin(φ − θ) = sinφ·cosθ − cosφ·sinθ
    //   cos(φ − θ) = cosφ·cosθ + sinφ·sinθ
    // Trades 4 trig calls per pixel per iteration (≈ 384 per fit) for
    // 32 trig calls total + 4 trig calls per iteration, leaving the
    // recovered angles within a few ULP of the direct evaluation.
    let mut sphi = [0.0f32; 16];
    let mut cphi = [0.0f32; 16];
    for i in 0..16 {
        let (s, c) = ring_phi[i].sin_cos();
        sphi[i] = s;
        cphi[i] = c;
    }

    let mut jtj: [[f32; 4]; 4];
    let mut ssr: f32;
    let mut iter = 0usize;

    loop {
        // Evaluate residual/Jacobian at current state.
        jtj = [[0f32; 4]; 4];
        let mut jtr = [0f32; 4];
        ssr = 0.0;

        let (s_t1, c_t1) = theta1.sin_cos();
        let (s_t2, c_t2) = theta2.sin_cos();

        for i in 0..16 {
            let s1a = sphi[i] * c_t1 - cphi[i] * s_t1;
            let c1a = cphi[i] * c_t1 + sphi[i] * s_t1;
            let s2a = sphi[i] * c_t2 - cphi[i] * s_t2;
            let c2a = cphi[i] * c_t2 + sphi[i] * s_t2;

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
/// [`crate::descriptor::CornerDescriptor`]. Returns `(θ₁, θ₂, |A|)` with:
/// * θ₁ ∈ [0, π)
/// * θ₂ ∈ (θ₁, θ₁ + π) ⊂ [0, 2π)
/// * the sector (θ₁, θ₂) (CCW) is a dark sector of the corner
pub(crate) fn canonicalize(theta1: f32, theta2: f32, amp: f32) -> (f32, f32, f32) {
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

// ---------------------------------------------------------------------------
// Sigma-LUT post-processing (folded from the former sigma_correction module)
// ---------------------------------------------------------------------------

/// Run the Gauss-Newton ring fit and apply a piecewise-linear σ-correction
/// LUT to the raw CRLB per-axis uncertainties. Recovered angles, amplitude,
/// and `fit_rms` are bit-identical to the raw [`fit`] output; only
/// `sigma_theta1` and `sigma_theta2` are rescaled.
///
/// This is the implementation of [`super::OrientationMethod::RingFit`].
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
/// Cells with small relative residual (clean, sharp inputs) need a larger
/// correction than cells with large relative residual (heavy noise / model
/// mismatch) where the CRLB σ is already approximately correct.
/// The first and last entries pin the function for clamping; intermediate
/// entries linearly interpolate.
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
/// residual saturates the LUT and the multiplier becomes 1.0.
const AMP_FLOOR: f32 = 1.0;

/// Return the σ-correction multiplier for a given fit by piecewise-linear
/// interpolation of [`LUT`] keyed on `fit_rms / max(|amp|, AMP_FLOOR)`.
/// Non-finite inputs clamp to the first LUT entry (the "clean fit"
/// multiplier). The trailing `(INFINITY, 1.0)` entry signals "flat to the
/// right" past the last finite breakpoint.
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
