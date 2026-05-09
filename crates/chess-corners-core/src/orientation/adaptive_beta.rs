//! V1 — adaptive-β two-axis orientation fit.
//!
//! Promotes the tanh slope `β` from a fixed constant ([`super::baseline`]
//! hardcodes `β = 4`) to a 5th Gauss-Newton parameter so the fit
//! adapts to the local edge sharpness implied by the upstream blur.
//! The Phase 2 baseline blur sweep showed a U-shape in the recovered
//! axis RMSE because the fixed `β = 4` is only optimal near
//! `σ_blur ≈ 2`; sharper edges want larger β, blurrier edges want
//! smaller β.
//!
//! Model (unchanged from baseline):
//! ```text
//! I(φ) = μ + A · tanh(β · sin(φ − θ₁)) · tanh(β · sin(φ − θ₂))
//! ```
//!
//! Five free parameters: `μ, A, θ₁, θ₂, β`.
//!
//! The Jacobian column for β is
//! ```text
//! ∂I/∂β = A · ( sech²₁ · sin(φ − θ₁) · h₂  +  h₁ · sech²₂ · sin(φ − θ₂) )
//! ```
//! where `hᵢ = tanh(β · sin(φ − θᵢ))` and `sech²ᵢ = 1 − hᵢ²`.
//!
//! The solver mirrors [`super::baseline::fit_with_seed`] but uses a 5×5
//! normal-equation solver, clamps |Δβ| per iteration, hard-clamps β to
//! `[BETA_MIN, BETA_MAX]`, and checks an additional convergence
//! criterion on `Δβ`. Outside the β degree of freedom and the residual
//! d.o.f. (`16 − 5 = 11` instead of `12`), behaviour matches baseline.

use super::baseline::{self, TwoAxisFit};
use core::f32::consts::PI;

/// Initial value for the tanh slope when the GN seed is built.
/// Matches the baseline's hardcoded `TANH_BETA` so the V1 fit starts
/// from the same effective model the legacy code converges to on the
/// `σ_blur ≈ 2` regime.
const BETA_INIT: f32 = 4.0;

/// Minimum admissible β. Below this the tanh is essentially linear
/// and the model becomes degenerate (`A` and `β` collapse into the
/// same direction in parameter space).
const BETA_MIN: f32 = 0.5;

/// Maximum admissible β. Above this the tanh saturates: `sech²(β·s)`
/// vanishes everywhere except at `s ≈ 0`, the β column of the
/// Jacobian goes to zero and the system becomes singular.
const BETA_MAX: f32 = 16.0;

/// Maximum Gauss-Newton iterations (matches baseline).
const MAX_ITERS: usize = 6;

/// Cap on angular step per iteration (radians), matches baseline.
const MAX_ANGLE_STEP: f32 = 0.5;

/// Cap on |Δβ| per iteration so the GN cannot leap across the
/// `BETA_MIN`/`BETA_MAX` envelope on a single noisy update.
const MAX_BETA_STEP: f32 = 2.0;

/// Convergence threshold on angular updates (radians, matches baseline).
const GN_TOL: f32 = 1e-4;

/// Convergence threshold on `ΔA` (matches baseline).
const AMP_TOL: f32 = 1e-4;

/// Convergence threshold on `Δβ`. Stricter than the angular tolerance
/// because β has the largest dynamic range (~30×) of all five
/// parameters.
const BETA_TOL: f32 = 1e-3;

/// Pivot threshold inside the 5×5 solver (matches baseline's 4×4).
const PIVOT_FLOOR: f32 = 1e-20;

/// Run the adaptive-β fit on pre-sampled ring values.
///
/// Returns the same [`TwoAxisFit`] shape as the baseline fit so the
/// dispatcher in [`super::api`] can normalise both into an
/// `AxisFitResult`. The recovered β is *not* part of the public
/// result — it is consumed entirely by the σ extraction.
pub(crate) fn fit(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
    let mean = samples.iter().copied().sum::<f32>() / 16.0;
    let centered_var = samples
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / 16.0;

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
        BETA_INIT,
        centered_var,
    )
}

/// Run the 5-parameter Gauss-Newton solver from supplied seeds.
#[allow(clippy::too_many_arguments)]
fn fit_with_seed(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    theta1_seed: f32,
    theta2_seed: f32,
    amp_seed: f32,
    mu_seed: f32,
    beta_seed: f32,
    centered_var: f32,
) -> TwoAxisFit {
    let mut mu = mu_seed;
    let mut amp = amp_seed;
    let mut theta1 = theta1_seed;
    let mut theta2 = theta2_seed;
    let mut beta = beta_seed.clamp(BETA_MIN, BETA_MAX);

    // Match the baseline's degenerate sentinel exactly so a flat ring
    // does not cause the V1 dispatcher to diverge from the baseline
    // contract.
    if !amp.is_finite() || amp.abs() < 1e-4 || centered_var < 1e-6 {
        return degenerate_fit(centered_var);
    }

    let mut jtj: [[f32; 5]; 5];
    let mut ssr: f32;
    let mut iter = 0usize;

    loop {
        jtj = [[0f32; 5]; 5];
        let mut jtr = [0f32; 5];
        ssr = 0.0;

        for i in 0..16 {
            let phi = ring_phi[i];
            let s1 = (phi - theta1).sin();
            let s2 = (phi - theta2).sin();
            let c1 = (phi - theta1).cos();
            let c2 = (phi - theta2).cos();

            let h1 = (beta * s1).tanh();
            let h2 = (beta * s2).tanh();
            let sech2_1 = 1.0 - h1 * h1;
            let sech2_2 = 1.0 - h2 * h2;

            let pred = mu + amp * h1 * h2;
            let r = samples[i] - pred;
            ssr += r * r;

            // Columns: [μ, A, θ₁, θ₂, β]
            let j = [
                1.0,
                h1 * h2,
                -amp * sech2_1 * beta * c1 * h2,
                -amp * h1 * sech2_2 * beta * c2,
                amp * (sech2_1 * s1 * h2 + h1 * sech2_2 * s2),
            ];

            for k in 0..5 {
                jtr[k] += j[k] * r;
                for l in 0..5 {
                    jtj[k][l] += j[k] * j[l];
                }
            }
        }

        if iter >= MAX_ITERS {
            break;
        }

        let delta = match solve_5x5(&jtj, &jtr) {
            Some(d) => d,
            None => break,
        };

        let mut d = delta;
        if d[2].abs() > MAX_ANGLE_STEP {
            d[2] = d[2].signum() * MAX_ANGLE_STEP;
        }
        if d[3].abs() > MAX_ANGLE_STEP {
            d[3] = d[3].signum() * MAX_ANGLE_STEP;
        }
        if d[4].abs() > MAX_BETA_STEP {
            d[4] = d[4].signum() * MAX_BETA_STEP;
        }

        if d[2].abs() < GN_TOL
            && d[3].abs() < GN_TOL
            && d[1].abs() < AMP_TOL
            && d[4].abs() < BETA_TOL
        {
            break;
        }

        mu += d[0];
        amp += d[1];
        theta1 += d[2];
        theta2 += d[3];
        beta = (beta + d[4]).clamp(BETA_MIN, BETA_MAX);
        iter += 1;
    }

    // Residual d.o.f. drops to N − params = 16 − 5 = 11.
    let residual_var = ssr / 11.0;
    let cov = invert_5x5(&jtj);
    let (sigma_t1, sigma_t2) = match cov {
        Some(c) => (
            (residual_var * c[2][2].max(0.0)).sqrt(),
            (residual_var * c[3][3].max(0.0)).sqrt(),
        ),
        None => (PI, PI),
    };

    let rms = (ssr / 16.0).sqrt();

    let (theta1_c, theta2_c, amp_c) = baseline::canonicalize(theta1, theta2, amp);
    let _ = mu;
    let _ = beta;

    TwoAxisFit {
        amp: amp_c,
        theta1: theta1_c,
        theta2: theta2_c,
        sigma_theta1: sigma_t1.min(PI),
        sigma_theta2: sigma_t2.min(PI),
        rms,
    }
}

/// Same fallback as the baseline degenerate path, replicated locally
/// so the V1 module is self-contained (no extra `pub(super)` surface
/// on baseline beyond the existing `canonicalize`/`fit_with_seed`).
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

/// 2nd-harmonic seed estimator. Identical to the baseline helper —
/// duplicated rather than re-exported to keep the variant
/// self-contained per the plan's "self-contained variant" guidance.
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
    let mag = (c2 * c2 + s2 * s2).sqrt() / 8.0;
    (theta, mag)
}

/// Solve a 5×5 linear system `A x = b` via Gauss elimination with
/// partial pivoting. Mirrors the baseline's `solve_4x4`: same pivot
/// floor (`PIVOT_FLOOR`), same `Option<…>` return on singularity,
/// dimension bumped to 5.
pub(crate) fn solve_5x5(a: &[[f32; 5]; 5], b: &[f32; 5]) -> Option<[f32; 5]> {
    let mut m = [[0f32; 6]; 5];
    for (i, row) in m.iter_mut().enumerate() {
        row[..5].copy_from_slice(&a[i]);
        row[5] = b[i];
    }
    for i in 0..5 {
        let mut max_row = i;
        for k in (i + 1)..5 {
            if m[k][i].abs() > m[max_row][i].abs() {
                max_row = k;
            }
        }
        m.swap(i, max_row);
        let pivot = m[i][i];
        if !pivot.is_finite() || pivot.abs() < PIVOT_FLOOR {
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
    Some([m[0][5], m[1][5], m[2][5], m[3][5], m[4][5]])
}

/// Invert a 5×5 matrix by solving `A X = I` column-wise.
fn invert_5x5(a: &[[f32; 5]; 5]) -> Option<[[f32; 5]; 5]> {
    let mut inv = [[0f32; 5]; 5];
    for col in 0..5 {
        let mut b = [0f32; 5];
        b[col] = 1.0;
        let x = solve_5x5(a, &b)?;
        for (row_idx, inv_row) in inv.iter_mut().enumerate() {
            inv_row[col] = x[row_idx];
        }
    }
    Some(inv)
}
