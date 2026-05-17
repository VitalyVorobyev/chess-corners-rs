//! Gauss-Newton solver body and 4×4 linear algebra for the ring fit.

use super::TwoAxisFit;
use super::TANH_BETA;
use core::f32::consts::{PI, TAU};

/// Maximum Gauss-Newton iterations in the two-axis fit.
const GN_MAX_ITERS: usize = 6;

/// Cap on angular step per Gauss-Newton iteration (radians). Prevents
/// runaway updates near pathological configurations.
const GN_MAX_ANGLE_STEP: f32 = 0.5;

/// Convergence threshold on angular updates (radians).
const GN_TOL: f32 = 1e-4;

/// Run the Gauss-Newton solver from the supplied initial parameters.
///
/// This is the body of the baseline fit, factored out so other
/// variants (structure-tensor seeding, robust grid seeding, etc.) can
/// reuse the exact same numerical machinery while supplying their own
/// seed.
///
/// `centered_var` is the sample variance, used only to size the
/// fallback `degenerate_fit` return when `amp_seed` is too small.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_with_seed(
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

    if !amp.is_finite() || amp.abs() < 1e-4 || centered_var < 1e-6 {
        return degenerate_fit(centered_var);
    }

    // Trig hoist: ring-phi sin/cos depend only on the (fixed) ring
    // geometry, not on the iterating GN parameters.
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

        let mut d = delta;
        if d[2].abs() > GN_MAX_ANGLE_STEP {
            d[2] = d[2].signum() * GN_MAX_ANGLE_STEP;
        }
        if d[3].abs() > GN_MAX_ANGLE_STEP {
            d[3] = d[3].signum() * GN_MAX_ANGLE_STEP;
        }

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
pub(super) fn degenerate_fit(centered_var: f32) -> TwoAxisFit {
    TwoAxisFit {
        amp: 0.0,
        theta1: 0.0,
        theta2: core::f32::consts::FRAC_PI_2,
        sigma_theta1: PI,
        sigma_theta2: PI,
        rms: centered_var.max(0.0).sqrt(),
    }
}

/// Canonicalize `(θ₁, θ₂, A)` into the convention documented on
/// [`crate::CornerDescriptor`]. Returns `(θ₁, θ₂, |A|)` with:
/// * `θ₁ ∈ [0, π)`
/// * `θ₂ ∈ (θ₁, θ₁ + π) ⊂ [0, 2π)`
/// * the sector `(θ₁, θ₂)` (CCW) is a dark sector of the corner.
///
/// Both [`super::fit_ring`] and the full-disk estimator
/// (`crate::orientation::disk_sector::fit`) route their raw `(θ₁, θ₂, A)`
/// through this function, so the public [`crate::AxisFitResult`]
/// observes the same convention regardless of which
/// [`crate::OrientationMethod`] produced it.
///
/// # Implementation notes
///
/// The intensity model `μ + A·tanh(sin(φ−θ₁))·tanh(sin(φ−θ₂))` admits
/// three model-preserving symmetries:
/// * Symmetric shift `(θ₁, θ₂) ↦ (θ₁ + π, θ₂ + π)` — both `tanh` factors
///   flip sign, the product is unchanged.
/// * Single-axis flip with amplitude flip
///   `(θ₁, θ₂, A) ↦ (θ₁ + π, θ₂, −A)` — one `tanh` factor flips sign,
///   `A` flips, the product is unchanged.
/// * Swap `(θ₁, θ₂) ↔ (θ₂, θ₁)` — multiplication commutes.
///
/// To put the input on the canonical branch we (1) make `A` non-negative
/// via the single-axis-flip symmetry, then (2) ensure the CCW sector
/// `(θ₁, θ₂)` is the *short* one (`diff < π`). The short-sector step uses
/// the `(θ₁, θ₂) ↔ (θ₂, θ₁)` swap rather than a single-axis flip:
/// swapping is a no-op on the model, while shifting `θ₁` by `π` without
/// also flipping `A` *does* change the model and would mislabel the dark
/// sector for the antipodal branch. Finally `θ₁` is folded into `[0, π)`
/// via the symmetric-shift symmetry.
pub(crate) fn canonicalize(theta1: f32, theta2: f32, amp: f32) -> (f32, f32, f32) {
    let mut t1 = wrap_to_tau(theta1);
    let mut t2 = wrap_to_tau(theta2);
    let mut a = amp;

    if a < 0.0 {
        t1 = wrap_to_tau(t1 + PI);
        a = -a;
    }

    let mut diff = (t2 - t1).rem_euclid(TAU);
    // INVARIANT: when the CCW arc `(t1, t2)` is the *long* one, swap the
    // axes so the canonical pair always describes the short arc. Swapping
    // is the only model-preserving way to do this. Earlier revisions
    // shifted `t1` by `π` here without flipping `A`, which silently
    // re-labels which of the two dark/bright sectors is canonical and
    // produced antipodal-flipped axes for inputs equivalent to the
    // ring fit's canonical seed but with `(t1, t2)` arriving in
    // descending order — see `crate::orientation::tests::
    // canonical_axes_match_across_methods`.
    if diff >= PI {
        core::mem::swap(&mut t1, &mut t2);
        diff = (t2 - t1).rem_euclid(TAU);
    }

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

/// Solve a 4x4 linear system `A x = b` with partial pivoting.
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

/// Invert a 4x4 matrix by solving `A X = I` column-wise. Returns
/// `None` on singularity.
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
