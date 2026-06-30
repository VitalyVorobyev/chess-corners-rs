//! Per-pair scoring functions and the `Fit` struct.

use super::super::ring_fit;
use super::data::DiskData;
use super::geometry::{line_separation, signed_line_delta, valid_pair};

pub(super) const WIDTHS_PX: [f32; 4] = [0.35, 0.70, 1.40, 2.80];

pub(super) const MIN_CONTRAST: f32 = 10.0;
pub(super) const MIN_CORR: f32 = 0.74;
pub(super) const ACCEPT_REL_MARGIN: f32 = 0.03;
pub(super) const ACCEPT_REL_RATIO: f32 = 0.92;
pub(super) const WEAK_DISAGREE_RAD: f32 = 12.0_f32.to_radians();
pub(super) const INFLATED_SIGMA_RAD: f32 = 10.0_f32.to_radians();

/// Upper bound on disk pixels for the per-candidate `tanh` cache.
/// At `MAX_SUPPORT_RADIUS = 8` a full disk has `≤ π·64 ≈ 201` pixels;
/// 320 leaves comfortable headroom. If `MAX_SUPPORT_RADIUS` grows past
/// ~9 this constant must grow with it (the `data.n <= MAX_TANH_PIXELS`
/// debug-assert in `best_disk_fit` guards the invariant; production
/// code falls back to the un-cached path).
pub(super) const MAX_TANH_PIXELS: usize = 320;

/// Candidate pair fit result.
#[derive(Clone, Copy, Debug)]
pub(super) struct Fit {
    pub(super) theta1: f32,
    pub(super) theta2: f32,
    pub(super) width: f32,
    pub(super) amp: f32,
    pub(super) rms: f32,
    pub(super) rel_rms: f32,
    pub(super) corr: f32,
    pub(super) edge_score: f32,
    pub(super) objective: f32,
}

/// Score a candidate pair `(theta0, theta1)` at `width` using the
/// un-cached (full tanh re-computation) path. Returns `None` if the
/// pair is invalid or the data are degenerate.
pub(super) fn score_pair(
    data: &DiskData,
    theta0: f32,
    theta1: f32,
    width: f32,
    edge_score: f32,
) -> Option<Fit> {
    if !valid_pair(theta0, theta1) {
        return None;
    }
    if data.val_energy <= 1e-9 {
        return None;
    }

    let (s0, c0) = theta0.sin_cos();
    let (s1, c1) = theta1.sin_cos();
    let mut sum_q = 0.0f32;
    let mut sum_q2 = 0.0f32;
    let mut dot = 0.0f32;

    for i in 0..data.n {
        let d0 = -s0 * data.xs[i] + c0 * data.ys[i];
        let d1 = -s1 * data.xs[i] + c1 * data.ys[i];
        let q = (d0 / width).tanh() * (d1 / width).tanh();
        sum_q += q;
        sum_q2 += q * q;
        dot += q * data.vals_centered[i];
    }

    let n_inv = 1.0 / data.n as f32;
    let denom = sum_q2 - sum_q * sum_q * n_inv;
    if denom <= 1e-9 {
        return None;
    }

    let amp = dot / denom;
    // Sufficient-stat residual: at the OLS optimum,
    //   SSR = val_energy - dot²/denom.
    // Replaces the previous full second pass through the disk.
    let ssr = (data.val_energy - dot * dot / denom).max(0.0);

    let rms = (ssr * n_inv).sqrt();
    let corr = dot.abs() / (denom * data.val_energy).sqrt();
    let rel_rms = rms / amp.abs().max(1.0);
    let edge = if edge_score >= 0.0 {
        edge_score
    } else {
        edge_pair_score(data, theta0, theta1)
    };
    let objective = rel_rms - 1.25 * edge;
    let (theta1_c, theta2_c, amp_c) = ring_fit::canonicalize(theta0, theta1, amp);

    Some(Fit {
        theta1: theta1_c,
        theta2: theta2_c,
        width,
        amp: amp_c,
        rms,
        rel_rms,
        corr,
        edge_score: edge,
        objective,
    })
}

/// Per-pair scorer that consumes precomputed tanh slices `h0`, `h1`
/// (one row per candidate, containing `tanh(d_p(θ, w)/w)` for each of
/// `data.n` disk pixels). The body matches `score_pair` exactly except
/// the `tanh` calls have been replaced by array reads.
pub(super) fn score_pair_cached(
    data: &DiskData,
    theta0: f32,
    theta1: f32,
    width: f32,
    edge_score: f32,
    h0: &[f32],
    h1: &[f32],
) -> Option<Fit> {
    if !valid_pair(theta0, theta1) {
        return None;
    }
    if data.val_energy <= 1e-9 {
        return None;
    }

    let mut sum_q = 0.0f32;
    let mut sum_q2 = 0.0f32;
    let mut dot = 0.0f32;
    for p in 0..data.n {
        let q = h0[p] * h1[p];
        sum_q += q;
        sum_q2 += q * q;
        dot += q * data.vals_centered[p];
    }
    let n_inv = 1.0 / data.n as f32;
    let denom = sum_q2 - sum_q * sum_q * n_inv;
    if denom <= 1e-9 {
        return None;
    }
    let amp = dot / denom;
    // Sufficient-stat residual: at the OLS optimum,
    //   SSR = Σ (y - ȳ)² - amp² · Σ (q - q̄)²
    //       = val_energy - dot² / denom.
    // FP rounding can push this slightly negative; clamp at zero.
    let ssr = (data.val_energy - dot * dot / denom).max(0.0);
    let rms = (ssr * n_inv).sqrt();
    let corr = dot.abs() / (denom * data.val_energy).sqrt();
    let rel_rms = rms / amp.abs().max(1.0);
    let objective = rel_rms - 1.25 * edge_score;
    let (theta1_c, theta2_c, amp_c) = ring_fit::canonicalize(theta0, theta1, amp);
    Some(Fit {
        theta1: theta1_c,
        theta2: theta2_c,
        width,
        amp: amp_c,
        rms,
        rel_rms,
        corr,
        edge_score,
        objective,
    })
}

/// Per-pair gradient-alignment score. Expensive (Gaussian kernel over
/// all disk pixels × all gradient directions); use the precomputed path
/// where possible.
pub(super) fn edge_pair_score(data: &DiskData, theta0: f32, theta1: f32) -> f32 {
    let mut total = 0.0f32;
    for i in 0..data.n {
        total += data.grad_weights[i];
    }
    if total <= 1e-9 {
        return 0.0;
    }

    let sigma = 4.0_f32.to_radians();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    for i in 0..data.n {
        let w = data.grad_weights[i];
        let d0 = signed_line_delta(data.grad_angles[i], theta0);
        let d1 = signed_line_delta(data.grad_angles[i], theta1);
        s0 += w * (-0.5 * (d0 / sigma) * (d0 / sigma)).exp();
        s1 += w * (-0.5 * (d1 / sigma) * (d1 / sigma)).exp();
    }

    s0 /= total;
    s1 /= total;
    let balance = 2.0 * s0.min(s1) / (s0 + s1).max(1e-9);
    (s0 + s1) * balance
}

/// Pre-fill `out[c] = Σ_p w_p · exp(-Δ²/2σ²)` (un-normalised) for each
/// candidate angle `θ_c`, returning `total = Σ_p w_p`. The pair score
/// is then `combine_edge_pair(out[i], out[j], total)`.
pub(super) fn precompute_edge_alignments(
    data: &DiskData,
    candidates: &super::candidates::CandidateSet,
    out: &mut [f32; super::candidates::MAX_CANDIDATES],
) -> f32 {
    let mut total = 0.0f32;
    for p in 0..data.n {
        total += data.grad_weights[p];
    }
    if total <= 1e-9 {
        return 0.0;
    }
    let sigma = 4.0_f32.to_radians();
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    for (slot, &theta_c) in out
        .iter_mut()
        .zip(candidates.angles.iter())
        .take(candidates.n)
    {
        let mut s = 0.0f32;
        for p in 0..data.n {
            let w_p = data.grad_weights[p];
            let d = signed_line_delta(data.grad_angles[p], theta_c);
            s += w_p * (-d * d * inv_2sigma2).exp();
        }
        *slot = s;
    }
    total
}

/// Deterministic comparator: better fit has lower `objective`, then
/// lower `rel_rms`, then higher `edge_score`, then narrower `width`,
/// then smaller `theta1`, then smaller `theta2`.
pub(super) fn fit_is_better(a: Fit, b: Fit) -> bool {
    if a.objective < b.objective - 1e-9 {
        return true;
    }
    if (a.objective - b.objective).abs() > 1e-9 {
        return false;
    }
    if a.rel_rms < b.rel_rms - 1e-9 {
        return true;
    }
    if (a.rel_rms - b.rel_rms).abs() > 1e-9 {
        return false;
    }
    if a.edge_score > b.edge_score + 1e-9 {
        return true;
    }
    if (a.edge_score - b.edge_score).abs() > 1e-9 {
        return false;
    }
    if a.width < b.width - 1e-9 {
        return true;
    }
    if (a.width - b.width).abs() > 1e-9 {
        return false;
    }
    if a.theta1 < b.theta1 - 1e-9 {
        return true;
    }
    if (a.theta1 - b.theta1).abs() > 1e-9 {
        return false;
    }
    a.theta2 < b.theta2
}

/// Map a `Fit`'s residual quality to a per-axis sigma estimate
/// (radians).
pub(super) fn sigma_for_fit(fit: &Fit) -> f32 {
    let sep_deg = line_separation(fit.theta1, fit.theta2).to_degrees();
    let floor_deg = if sep_deg >= 55.0 { 1.5 } else { 3.0 };
    let model_deg = floor_deg + (8.0 * fit.rel_rms).min(6.0);
    (model_deg * 0.55).to_radians()
}
