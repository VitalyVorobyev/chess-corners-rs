//! Suspicious-fit gating and the deterministic grid-search fallback.

#[cfg(feature = "std")]
use super::seed::{canonical_seed_models, is_canonical_ring_phi, score_cached_seed_model};
use super::seed::{
    insert_seed_candidate, line_separation, score_seed_pair, second_harmonic, valid_seed_pair,
    SeedCandidate, ROBUST_GRID_STEPS, ROBUST_GRID_STEP_DEG, ROBUST_TOP_SEEDS,
};
use super::solver::fit_with_seed;
use super::{TwoAxisFit, AMP_FLOOR, TANH_BETA};

/// Trigger the grid-search fallback when the primary fit's relative
/// residual exceeds this fraction of `|amp|` *or* the recovered axis
/// separation falls outside the orthogonality window.
const ROBUST_REL_RMS_TRIGGER: f32 = 0.12;
const ROBUST_SEP_TRIGGER_MIN_RAD: f32 = 65.0_f32.to_radians();
const ROBUST_SEP_TRIGGER_MAX_RAD: f32 = 115.0_f32.to_radians();

/// Fit a two-axis chessboard-corner intensity model to 16 ring
/// samples, with the deterministic grid-search fallback used when the
/// fast 2nd-harmonic seed produces a suspicious fit.
pub(super) fn fit(samples: &[f32; 16], ring_phi: &[f32; 16]) -> TwoAxisFit {
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

    let primary = fit_with_seed(
        samples,
        ring_phi,
        theta1_seed,
        theta2_seed,
        amp_seed,
        mu_seed,
        centered_var,
    );

    if !fit_is_suspicious(&primary) {
        return primary;
    }

    match robust_grid_fit(samples, ring_phi, mean, centered_var) {
        Some(candidate) if candidate.rms < primary.rms - 1e-3 => candidate,
        _ => primary,
    }
}

pub(crate) fn fit_is_suspicious(fit: &TwoAxisFit) -> bool {
    if !fit.amp.is_finite() || !fit.rms.is_finite() || fit.amp < AMP_FLOOR {
        return false;
    }
    let rel_rms = relative_rms(fit);
    let sep = line_separation(fit.theta1, fit.theta2);
    !(ROBUST_SEP_TRIGGER_MIN_RAD..=ROBUST_SEP_TRIGGER_MAX_RAD).contains(&sep)
        || rel_rms >= ROBUST_REL_RMS_TRIGGER
}

pub(super) fn relative_rms(fit: &TwoAxisFit) -> f32 {
    if !fit.rms.is_finite() {
        return f32::INFINITY;
    }
    fit.rms.max(0.0) / fit.amp.max(AMP_FLOOR)
}

fn robust_grid_fit(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    mean: f32,
    centered_var: f32,
) -> Option<TwoAxisFit> {
    if centered_var < 1e-6 {
        return None;
    }

    let mut centered = [0.0f32; 16];
    let mut centered_ss = 0.0f32;
    for i in 0..16 {
        let v = samples[i] - mean;
        centered[i] = v;
        centered_ss += v * v;
    }
    if !centered_ss.is_finite() || centered_ss <= 1e-6 {
        return None;
    }

    #[cfg(feature = "std")]
    if is_canonical_ring_phi(ring_phi) {
        return robust_grid_fit_cached(
            samples,
            ring_phi,
            mean,
            centered_var,
            &centered,
            centered_ss,
        );
    }

    let mut sphi = [0.0f32; 16];
    let mut cphi = [0.0f32; 16];
    for i in 0..16 {
        let (s, c) = ring_phi[i].sin_cos();
        sphi[i] = s;
        cphi[i] = c;
    }

    let mut h = [[0.0f32; 16]; ROBUST_GRID_STEPS];
    let step = ROBUST_GRID_STEP_DEG.to_radians();
    for (ai, row) in h.iter_mut().enumerate() {
        let theta = ai as f32 * step;
        let (s_t, c_t) = theta.sin_cos();
        for i in 0..16 {
            let s = sphi[i] * c_t - cphi[i] * s_t;
            row[i] = (TANH_BETA * s).tanh();
        }
    }

    let mut top: [Option<SeedCandidate>; ROBUST_TOP_SEEDS] = [None; ROBUST_TOP_SEEDS];
    for ai in 0..ROBUST_GRID_STEPS {
        let theta1 = ai as f32 * step;
        for bi in (ai + 1)..ROBUST_GRID_STEPS {
            let theta2 = bi as f32 * step;
            if !valid_seed_pair(theta1, theta2) {
                continue;
            }
            let Some(candidate) =
                score_seed_pair(&centered, centered_ss, mean, &h[ai], &h[bi], theta1, theta2)
            else {
                continue;
            };
            insert_seed_candidate(&mut top, candidate);
        }
    }

    refine_seed_candidates(samples, ring_phi, centered_var, top)
}

#[cfg(feature = "std")]
fn robust_grid_fit_cached(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    mean: f32,
    centered_var: f32,
    centered: &[f32; 16],
    centered_ss: f32,
) -> Option<TwoAxisFit> {
    let mut top: [Option<SeedCandidate>; ROBUST_TOP_SEEDS] = [None; ROBUST_TOP_SEEDS];
    for model in canonical_seed_models() {
        let Some(candidate) = score_cached_seed_model(centered, centered_ss, mean, model) else {
            continue;
        };
        insert_seed_candidate(&mut top, candidate);
    }

    refine_seed_candidates(samples, ring_phi, centered_var, top)
}

fn refine_seed_candidates(
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
    centered_var: f32,
    top: [Option<SeedCandidate>; ROBUST_TOP_SEEDS],
) -> Option<TwoAxisFit> {
    let mut best: Option<TwoAxisFit> = None;
    for seed in top.into_iter().flatten() {
        let fit = fit_with_seed(
            samples,
            ring_phi,
            seed.theta1,
            seed.theta2,
            seed.amp,
            seed.mu,
            centered_var,
        );
        if best.is_none_or(|b| fit.rms < b.rms - 1e-6) {
            best = Some(fit);
        }
    }
    best
}
