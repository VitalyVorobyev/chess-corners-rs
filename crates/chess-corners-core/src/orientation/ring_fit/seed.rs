//! Seed generation for the ring fit: 2nd-harmonic estimate, the
//! deterministic seed grid, and the cached canonical seed models for
//! the radius-5 ring.

use super::TANH_BETA;
use core::f32::consts::PI;
#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(feature = "std")]
use crate::detect::chess::ring::RING5;

/// Global seed grid step size used only when the fast 2nd-harmonic
/// seed looks suspicious. 8° is coarse enough to keep the fallback
/// cheap and close enough for the following Gauss-Newton solve to
/// converge into the correct basin on the extreme-skew ring traces.
pub(super) const ROBUST_GRID_STEP_DEG: f32 = 8.0;
pub(super) const ROBUST_GRID_STEPS: usize = 23;
pub(super) const ROBUST_TOP_SEEDS: usize = 1;
pub(super) const ROBUST_MIN_SEP_RAD: f32 = 12.0_f32.to_radians();
pub(super) const ROBUST_MAX_SEP_RAD: f32 = 89.5_f32.to_radians();

#[derive(Clone, Copy, Debug)]
pub(super) struct SeedCandidate {
    pub theta1: f32,
    pub theta2: f32,
    pub mu: f32,
    pub amp: f32,
    pub rms: f32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SeedModel {
    pub theta1: f32,
    pub theta2: f32,
    pub q_mean: f32,
    pub q_centered: [f32; 16],
    pub denom: f32,
}

#[cfg(feature = "std")]
static CANONICAL_SEED_MODELS: OnceLock<Vec<SeedModel>> = OnceLock::new();

/// 2nd-harmonic estimate: returns `(theta_seed, magnitude)` where
/// `theta_seed` is a sector-midpoint direction (mod π) and magnitude
/// is a non-negative estimate of the A parameter amplitude.
pub(super) fn second_harmonic(samples: &[f32; 16], ring_phi: &[f32; 16], mean: f32) -> (f32, f32) {
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

pub(super) fn valid_seed_pair(theta1: f32, theta2: f32) -> bool {
    let sep = line_separation(theta1, theta2);
    (ROBUST_MIN_SEP_RAD..=ROBUST_MAX_SEP_RAD).contains(&sep)
}

pub(super) fn line_separation(theta1: f32, theta2: f32) -> f32 {
    let d = (theta1 - theta2 + PI * 0.5).rem_euclid(PI) - PI * 0.5;
    d.abs()
}

pub(super) fn score_seed_pair(
    centered: &[f32; 16],
    centered_ss: f32,
    mean: f32,
    h1: &[f32; 16],
    h2: &[f32; 16],
    theta1: f32,
    theta2: f32,
) -> Option<SeedCandidate> {
    let mut q_sum = 0.0f32;
    let mut q2_sum = 0.0f32;
    let mut num = 0.0f32;
    for i in 0..16 {
        let q = h1[i] * h2[i];
        q_sum += q;
        q2_sum += q * q;
        num += centered[i] * q;
    }
    let q_mean = q_sum / 16.0;
    let denom = q2_sum - q_sum * q_mean;
    if !denom.is_finite() || denom <= 1e-6 {
        return None;
    }

    let amp = num / denom;
    let mu = mean - amp * q_mean;
    let ssr = (centered_ss - num * num / denom).max(0.0);
    let rms = (ssr / 16.0).sqrt();
    if !(rms.is_finite() && amp.is_finite() && mu.is_finite()) {
        return None;
    }

    Some(SeedCandidate {
        theta1,
        theta2,
        mu,
        amp,
        rms,
    })
}

pub(super) fn insert_seed_candidate(
    top: &mut [Option<SeedCandidate>; ROBUST_TOP_SEEDS],
    candidate: SeedCandidate,
) {
    let mut pos = None;
    for (i, existing) in top.iter().enumerate() {
        if existing.is_none_or(|seed| candidate.rms < seed.rms) {
            pos = Some(i);
            break;
        }
    }
    let Some(pos) = pos else {
        return;
    };
    for i in (pos + 1..ROBUST_TOP_SEEDS).rev() {
        top[i] = top[i - 1];
    }
    top[pos] = Some(candidate);
}

#[cfg(feature = "std")]
pub(super) fn score_cached_seed_model(
    centered: &[f32; 16],
    centered_ss: f32,
    mean: f32,
    model: &SeedModel,
) -> Option<SeedCandidate> {
    let mut num = 0.0f32;
    for (&v, &q) in centered.iter().zip(model.q_centered.iter()) {
        num += v * q;
    }
    let amp = num / model.denom;
    let mu = mean - amp * model.q_mean;
    let ssr = (centered_ss - num * num / model.denom).max(0.0);
    let rms = (ssr / 16.0).sqrt();
    if !(rms.is_finite() && amp.is_finite() && mu.is_finite()) {
        return None;
    }

    Some(SeedCandidate {
        theta1: model.theta1,
        theta2: model.theta2,
        mu,
        amp,
        rms,
    })
}

#[cfg(feature = "std")]
pub(super) fn canonical_seed_models() -> &'static [SeedModel] {
    CANONICAL_SEED_MODELS.get_or_init(build_canonical_seed_models)
}

#[cfg(feature = "std")]
fn build_canonical_seed_models() -> Vec<SeedModel> {
    let phi = canonical_ring_phi();
    let mut sphi = [0.0f32; 16];
    let mut cphi = [0.0f32; 16];
    for i in 0..16 {
        let (s, c) = phi[i].sin_cos();
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

    let mut models = Vec::with_capacity(ROBUST_GRID_STEPS * ROBUST_GRID_STEPS / 2);
    for ai in 0..ROBUST_GRID_STEPS {
        let theta1 = ai as f32 * step;
        for bi in (ai + 1)..ROBUST_GRID_STEPS {
            let theta2 = bi as f32 * step;
            if !valid_seed_pair(theta1, theta2) {
                continue;
            }

            let mut q_sum = 0.0f32;
            let mut q2_sum = 0.0f32;
            let mut q_centered = [0.0f32; 16];
            for (i, q_out) in q_centered.iter_mut().enumerate() {
                let q = h[ai][i] * h[bi][i];
                q_sum += q;
                q2_sum += q * q;
                *q_out = q;
            }
            let q_mean = q_sum / 16.0;
            let denom = q2_sum - q_sum * q_mean;
            if !denom.is_finite() || denom <= 1e-6 {
                continue;
            }
            for q in &mut q_centered {
                *q -= q_mean;
            }

            models.push(SeedModel {
                theta1,
                theta2,
                q_mean,
                q_centered,
                denom,
            });
        }
    }
    models
}

#[cfg(feature = "std")]
pub(super) fn is_canonical_ring_phi(ring_phi: &[f32; 16]) -> bool {
    let expected = canonical_ring_phi();
    ring_phi
        .iter()
        .zip(expected.iter())
        .all(|(&got, &want)| (got - want).abs() <= 1e-6)
}

#[cfg(feature = "std")]
fn canonical_ring_phi() -> [f32; 16] {
    let mut phi = [0.0f32; 16];
    for (i, &(dx, dy)) in RING5.iter().enumerate() {
        phi[i] = (dy as f32).atan2(dx as f32);
    }
    phi
}
