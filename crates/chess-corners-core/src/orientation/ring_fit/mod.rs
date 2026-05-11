//! Ring-fit two-axis orientation method.
//!
//! Fits the parametric two-axis intensity model
//! `I(φ) = μ + A·tanh(β·sin(φ−θ₁))·tanh(β·sin(φ−θ₂))` to 16 ring
//! samples via Gauss-Newton seeded from the 2nd-harmonic orientation,
//! with deterministic grid-search fallback for extreme-skew traces and
//! a calibrated σ-LUT post-processor.
//!
//! Submodules:
//!
//! - [`solver`] — Gauss-Newton body, 4×4 linear algebra, canonicalization.
//! - [`seed`] — 2nd-harmonic seeding plus the deterministic grid and
//!   cached canonical seed models for the radius-5 ring.
//! - [`robust`] — suspicion gating and the grid-search fallback that
//!   fires when the primary Gauss-Newton solve looks wrong.
//! - [`uncertainty`] — top-level `fit_ring` entry that applies the
//!   piecewise-linear σ-correction LUT to the raw CRLB sigmas.

mod robust;
mod seed;
mod solver;
mod uncertainty;

/// Smoothing slope of the tanh sign approximation used in the
/// intensity model. Fixed constant — not a fit parameter. Reflects
/// the effective ring-integration blur at the sampled radius.
pub(super) const TANH_BETA: f32 = 4.0;

/// Floor on `|amp|` used when forming the relative residual and the
/// suspicion gate. Below this the fit is essentially degenerate (σ
/// already π); the relative residual saturates the LUT and the
/// multiplier becomes 1.0.
pub(super) const AMP_FLOOR: f32 = 1.0;

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

pub(crate) use robust::fit_is_suspicious;
pub(crate) use solver::canonicalize;
pub(crate) use uncertainty::fit_ring;
