//! Ring-fit two-axis orientation method.
//!
//! Fits the parametric two-axis intensity model
//! `I(φ) = μ + A·tanh(β·sin(φ−θ₁))·tanh(β·sin(φ−θ₂))` to 16 ring
//! samples via Gauss-Newton seeded from the 2nd-harmonic orientation,
//! with deterministic grid-search fallback for extreme-skew traces. Each
//! axis carries the residual-scaled least-squares standard error of the
//! fit ([`axis_std_errors`]).
//!
//! Submodules:
//!
//! - [`solver`] — Gauss-Newton body, 4×4 linear algebra, canonicalization.
//! - [`seed`] — 2nd-harmonic seeding plus the deterministic grid and
//!   cached canonical seed models for the radius-5 ring.
//! - [`robust`] — top-level `fit_ring` entry: suspicion gating and the
//!   grid-search fallback that fires when the primary Gauss-Newton solve
//!   looks wrong.

mod robust;
mod seed;
mod solver;

/// Smoothing slope of the tanh sign approximation used in the
/// intensity model. Fixed constant — not a fit parameter. Reflects
/// the effective ring-integration blur at the sampled radius.
pub(super) const TANH_BETA: f32 = 4.0;

/// Floor on `|amp|` used when forming the relative residual and the
/// suspicion gate. Below this the fit is essentially degenerate.
pub(super) const AMP_FLOOR: f32 = 1.0;

/// Result of the parametric two-axis intensity fit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwoAxisFit {
    pub amp: f32,
    pub theta1: f32,
    pub theta2: f32,
    /// Residual-scaled least-squares standard error of `theta1` (radians).
    pub fit_std_error_theta1: f32,
    /// Residual-scaled least-squares standard error of `theta2` (radians).
    pub fit_std_error_theta2: f32,
    pub rms: f32,
}

pub(crate) use robust::{fit_is_suspicious, fit_ring};
pub(crate) use solver::{axis_std_errors, canonicalize};
