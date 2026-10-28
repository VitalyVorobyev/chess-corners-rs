//! Angle metrics for comparing undirected corner axes.

use core::f32::consts::PI;

/// Undirected angular distance between two line angles (radians), folded
/// into `[0, π/2]`. Lines are equivalent mod π, so this is the metric for
/// "do these two `axes[*].angle` values point along the same line".
pub fn line_delta(a: f32, b: f32) -> f32 {
    ((a - b + PI * 0.5).rem_euclid(PI) - PI * 0.5).abs()
}
