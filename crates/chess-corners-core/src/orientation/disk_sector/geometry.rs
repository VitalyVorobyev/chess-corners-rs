//! Angular geometry helpers for undirected line directions.

use core::f32::consts::PI;

/// Wrap an angle into `[0, π)`.
#[inline]
pub(super) fn wrap_pi(a: f32) -> f32 {
    a.rem_euclid(PI)
}

/// Signed angular distance from `b` to `a` in `(-π/2, π/2]`.
#[inline]
pub(super) fn signed_line_delta(a: f32, b: f32) -> f32 {
    (a - b + PI * 0.5).rem_euclid(PI) - PI * 0.5
}

/// Undirected angular distance between two line angles in `[0, π/2]`.
#[inline]
pub(super) fn line_delta(a: f32, b: f32) -> f32 {
    signed_line_delta(a, b).abs()
}

/// Angular separation between a pair of line directions, i.e.
/// `min(|a0 − a1|, π − |a0 − a1|)` mapped to `[0, π/2]`.
#[inline]
pub(super) fn line_separation(a0: f32, a1: f32) -> f32 {
    line_delta(a0, a1)
}

/// Max undirected axis error between two axis pairs, tested both
/// directly and with the axes swapped.
#[inline]
pub(super) fn pair_disagreement(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    let direct = line_delta(a0, b0).max(line_delta(a1, b1));
    let swapped = line_delta(a0, b1).max(line_delta(a1, b0));
    direct.min(swapped)
}

/// Return `true` when `(a0, a1)` is a valid candidate pair: separation
/// must be in `(12°, 89.5°]`.
#[inline]
pub(super) fn valid_pair(a0: f32, a1: f32) -> bool {
    let sep = line_separation(a0, a1);
    (12.0_f32.to_radians()..=89.5_f32.to_radians()).contains(&sep)
}
