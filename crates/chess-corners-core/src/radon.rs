//! Shared primitives for the Duda-Frese (2018) localized Radon
//! response — used by both [`RadonPeakRefiner`](crate::refine_radon)
//! (per-candidate subpixel refiner) and the forthcoming whole-image
//! `radon_detector` path.
//!
//! The module exists so the angular basis, the peak-fit, and the
//! response-map box blur live in exactly one place. When the detector
//! and refiner disagree on those primitives, they stop being comparable.

use serde::{Deserialize, Serialize};

/// Number of discrete ray angles. The paper samples
/// `{0, π/4, π/2, 3π/4}`.
pub const ANGLES: usize = 4;

/// `cos α` for the four ray angles, in order.
pub const DIR_COS: [f32; ANGLES] = [
    1.0,
    core::f32::consts::FRAC_1_SQRT_2,
    0.0,
    -core::f32::consts::FRAC_1_SQRT_2,
];

/// `sin α` for the four ray angles, in order.
pub const DIR_SIN: [f32; ANGLES] = [
    0.0,
    core::f32::consts::FRAC_1_SQRT_2,
    1.0,
    core::f32::consts::FRAC_1_SQRT_2,
];

/// Subpixel peak-fitting mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PeakFitMode {
    /// Classic parabolic fit on the raw response values.
    Parabolic,
    /// Parabolic fit on `log(response)` — equivalent to fitting a
    /// Gaussian through three samples. Paper default; recommended.
    #[default]
    Gaussian,
}

/// Fit the peak of three samples along one axis. Returns a fractional
/// offset in `[-0.5, 0.5]` grid-steps from the middle sample.
///
/// The parabolic mode is `y = a + b·x + c·x²`; the Gaussian mode is the
/// same fit applied to `log(y)`, provided all three samples are
/// strictly positive. Negative or zero samples trigger a parabolic
/// fallback. A denominator near zero (flat or rising slope at the
/// "peak") returns 0.0 rather than diverging.
#[inline]
pub fn fit_peak_frac(y_minus: f32, y_c: f32, y_plus: f32, mode: PeakFitMode) -> f32 {
    let (ym, y0, yp) = match mode {
        PeakFitMode::Gaussian if y_minus > 0.0 && y_c > 0.0 && y_plus > 0.0 => {
            (y_minus.ln(), y_c.ln(), y_plus.ln())
        }
        _ => (y_minus, y_c, y_plus),
    };
    let denom = ym - 2.0 * y0 + yp;
    // A true maximum has denom < 0. If denom ≥ 0 the neighbours aren't
    // strictly below the centre; fall back to "no subpixel shift"
    // rather than producing a divergent extrapolation.
    if denom > -1e-12 {
        return 0.0;
    }
    let frac = 0.5 * (ym - yp) / denom;
    frac.clamp(-0.5, 0.5)
}

/// Separable `(2·radius+1)²` box blur applied in place to a flat
/// row-major `side × side` response grid. `scratch` must match `resp`
/// in length and is used as temporary storage. `radius = 0` is a no-op.
pub fn box_blur_inplace(resp: &mut [f32], scratch: &mut [f32], side: usize, radius: usize) {
    debug_assert_eq!(resp.len(), side * side);
    debug_assert_eq!(scratch.len(), side * side);
    if radius == 0 {
        return;
    }
    // Horizontal pass: resp -> scratch.
    for y in 0..side {
        let row_start = y * side;
        for x in 0..side {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(side);
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for xx in x0..x1 {
                acc += resp[row_start + xx];
                n += 1.0;
            }
            scratch[row_start + x] = acc / n;
        }
    }
    // Vertical pass: scratch -> resp.
    for x in 0..side {
        for y in 0..side {
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(side);
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for yy in y0..y1 {
                acc += scratch[yy * side + x];
                n += 1.0;
            }
            resp[y * side + x] = acc / n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_peak_frac_symmetric_parabola_is_zero() {
        // y = -x² + 1 sampled at ±1 and 0 → peak at 0.
        let f = fit_peak_frac(0.0, 1.0, 0.0, PeakFitMode::Parabolic);
        assert!(f.abs() < 1e-6, "expected 0.0, got {f}");
    }

    #[test]
    fn fit_peak_frac_shifted_parabola_recovers_offset() {
        // y = -(x - 0.25)² + 1 sampled at x = -1, 0, 1.
        let y_of = |x: f32| -(x - 0.25).powi(2) + 1.0;
        let f = fit_peak_frac(y_of(-1.0), y_of(0.0), y_of(1.0), PeakFitMode::Parabolic);
        assert!((f - 0.25).abs() < 1e-5, "expected 0.25, got {f}");
    }

    #[test]
    fn fit_peak_frac_gaussian_mode_handles_log() {
        // Pure Gaussian: y = exp(-(x-0.2)²/(2·0.5²)).
        let g = |x: f32| (-((x - 0.2f32).powi(2)) / 0.5).exp();
        let f = fit_peak_frac(g(-1.0), g(0.0), g(1.0), PeakFitMode::Gaussian);
        assert!((f - 0.2).abs() < 1e-5, "Gaussian-log fit off: {f}");
    }

    #[test]
    fn fit_peak_frac_rejects_non_maximum() {
        // Non-maximum (denominator ≥ 0) → 0.0 fallback.
        let f = fit_peak_frac(1.0, 0.5, 1.0, PeakFitMode::Parabolic);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn fit_peak_frac_gaussian_falls_back_on_nonpositive() {
        // Any ≤0 sample → parabolic branch.
        let parab = fit_peak_frac(-0.5, 2.0, 1.5, PeakFitMode::Parabolic);
        let gauss = fit_peak_frac(-0.5, 2.0, 1.5, PeakFitMode::Gaussian);
        assert_eq!(parab, gauss);
    }

    #[test]
    fn box_blur_zero_radius_is_identity() {
        let side = 5usize;
        let mut resp: Vec<f32> = (0..(side * side)).map(|i| i as f32).collect();
        let before = resp.clone();
        let mut scratch = vec![0.0; side * side];
        box_blur_inplace(&mut resp, &mut scratch, side, 0);
        assert_eq!(resp, before);
    }

    #[test]
    fn box_blur_smooths_impulse() {
        let side = 5usize;
        let mut resp = vec![0.0f32; side * side];
        let mut scratch = vec![0.0f32; side * side];
        let mid = side / 2;
        resp[mid * side + mid] = 9.0;
        box_blur_inplace(&mut resp, &mut scratch, side, 1);
        // 3×3 blur of an impulse = 9/9 = 1 at the center.
        assert!(
            (resp[mid * side + mid] - 1.0).abs() < 1e-6,
            "center after blur = {}",
            resp[mid * side + mid]
        );
    }

    #[test]
    fn box_blur_preserves_constant_field() {
        let side = 7usize;
        let mut resp = vec![3.5f32; side * side];
        let mut scratch = vec![0.0f32; side * side];
        box_blur_inplace(&mut resp, &mut scratch, side, 1);
        for v in &resp {
            assert!((v - 3.5).abs() < 1e-6);
        }
    }
}
