//! Saddle-point quadratic-surface refiner.
//!
//! Fits a 2nd-order surface `I(x, y) = a x² + b x y + c y² + d x + e y + f`
//! to the image patch around the seed and locates the unique
//! stationary point of the resulting quadratic. The Hessian
//! `[2a b; b 2c]` must have negative determinant (a saddle) for the
//! corner to be accepted.

use super::{CornerRefiner, RefineContext, RefineResult, RefineStatus};
use serde::{Deserialize, Serialize};

/// Configuration for the [`SaddlePointRefiner`].
///
/// All thresholds below are advanced tuning knobs. The defaults are
/// appropriate for most scenes; adjust only if you observe excessive
/// rejection or acceptance of clearly-wrong refinements.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct SaddlePointConfig {
    /// Half-size of the image patch used for the surface fit, in image
    /// pixels (integer `>= 1`; the patch is `(2·radius+1)²`). Default `2`
    /// (5×5 patch).
    ///
    /// Advanced tuning.
    #[cfg_attr(feature = "schemars", schemars(range(min = 1)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub radius: i32,
    /// Margin on the Hessian determinant of the fitted quadratic
    /// (unitless, `>= 0`): a saddle is confirmed only when
    /// `det(H) < -det_margin`. Increase to require a sharper saddle.
    ///
    /// Advanced tuning.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    pub det_margin: f32,
    /// Maximum displacement from the seed to the fitted stationary point,
    /// in pixels (`>= 0`); refinements that move farther are rejected.
    /// The effective bound is also capped at `radius + 0.5`.
    ///
    /// Advanced tuning.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub max_offset: f32,
    /// Minimum absolute value of `det(H)` (unitless, `>= 0`). Rejects
    /// near-flat surfaces whose determinant is too close to zero to be
    /// meaningful.
    ///
    /// Advanced tuning.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    pub min_abs_det: f32,
}

impl Default for SaddlePointConfig {
    fn default() -> Self {
        Self {
            radius: 2,
            det_margin: 1e-3,
            max_offset: 1.5,
            min_abs_det: 1e-4,
        }
    }
}

/// Saddle-point quadratic-surface subpixel refiner.
///
/// Fits a 2nd-order surface to the image patch and locates the saddle
/// (the unique stationary point where the Hessian is indefinite). The
/// refiner requires the image intensity patch (passed via
/// [`RefineContext::image`]); it ignores the response map.
///
/// Reuses a fixed-size `6×6` scratch matrix across calls so there is no
/// per-corner allocation.
#[derive(Debug)]
pub struct SaddlePointRefiner {
    cfg: SaddlePointConfig,
    m: [f32; 36],
    rhs: [f32; 6],
}

impl SaddlePointRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: SaddlePointConfig) -> Self {
        Self {
            cfg,
            m: [0.0; 36],
            rhs: [0.0; 6],
        }
    }

    fn solve_6x6(&mut self) -> Option<[f32; 6]> {
        // Simple Gauss-Jordan elimination with partial pivoting on the stack.
        for i in 0..6 {
            let mut pivot = i;
            let mut pivot_val = self.m[i * 6 + i].abs();
            for r in (i + 1)..6 {
                let v = self.m[r * 6 + i].abs();
                if v > pivot_val {
                    pivot = r;
                    pivot_val = v;
                }
            }

            if pivot_val < 1e-9 {
                return None;
            }

            if pivot != i {
                for c in i..6 {
                    self.m.swap(i * 6 + c, pivot * 6 + c);
                }
                self.rhs.swap(i, pivot);
            }

            let diag = self.m[i * 6 + i];
            let inv_diag = 1.0 / diag;

            for c in i..6 {
                self.m[i * 6 + c] *= inv_diag;
            }
            self.rhs[i] *= inv_diag;

            for r in 0..6 {
                if r == i {
                    continue;
                }
                let factor = self.m[r * 6 + i];
                if factor == 0.0 {
                    continue;
                }
                for c in i..6 {
                    self.m[r * 6 + c] -= factor * self.m[i * 6 + c];
                }
                self.rhs[r] -= factor * self.rhs[i];
            }
        }

        let mut out = [0.0f32; 6];
        out.copy_from_slice(&self.rhs);
        Some(out)
    }
}

impl CornerRefiner for SaddlePointRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        self.cfg.radius
    }

    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        let img = match ctx.image {
            Some(view) => view,
            None => {
                return RefineResult {
                    x: seed_xy[0],
                    y: seed_xy[1],
                    score: 0.0,
                    status: RefineStatus::Rejected,
                    covariance_per_noise_variance: None,
                }
            }
        };

        let cx = seed_xy[0].round() as i32;
        let cy = seed_xy[1].round() as i32;
        let r = self.cfg.radius;

        if !img.supports_patch(cx, cy, r) {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: 0.0,
                status: RefineStatus::OutOfBounds,
                covariance_per_noise_variance: None,
            };
        }

        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for dy in -r..=r {
            let gy = cy + dy;
            for dx in -r..=r {
                let gx = cx + dx;
                sum += img.sample(gx, gy);
                count += 1.0;
            }
        }

        let mean = if count > 0.0 { sum / count } else { 0.0 };

        self.m.fill(0.0);
        self.rhs.fill(0.0);

        for dy in -r..=r {
            let gy = cy + dy;
            for dx in -r..=r {
                let gx = cx + dx;
                let i = img.sample(gx, gy) - mean;

                let x = gx as f32 - seed_xy[0];
                let y = gy as f32 - seed_xy[1];
                let phi = [x * x, x * y, y * y, x, y, 1.0];

                for row in 0..6 {
                    self.rhs[row] += phi[row] * i;
                    for col in row..6 {
                        self.m[row * 6 + col] += phi[row] * phi[col];
                    }
                }
            }
        }

        // Fill the lower triangle to make elimination logic simpler.
        for row in 0..6 {
            for col in 0..row {
                self.m[row * 6 + col] = self.m[col * 6 + row];
            }
        }

        // The solve eliminates in place; keep the normal matrix for the
        // position covariance.
        let normal = self.m;
        let coeffs = match self.solve_6x6() {
            Some(c) => c,
            None => {
                return RefineResult {
                    x: seed_xy[0],
                    y: seed_xy[1],
                    score: 0.0,
                    status: RefineStatus::IllConditioned,
                    covariance_per_noise_variance: None,
                }
            }
        };

        let a = coeffs[0];
        let b = coeffs[1];
        let c = coeffs[2];
        let d = coeffs[3];
        let e = coeffs[4];

        let det_h = 4.0 * a * c - b * b;
        if det_h > -self.cfg.det_margin || det_h.abs() < self.cfg.min_abs_det {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det_h,
                status: RefineStatus::IllConditioned,
                covariance_per_noise_variance: None,
            };
        }

        let inv_det = 1.0 / det_h;
        let ux = -(2.0 * c * d - b * e) * inv_det;
        let uy = (b * d - 2.0 * a * e) * inv_det;

        let max_off = self.cfg.max_offset.min(r as f32 + 0.5);
        if ux.abs() > max_off || uy.abs() > max_off {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det_h,
                status: RefineStatus::Rejected,
                covariance_per_noise_variance: None,
            };
        }

        let score = (-det_h).sqrt();
        let covariance_per_noise_variance = position_covariance(&normal, &coeffs, [ux, uy]);
        RefineResult {
            covariance_per_noise_variance,
            ..RefineResult::accepted([seed_xy[0] + ux, seed_xy[1] + uy], score)
        }
    }
}

/// First-order covariance `[σ²ₓ, σₓᵧ, σ²ᵧ]` of the saddle location per unit
/// image-noise variance (px² per gray-level²).
///
/// With white pixel noise of variance `σₙ²`, the surface coefficients
/// `θ = (a, b, c, d, e, f)` have covariance `σₙ²·(ΦᵀΦ)⁻¹`. The location
/// `u = −H⁻¹·(d, e)`, `H = [2a b; b 2c]`, has Jacobian
/// `J = −H⁻¹·[[2uₓ, u_y, 0, 1, 0, 0], [0, uₓ, 2u_y, 0, 1, 0]]`, so
/// `Σᵤ = σₙ²·J(ΦᵀΦ)⁻¹Jᵀ`; this returns `J(ΦᵀΦ)⁻¹Jᵀ`. The fit's own residual
/// is not used for the scale: on a sharp corner it measures how far the
/// patch is from a quadratic, which biases the location but does not
/// scatter it. `None` if the normal matrix is singular.
fn position_covariance(normal: &[f32; 36], coeffs: &[f32; 6], u: [f32; 2]) -> Option<[f32; 3]> {
    let theta = coeffs.map(f64::from);
    let (a, b, c) = (theta[0], theta[1], theta[2]);
    let det = 4.0 * a * c - b * b;
    // H⁻¹ for H = [2a b; b 2c].
    let h_inv = [[2.0 * c / det, -b / det], [-b / det, 2.0 * a / det]];
    let (ux, uy) = (f64::from(u[0]), f64::from(u[1]));
    let rows = [
        [2.0 * ux, uy, 0.0, 1.0, 0.0, 0.0],
        [0.0, ux, 2.0 * uy, 0.0, 1.0, 0.0],
    ];
    // J = −H⁻¹·rows.
    let mut jac = [[0.0f64; 6]; 2];
    for (i, jrow) in jac.iter_mut().enumerate() {
        for (k, jk) in jrow.iter_mut().enumerate() {
            *jk = -(h_inv[i][0] * rows[0][k] + h_inv[i][1] * rows[1][k]);
        }
    }

    // X = (ΦᵀΦ)⁻¹·Jᵀ by Gaussian elimination with partial pivoting.
    let mut m = [[0.0f64; 8]; 6];
    for (row, mrow) in m.iter_mut().enumerate() {
        for col in 0..6 {
            mrow[col] = f64::from(normal[row * 6 + col]);
        }
        mrow[6] = jac[0][row];
        mrow[7] = jac[1][row];
    }
    for col in 0..6 {
        let pivot = (col..6).max_by(|&p, &q| m[p][col].abs().total_cmp(&m[q][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        for row in (col + 1)..6 {
            let factor = m[row][col] / m[col][col];
            for k in col..8 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    let mut x = [[0.0f64; 2]; 6];
    for row in (0..6).rev() {
        for rhs in 0..2 {
            let tail: f64 = ((row + 1)..6).map(|k| m[row][k] * x[k][rhs]).sum();
            x[row][rhs] = (m[row][6 + rhs] - tail) / m[row][row];
        }
    }
    let cov = |i: usize, j: usize| -> f64 { (0..6).map(|k| jac[i][k] * x[k][j]).sum::<f64>() };
    Some([cov(0, 0) as f32, cov(0, 1) as f32, cov(1, 1) as f32])
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_checkerboard;
    use super::*;
    use crate::imageview::ImageView;

    /// Predicted (`σₙ²·covariance_per_noise_variance`) against empirical
    /// position scatter over noise realizations, on anti-aliased corners
    /// with and without blur. A measurement, not a gate: `cargo test -p chess-corners-core --release
    /// covariance_matches_scatter -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn covariance_matches_scatter() {
        use chess_corners_testutil::{aa_chessboard, add_gaussian_noise, gaussian_blur};
        let size = 21;
        println!("noise  blur  offset       σx_emp  σx_pred  σy_emp  σy_pred  ratio_x  ratio_y");
        for &noise in &[1.0f32, 2.0, 4.0, 8.0] {
            for &blur in &[0.0f32, 0.8, 1.5] {
                for &(ox, oy) in &[(10.0f32, 10.0f32), (10.3, 10.6), (10.45, 9.8)] {
                    let mut clean = aa_chessboard(size, 10, (ox, oy), 40, 200);
                    if blur > 0.0 {
                        gaussian_blur(&mut clean, size, blur);
                    }
                    // Gaussian noise plus the 8-bit rounding (variance 1/12).
                    let noise_var = f64::from(noise * noise) + 1.0 / 12.0;
                    let mut xs = Vec::new();
                    let mut preds = Vec::new();
                    for seed in 0..400u64 {
                        let mut img = clean.clone();
                        add_gaussian_noise(&mut img, noise, seed + 1);
                        let view = ImageView::from_u8_slice(size, size, &img).unwrap();
                        let mut refiner = SaddlePointRefiner::new(SaddlePointConfig::default());
                        let res = refiner.refine(
                            [ox.round(), oy.round()],
                            RefineContext {
                                image: Some(view),
                                response: None,
                            },
                        );
                        if res.status != RefineStatus::Accepted {
                            continue;
                        }
                        xs.push([f64::from(res.x), f64::from(res.y)]);
                        preds.push(res.covariance_per_noise_variance.unwrap());
                    }
                    let n = xs.len() as f64;
                    let mean = [
                        xs.iter().map(|p| p[0]).sum::<f64>() / n,
                        xs.iter().map(|p| p[1]).sum::<f64>() / n,
                    ];
                    let var = |k: usize| {
                        xs.iter().map(|p| (p[k] - mean[k]).powi(2)).sum::<f64>() / (n - 1.0)
                    };
                    let pred = |k: usize| {
                        noise_var * preds.iter().map(|c| f64::from(c[k])).sum::<f64>() / n
                    };
                    let (ex, ey) = (var(0).sqrt(), var(1).sqrt());
                    let (px, py) = (pred(0).sqrt(), pred(2).sqrt());
                    println!(
                        "{noise:5.1}  {blur:4.1}  ({ox:5.2},{oy:5.2})  {ex:7.4}  {px:7.4}  {ey:7.4}  {py:7.4}  {:7.2}  {:7.2}   n={}",
                        ex / px, ey / py, xs.len()
                    );
                }
            }
        }
    }

    #[test]
    fn saddle_point_recovers_stationary_point_and_rejects_flat() {
        let img = synthetic_checkerboard(17, (8.2, 8.6), 30, 230);
        let view = ImageView::from_u8_slice(17, 17, &img).unwrap();
        let ctx = RefineContext {
            image: Some(view),
            response: None,
        };
        let mut refiner = SaddlePointRefiner::new(SaddlePointConfig::default());
        let res = refiner.refine([8.0, 9.0], ctx);
        assert_eq!(res.status, RefineStatus::Accepted);
        let true_xy = [8.2f32, 8.6f32];
        let seed_err = ((8.0 - true_xy[0]).powi(2) + (9.0 - true_xy[1]).powi(2)).sqrt();
        let refined_err = ((res.x - true_xy[0]).powi(2) + (res.y - true_xy[1]).powi(2)).sqrt();
        assert!(
            refined_err <= seed_err * 1.6 && refined_err < 1.0,
            "refined_err {refined_err} seed_err {seed_err} res {:?}",
            (res.x, res.y)
        );

        let flat = vec![128u8; 25];
        let flat_view = ImageView::from_u8_slice(5, 5, &flat).unwrap();
        let flat_ctx = RefineContext {
            image: Some(flat_view),
            response: None,
        };
        let flat_res = refiner.refine([2.0, 2.0], flat_ctx);
        assert_ne!(flat_res.status, RefineStatus::Accepted);
    }
}
