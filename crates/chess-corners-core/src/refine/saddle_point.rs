//! Saddle-point quadratic-surface refiner.
//!
//! Fits a 2nd-order surface `I(x, y) = a x² + b x y + c y² + d x + e y + f`
//! to the image patch around the seed and locates the unique
//! stationary point of the resulting quadratic. The Hessian
//! `[2a b; b 2c]` must have negative determinant (a saddle) for the
//! corner to be accepted.

use super::{CornerRefiner, PositionCovariance, RefineContext, RefineResult, RefineStatus};
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
/// The surface is fitted on the integer pixel grid around the rounded
/// seed, so the least-squares normal matrix depends only on
/// [`SaddlePointConfig::radius`] and is inverted once at construction.
/// Refining a corner allocates nothing.
///
/// A `radius < 1` patch cannot determine a quadratic; such a refiner
/// reports [`RefineStatus::IllConditioned`] for every in-bounds seed.
#[derive(Debug)]
pub struct SaddlePointRefiner {
    cfg: SaddlePointConfig,
    /// `(ΦᵀΦ)⁻¹` for the [`quadratic_basis`] over the `(2·radius+1)²`
    /// integer offsets; `None` when that matrix is singular (`radius < 1`).
    normal_inv: Option<[[f32; 6]; 6]>,
}

impl SaddlePointRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: SaddlePointConfig) -> Self {
        Self {
            cfg,
            normal_inv: inverse_normal_matrix(cfg.radius),
        }
    }
}

impl CornerRefiner for SaddlePointRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        self.cfg.radius
    }

    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        let unmoved = |score: f32, status: RefineStatus| RefineResult {
            x: seed_xy[0],
            y: seed_xy[1],
            score,
            status,
            covariance_per_noise_variance: None,
        };

        let Some(img) = ctx.image else {
            return unmoved(0.0, RefineStatus::Rejected);
        };

        let cx = seed_xy[0].round() as i32;
        let cy = seed_xy[1].round() as i32;
        let r = self.cfg.radius;

        if !img.supports_patch(cx, cy, r) {
            return unmoved(0.0, RefineStatus::OutOfBounds);
        }
        let Some(normal_inv) = &self.normal_inv else {
            return unmoved(0.0, RefineStatus::IllConditioned);
        };

        let fit = match self.fit_patch(normal_inv, |dx, dy| img.sample(cx + dx, cy + dy)) {
            Ok(fit) => fit,
            Err(det_h) => return unmoved(det_h, RefineStatus::IllConditioned),
        };
        let x = cx as f32 + fit.u[0];
        let y = cy as f32 + fit.u[1];

        let max_off = self.cfg.max_offset.min(r as f32 + 0.5);
        if (x - seed_xy[0]).abs() > max_off || (y - seed_xy[1]).abs() > max_off {
            return unmoved(fit.det_h, RefineStatus::Rejected);
        }

        RefineResult {
            covariance_per_noise_variance: Some(fit.covariance(normal_inv)),
            ..RefineResult::accepted([x, y], (-fit.det_h).sqrt())
        }
    }
}

/// Saddle of the quadratic fitted to one patch.
#[derive(Clone, Copy, Debug)]
struct PatchSaddle {
    /// Stationary point relative to the patch centre pixel.
    u: [f32; 2],
    /// Quadratic coefficients `(a, b, c)`; the Hessian is `[2a b; b 2c]`.
    quad: [f32; 3],
    /// `det(H) = 4ac − b²`, negative for a saddle.
    det_h: f32,
}

impl PatchSaddle {
    #[inline]
    fn covariance(&self, normal_inv: &[[f32; 6]; 6]) -> PositionCovariance {
        position_covariance(normal_inv, self.quad, self.det_h, self.u)
    }
}

impl SaddlePointRefiner {
    /// Fit the quadratic surface to the `(2·radius+1)²` patch given by
    /// `sample(dx, dy)` (offsets from the patch centre) and locate its
    /// saddle. `Err(det_h)` when the surface is not a clear saddle or its
    /// stationary point is not finite.
    #[inline]
    fn fit_patch(
        &self,
        normal_inv: &[[f32; 6]; 6],
        sample: impl Fn(i32, i32) -> f32,
    ) -> Result<PatchSaddle, f32> {
        let r = self.cfg.radius;
        let mut sum = 0.0f32;
        for dy in -r..=r {
            for dx in -r..=r {
                sum += sample(dx, dy);
            }
        }
        let side = (2 * r + 1) as f32;
        let mean = sum / (side * side);

        // g = Φᵀz with z the mean-subtracted patch; the fit is θ = (ΦᵀΦ)⁻¹·g.
        let mut g = [0.0f32; 6];
        for dy in -r..=r {
            for dx in -r..=r {
                let z = sample(dx, dy) - mean;
                let phi = quadratic_basis(dx as f32, dy as f32);
                for (gk, phik) in g.iter_mut().zip(phi) {
                    *gk += phik * z;
                }
            }
        }
        let coef =
            |row: usize| -> f32 { normal_inv[row].iter().zip(&g).map(|(pk, gk)| pk * gk).sum() };
        let (a, b, c, d, e) = (coef(0), coef(1), coef(2), coef(3), coef(4));

        let det_h = 4.0 * a * c - b * b;
        if det_h > -self.cfg.det_margin || det_h.abs() < self.cfg.min_abs_det {
            return Err(det_h);
        }

        // Stationary point u = −H⁻¹·(d, e), H = [2a b; b 2c].
        let inv_det = 1.0 / det_h;
        let ux = -(2.0 * c * d - b * e) * inv_det;
        let uy = (b * d - 2.0 * a * e) * inv_det;
        // det_h can be exactly 0 when both thresholds are 0; a non-finite u
        // would pass the offset check, since NaN comparisons are false.
        if !(ux.is_finite() && uy.is_finite()) {
            return Err(det_h);
        }
        Ok(PatchSaddle {
            u: [ux, uy],
            quad: [a, b, c],
            det_h,
        })
    }
}

/// Regressors of the surface `a x² + b x y + c y² + d x + e y + f`.
#[inline]
fn quadratic_basis(x: f32, y: f32) -> [f32; 6] {
    [x * x, x * y, y * y, x, y, 1.0]
}

/// `(ΦᵀΦ)⁻¹` of the quadratic fit over the integer offsets
/// `[-radius, radius]²`, or `None` when the grid cannot determine a
/// quadratic (`radius < 1`).
fn inverse_normal_matrix(radius: i32) -> Option<[[f32; 6]; 6]> {
    // Augmented [ΦᵀΦ | I], reduced by Gauss–Jordan with partial pivoting.
    let mut m = [[0.0f64; 12]; 6];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let phi = quadratic_basis(dx as f32, dy as f32).map(f64::from);
            for (row, mrow) in m.iter_mut().enumerate() {
                for col in 0..6 {
                    mrow[col] += phi[row] * phi[col];
                }
            }
        }
    }
    for (row, mrow) in m.iter_mut().enumerate() {
        mrow[6 + row] = 1.0;
    }
    for col in 0..6 {
        let pivot = (col..6).max_by(|&p, &q| m[p][col].abs().total_cmp(&m[q][col].abs()))?;
        if m[pivot][col].abs() < 1e-9 {
            return None;
        }
        m.swap(col, pivot);
        let inv_diag = 1.0 / m[col][col];
        for v in &mut m[col] {
            *v *= inv_diag;
        }
        let pivot_row = m[col];
        for (row, mrow) in m.iter_mut().enumerate() {
            if row != col {
                let factor = mrow[col];
                for (v, pv) in mrow.iter_mut().zip(pivot_row) {
                    *v -= factor * pv;
                }
            }
        }
    }
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|col| m[row][6 + col] as f32)
    }))
}

/// First-order covariance of the saddle location per unit image-noise
/// variance (px² per gray-level²).
///
/// With white pixel noise of variance `σₙ²`, the surface coefficients
/// `θ = (a, b, c, d, e, f)` have covariance `σₙ²·(ΦᵀΦ)⁻¹`. The location
/// `u = −H⁻¹·(d, e)`, `H = [2a b; b 2c]`, has Jacobian
/// `J = −H⁻¹·[[2uₓ, u_y, 0, 1, 0, 0], [0, uₓ, 2u_y, 0, 1, 0]]`, so
/// `Σᵤ = σₙ²·J(ΦᵀΦ)⁻¹Jᵀ`; this returns `J(ΦᵀΦ)⁻¹Jᵀ`. `u` is in the fit's
/// own coordinates (relative to the rounded seed); the covariance does not
/// depend on that origin. The fit's residual is not used for the scale: on
/// a sharp corner it measures how far the patch is from a quadratic, which
/// biases the location but does not scatter it.
fn position_covariance(
    normal_inv: &[[f32; 6]; 6],
    [a, b, c]: [f32; 3],
    det_h: f32,
    [ux, uy]: [f32; 2],
) -> PositionCovariance {
    // −H⁻¹ for H = [2a b; b 2c].
    let inv_det = 1.0 / det_h;
    let neg_h_inv = [
        [-2.0 * c * inv_det, b * inv_det],
        [b * inv_det, -2.0 * a * inv_det],
    ];
    let du = [
        [2.0 * ux, uy, 0.0, 1.0, 0.0, 0.0],
        [0.0, ux, 2.0 * uy, 0.0, 1.0, 0.0],
    ];
    let jac: [[f32; 6]; 2] = std::array::from_fn(|i| {
        std::array::from_fn(|k| neg_h_inv[i][0] * du[0][k] + neg_h_inv[i][1] * du[1][k])
    });
    // J·(ΦᵀΦ)⁻¹, then contract with Jᵀ.
    let jp: [[f32; 6]; 2] = std::array::from_fn(|i| {
        std::array::from_fn(|k| (0..6).map(|l| jac[i][l] * normal_inv[l][k]).sum())
    });
    let cov = |i: usize, j: usize| -> f32 { (0..6).map(|k| jp[i][k] * jac[j][k]).sum() };
    PositionCovariance::new(cov(0, 0), cov(0, 1), cov(1, 1))
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_checkerboard;
    use super::*;
    use crate::imageview::ImageView;
    use chess_corners_testutil::{aa_chessboard, add_gaussian_noise, gaussian_blur};

    const SIZE: usize = 21;

    fn board(offset: (f32, f32), blur: f32) -> Vec<u8> {
        let mut img = aa_chessboard(SIZE, 10, offset, 40, 200);
        if blur > 0.0 {
            gaussian_blur(&mut img, SIZE, blur);
        }
        img
    }

    fn refine_default(img: &[u8], seed: [f32; 2]) -> RefineResult {
        let view = ImageView::from_u8_slice(SIZE, SIZE, img).unwrap();
        SaddlePointRefiner::new(SaddlePointConfig::default()).refine(
            seed,
            RefineContext {
                image: Some(view),
                response: None,
            },
        )
    }

    /// Empirical and predicted 1σ position scatter `([σx, σy], [σx, σy])`
    /// over `draws` noise realizations of one corner, seeded at the
    /// rounded true position.
    fn scatter(noise: f32, blur: f32, offset: (f32, f32), draws: u64) -> ([f64; 2], [f64; 2]) {
        let clean = board(offset, blur);
        // Gaussian noise plus the 8-bit rounding (variance 1/12).
        let noise_var = f64::from(noise * noise) + 1.0 / 12.0;
        let mut xs = Vec::new();
        let mut pred = [0.0f64; 2];
        for seed in 0..draws {
            let mut img = clean.clone();
            add_gaussian_noise(&mut img, noise, seed + 1);
            let res = refine_default(&img, [offset.0.round(), offset.1.round()]);
            if res.status != RefineStatus::Accepted {
                continue;
            }
            let cov = res.covariance_per_noise_variance.unwrap();
            pred[0] += f64::from(cov.xx);
            pred[1] += f64::from(cov.yy);
            xs.push([f64::from(res.x), f64::from(res.y)]);
        }
        let n = xs.len() as f64;
        assert!(n > 0.9 * draws as f64, "only {n} of {draws} draws accepted");
        let sigma = |k: usize| {
            let mean = xs.iter().map(|p| p[k]).sum::<f64>() / n;
            (xs.iter().map(|p| (p[k] - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        };
        let predicted = |k: usize| (noise_var * pred[k] / n).sqrt();
        ([sigma(0), sigma(1)], [predicted(0), predicted(1)])
    }

    /// Predicted (`σₙ²·covariance_per_noise_variance`) against empirical
    /// position scatter. 300 draws give the empirical σ a relative standard
    /// error of `1/sqrt(2·300) ≈ 4%`, so the ±15% band is ≈3.7 standard
    /// errors wide.
    #[test]
    fn covariance_predicts_noise_scatter() {
        for offset in [(10.3f32, 10.6f32), (10.45, 9.8)] {
            let (emp, pred) = scatter(4.0, 0.8, offset, 300);
            for k in 0..2 {
                let ratio = pred[k] / emp[k];
                assert!(
                    (0.85..=1.15).contains(&ratio),
                    "offset {offset:?} axis {k}: predicted σ {} vs empirical σ {} (ratio {ratio})",
                    pred[k],
                    emp[k]
                );
            }
        }
    }

    /// Full sweep of [`covariance_predicts_noise_scatter`] over noise, blur
    /// and subpixel offsets. A measurement, not a gate:
    /// `cargo test -p chess-corners-core --release covariance_matches_scatter
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn covariance_matches_scatter() {
        println!("noise  blur  offset       σx_emp  σx_pred  σy_emp  σy_pred  ratio_x  ratio_y");
        for &noise in &[1.0f32, 2.0, 4.0, 8.0] {
            for &blur in &[0.0f32, 0.8, 1.5] {
                for &(ox, oy) in &[(10.0f32, 10.0f32), (10.3, 10.6), (10.45, 9.8)] {
                    let ([ex, ey], [px, py]) = scatter(noise, blur, (ox, oy), 400);
                    println!(
                        "{noise:5.1}  {blur:4.1}  ({ox:5.2},{oy:5.2})  {ex:7.4}  {px:7.4}  {ey:7.4}  {py:7.4}  {:7.2}  {:7.2}",
                        px / ex,
                        py / ey
                    );
                }
            }
        }
    }

    /// The fit runs on integer offsets around the rounded seed; the
    /// fitted surface, and so the saddle, is the same as a fit centred on
    /// the fractional seed. Reference positions come from such a
    /// seed-centred fit.
    #[test]
    fn matches_seed_centred_fit() {
        // (corner, blur, noise σ, seed, seed-centred refined position)
        let cases = [
            (
                (10.0f32, 10.0f32),
                0.0f32,
                0.0f32,
                [10.0f32, 10.0],
                [10.0f32, 10.0],
            ),
            ((10.3, 10.6), 0.8, 0.0, [10.0, 11.0], [10.202804, 10.709558]),
            ((10.45, 9.8), 1.5, 4.0, [10.2, 9.7], [10.6069975, 9.603654]),
            ((9.7, 10.35), 0.8, 8.0, [9.6, 10.4], [9.912121, 10.220791]),
        ];
        for (offset, blur, noise, seed, expected) in cases {
            let mut img = board(offset, blur);
            if noise > 0.0 {
                add_gaussian_noise(&mut img, noise, 7);
            }
            let res = refine_default(&img, seed);
            assert_eq!(res.status, RefineStatus::Accepted, "corner {offset:?}");
            assert!(
                (res.x - expected[0]).abs() < 1e-4 && (res.y - expected[1]).abs() < 1e-4,
                "corner {offset:?}: got ({}, {}), expected {expected:?}",
                res.x,
                res.y
            );
        }
    }

    /// `covariance_per_noise_variance` equals `Σₖ (∂u/∂Iₖ)(∂u/∂Iₖ)ᵀ` over
    /// the patch pixels, the noise-propagation sum it linearizes. Central
    /// differences on f32 patches of smooth, skewed (perspective-like)
    /// saddles off the pixel centre, so `xy ≠ 0` and the `u`-dependent
    /// Jacobian terms matter.
    #[test]
    fn covariance_matches_finite_differences() {
        let refiner = SaddlePointRefiner::new(SaddlePointConfig::default());
        let normal_inv = refiner.normal_inv.unwrap();
        let r = refiner.cfg.radius;
        let side = (2 * r + 1) as usize;
        let eps = 0.25f32;
        // (edge-normal angles rad, saddle centre relative to the patch centre, edge width px)
        for (angles, centre, width) in [
            ([0.3f32, 1.4f32], [0.3f32, -0.2f32], 1.0f32),
            ([-0.2, 1.05], [-0.35, 0.15], 1.4),
        ] {
            let normals = angles.map(|a| [a.cos(), a.sin()]);
            let mut patch = vec![0.0f32; side * side];
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (dx as f32 - centre[0], dy as f32 - centre[1]);
                    let edge = |n: [f32; 2]| ((n[0] * x + n[1] * y) / width).tanh();
                    patch[((dy + r) as usize) * side + (dx + r) as usize] =
                        120.0 + 80.0 * edge(normals[0]) * edge(normals[1]);
                }
            }
            let fit_u = |patch: &[f32]| {
                refiner
                    .fit_patch(&normal_inv, |dx, dy| {
                        patch[((dy + r) as usize) * side + (dx + r) as usize]
                    })
                    .unwrap()
            };
            let fit = fit_u(&patch);
            let cov = fit.covariance(&normal_inv);

            let mut fd = [0.0f64; 3];
            for k in 0..patch.len() {
                let mut plus = patch.clone();
                plus[k] += eps;
                let mut minus = patch.clone();
                minus[k] -= eps;
                let (up, um) = (fit_u(&plus).u, fit_u(&minus).u);
                let gx = f64::from(up[0] - um[0]) / f64::from(2.0 * eps);
                let gy = f64::from(up[1] - um[1]) / f64::from(2.0 * eps);
                fd[0] += gx * gx;
                fd[1] += gx * gy;
                fd[2] += gy * gy;
            }

            let scale = (fd[0] * fd[2]).sqrt();
            let got = [cov.xx, cov.xy, cov.yy].map(f64::from);
            assert!(
                fd[1].abs() > 0.03 * scale,
                "fixture must have xy ≠ 0: {fd:?}"
            );
            for (name, g, f) in [
                ("xx", got[0], fd[0]),
                ("xy", got[1], fd[1]),
                ("yy", got[2], fd[2]),
            ] {
                assert!(
                    (g - f).abs() <= 1e-3 * scale,
                    "{angles:?} {name}: analytic {g:e} vs finite-difference {f:e}"
                );
            }
        }
    }

    /// With both determinant thresholds at zero a flat patch has
    /// `det(H) = 0` exactly; the refiner must not accept a non-finite point.
    #[test]
    fn zero_thresholds_never_accept_non_finite_points() {
        let flat = vec![128u8; SIZE * SIZE];
        let view = ImageView::from_u8_slice(SIZE, SIZE, &flat).unwrap();
        let cfg = SaddlePointConfig {
            det_margin: 0.0,
            min_abs_det: 0.0,
            ..SaddlePointConfig::default()
        };
        let res = SaddlePointRefiner::new(cfg).refine(
            [10.0, 10.0],
            RefineContext {
                image: Some(view),
                response: None,
            },
        );
        assert_eq!(res.status, RefineStatus::IllConditioned);
        assert_eq!([res.x, res.y], [10.0, 10.0]);
    }

    #[test]
    fn rejects_radius_below_one_without_panicking() {
        let img = board((10.3, 10.6), 0.8);
        let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
        for radius in [0, -1] {
            let cfg = SaddlePointConfig {
                radius,
                ..SaddlePointConfig::default()
            };
            let res = SaddlePointRefiner::new(cfg).refine(
                [10.0, 11.0],
                RefineContext {
                    image: Some(view),
                    response: None,
                },
            );
            assert_eq!(res.status, RefineStatus::IllConditioned, "radius {radius}");
            assert_eq!([res.x, res.y], [10.0, 11.0]);
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
