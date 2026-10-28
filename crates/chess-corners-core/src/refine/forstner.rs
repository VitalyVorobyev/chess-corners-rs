//! Förstner-style gradient-based subpixel refinement.
//!
//! The Förstner operator fits a subpixel corner location by solving a
//! weighted least-squares system on the image gradient structure tensor
//! within a local window. The thresholds in [`ForstnerConfig`] control
//! when the system is well-conditioned enough to yield a reliable
//! estimate.
//!
//! Reference: Förstner, W. & Gülch, E. (1987). "A fast operator for
//! detection and precise location of distinct points, corners and
//! centres of circular features."

use super::{CornerRefiner, RefineContext, RefineResult, RefineStatus};
use crate::imageview::ImageView;
use crate::uncertainty::{
    knot, CovarianceModel, PixelGradients, PixelTile, SystematicModel, Uncertainty, EDGE_REACH,
};
use serde::{Deserialize, Serialize};

/// Systematic error of the single-shot Förstner estimate by edge width
/// (see [`SystematicModel`]). The estimate is pulled toward the window
/// centre with a gain that falls from about 0.78 on sharp edges through 0
/// near an edge width of 1.5 px (there it says nothing about where in the
/// pixel the corner is) to −0.2, an overshoot, on heavily blurred ones.
/// Fitted on camera-formed synthetic boards from two generators (cells
/// 6–12, rotations 0–45°, shear, lens blur σ 0.3–3) by
/// `tools/refiner_robustness/fit_systematic.py`.
const SYSTEMATIC: SystematicModel = SystematicModel {
    knots: &[
        knot(1.09, 0.778, 0.0252),  // rms 0.0716, n 5605
        knot(1.14, 0.769, 0.0164),  // rms 0.0719, n 7567
        knot(1.18, 0.723, 0.0152),  // rms 0.0852, n 7598
        knot(1.24, 0.618, 0.0163),  // rms 0.1229, n 6582
        knot(1.30, 0.416, 0.0731),  // rms 0.1892, n 9438
        knot(1.35, 0.338, 0.0765),  // rms 0.2090, n 8006
        knot(1.40, 0.193, 0.0452),  // rms 0.2167, n 5755
        knot(1.47, 0.152, 0.0385),  // rms 0.2470, n 6012
        knot(1.58, -0.143, 0.0511), // rms 0.3321, n 6052
        knot(1.68, -0.215, 0.0503), // rms 0.3414, n 7437
    ],
    // Real stereo calibration set: `tools/refiner_robustness/real_validation/fit_floor.py`.
    floor: 0.110,
};

/// Configuration for the [`ForstnerRefiner`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct ForstnerConfig {
    /// Half-size of the local gradient window, in image pixels (integer
    /// `>= 1`; the window is `(2·radius+1)²`). Default `2` gives a 5×5
    /// patch, large enough to capture the gradient structure around a
    /// corner while staying local.
    #[cfg_attr(feature = "schemars", schemars(range(min = 1)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub radius: i32,
    /// Minimum trace of the structure tensor (sum of eigenvalues; `>= 0`,
    /// in squared gradient units). Rejects flat regions with too little
    /// gradient energy; the default `25.0` is roughly a mean gradient
    /// magnitude of 5 per pixel in a 5×5 window.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    pub min_trace: f32,
    /// Minimum determinant of the structure tensor (product of
    /// eigenvalues; `>= 0`). Guards against singular systems where the
    /// least-squares solution is numerically unstable; the default `1e-3`
    /// rejects only truly degenerate cases.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    pub min_det: f32,
    /// Maximum ratio of the larger to the smaller structure-tensor
    /// eigenvalue (unitless, `>= 1`). A high ratio indicates an edge
    /// rather than a corner; the default `50.0` is permissive (literature
    /// values span roughly 10–100).
    #[cfg_attr(feature = "schemars", schemars(range(min = 1.0)))]
    pub max_condition_number: f32,
    /// Maximum displacement from the initial integer seed to the refined
    /// location, in pixels (`>= 0`). Larger offsets suggest a mislocated
    /// seed and the result is rejected; the default is `1.5`.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub max_offset: f32,
}

impl Default for ForstnerConfig {
    fn default() -> Self {
        Self {
            radius: 2,
            min_trace: 25.0,
            min_det: 1e-3,
            max_condition_number: 50.0,
            max_offset: 1.5,
        }
    }
}

/// Förstner structure-tensor subpixel refiner.
///
/// Solves a weighted least-squares system on the image gradient
/// structure tensor within a local window. Requires the image intensity
/// patch (passed via [`RefineContext::image`]); ignores the response map.
#[derive(Debug)]
pub struct ForstnerRefiner {
    cfg: ForstnerConfig,
    pub(crate) uncertainty: Uncertainty,
    grads: PixelGradients,
    tile: PixelTile,
}

impl ForstnerRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: ForstnerConfig) -> Self {
        Self {
            cfg,
            uncertainty: Uncertainty::default(),
            grads: PixelGradients::default(),
            tile: PixelTile::default(),
        }
    }

    /// Covariance model of the estimate `seed + u` on the window centred on
    /// pixel `centre`, with
    /// `a_inv = A⁻¹` of the structure tensor `A`.
    ///
    /// `u = A⁻¹b` with `A = Σ w g gᵀ` and `b = Σ w g gᵀ p` (`p` the pixel
    /// position relative to the seed), so `du = A⁻¹ Σ w (dg gᵀ + g dgᵀ)(p − u)`,
    /// and each central-difference gradient `g` is linear in two pairs of
    /// pixels.
    fn covariance(
        &mut self,
        img: &ImageView<'_>,
        centre: [i32; 2],
        seed: [f32; 2],
        u: [f32; 2],
        a_inv: [[f32; 2]; 2],
    ) -> Option<CovarianceModel> {
        let r = self.cfg.radius;
        let side = (2 * r + 3) as usize;
        self.tile.load(img, centre, (r + 1).max(EDGE_REACH));
        let tile = &self.tile;
        self.grads
            .reset([centre[0] - r - 1, centre[1] - r - 1], side);
        let mul = |v: [f32; 2]| {
            [
                a_inv[0][0] * v[0] + a_inv[0][1] * v[1],
                a_inv[1][0] * v[0] + a_inv[1][1] * v[1],
            ]
        };
        for dy in -r..=r {
            let gy = centre[1] + dy;
            for dx in -r..=r {
                let gx = centre[0] + dx;
                let g = [
                    0.5 * (tile.at(gx + 1, gy) - tile.at(gx - 1, gy)),
                    0.5 * (tile.at(gx, gy + 1) - tile.at(gx, gy - 1)),
                ];
                let p = [gx as f32 - seed[0], gy as f32 - seed[1]];
                let w = 1.0 / (1.0 + 0.5 * (p[0] * p[0] + p[1] * p[1]));
                let q = [p[0] - u[0], p[1] - u[1]];
                let gq = g[0] * q[0] + g[1] * q[1];
                // ∂u/∂gₓ = A⁻¹·w·(eₓ (g·q) + g qₓ), ∂gₓ/∂I(x ± 1) = ±½.
                let vx = mul([w * (gq + g[0] * q[0]), w * g[1] * q[0]]).map(|v| 0.5 * v);
                let vy = mul([w * g[0] * q[1], w * (gq + g[1] * q[1])]).map(|v| 0.5 * v);
                let (at, row) = (self.grads.index(gx, gy), self.grads.side());
                self.grads.add_at(at + 1, vx);
                self.grads.add_at(at - 1, vx.map(|v| -v));
                self.grads.add_at(at + row, vy);
                self.grads.add_at(at - row, vy.map(|v| -v));
            }
        }
        let noise = self.grads.covariance()?;
        let offset = [
            seed[0] + u[0] - centre[0] as f32,
            seed[1] + u[1] - centre[1] as f32,
        ];
        let systematic = SYSTEMATIC.covariance(offset, self.tile.edge_width(centre), 1.0);
        Some(CovarianceModel {
            noise,
            systematic,
            growth: [0.0; 2],
        })
    }
}

impl CornerRefiner for ForstnerRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        // Gradients sample one pixel beyond the interior, so reserve an extra pixel.
        self.cfg.radius + 1
    }

    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        let img = match ctx.image {
            Some(view) => view,
            None => return RefineResult::unmoved(seed_xy, 0.0, RefineStatus::Rejected),
        };

        let cx = seed_xy[0].round() as i32;
        let cy = seed_xy[1].round() as i32;
        let patch_r = self.cfg.radius;

        if !img.supports_patch(cx, cy, patch_r + 1) {
            return RefineResult::unmoved(seed_xy, 0.0, RefineStatus::OutOfBounds);
        }

        let mut a00 = 0.0;
        let mut a01 = 0.0;
        let mut a11 = 0.0;
        let mut bx = 0.0;
        let mut by = 0.0;

        for dy in -patch_r..=patch_r {
            let gy = cy + dy;
            for dx in -patch_r..=patch_r {
                let gx = cx + dx;

                let ix_plus = img.sample(gx + 1, gy);
                let ix_minus = img.sample(gx - 1, gy);
                let iy_plus = img.sample(gx, gy + 1);
                let iy_minus = img.sample(gx, gy - 1);

                let gx_f = 0.5 * (ix_plus - ix_minus);
                let gy_f = 0.5 * (iy_plus - iy_minus);

                let px = gx as f32 - seed_xy[0];
                let py = gy as f32 - seed_xy[1];
                let gxgx = gx_f * gx_f;
                let gxgy = gx_f * gy_f;
                let gygy = gy_f * gy_f;
                let dist2 = px * px + py * py;
                let w = 1.0 / (1.0 + 0.5 * dist2);

                a00 += w * gxgx;
                a01 += w * gxgy;
                a11 += w * gygy;

                // b = Σ w g gᵀ p  (derivation from minimizing first-moment error)
                bx += w * (gxgx * px + gxgy * py);
                by += w * (gxgy * px + gygy * py);
            }
        }

        let trace = a00 + a11;
        let det = a00 * a11 - a01 * a01;
        if trace < self.cfg.min_trace || det <= self.cfg.min_det {
            return RefineResult::unmoved(seed_xy, det, RefineStatus::IllConditioned);
        }

        let discr = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda_min = 0.5 * (trace - discr);
        let lambda_max = 0.5 * (trace + discr);

        if lambda_min <= 0.0 {
            return RefineResult::unmoved(seed_xy, det, RefineStatus::IllConditioned);
        }

        let cond = lambda_max / lambda_min;
        if !cond.is_finite() || cond > self.cfg.max_condition_number {
            return RefineResult::unmoved(seed_xy, det, RefineStatus::IllConditioned);
        }

        let inv_det = 1.0 / det;
        let ux = (a11 * bx - a01 * by) * inv_det;
        let uy = (-a01 * bx + a00 * by) * inv_det;

        let max_off = self.cfg.max_offset.min(self.cfg.radius as f32 + 0.5);
        if ux.abs() > max_off || uy.abs() > max_off {
            return RefineResult::unmoved(seed_xy, det, RefineStatus::Rejected);
        }

        let score = det / (trace * trace + 1e-6);
        let mut res = RefineResult::accepted([seed_xy[0] + ux, seed_xy[1] + uy], score);
        if self.uncertainty.reports() {
            let a_inv = [
                [a11 * inv_det, -a01 * inv_det],
                [-a01 * inv_det, a00 * inv_det],
            ];
            res.set_model(self.covariance(&img, [cx, cy], seed_xy, [ux, uy], a_inv));
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{synthetic_checkerboard, tanh_saddle};
    use super::*;
    use crate::imageview::ImageView;

    /// A refiner reporting a covariance, which these tests examine.
    fn propagated(cfg: ForstnerConfig) -> ForstnerRefiner {
        let mut refiner = ForstnerRefiner::new(cfg);
        refiner.uncertainty = Uncertainty::Propagated;
        refiner
    }

    /// `covariance_per_noise_variance` is `Σₚ (∂x̂/∂Iₚ)(∂x̂/∂Iₚ)ᵀ`: central
    /// differences of the full estimate over every pixel (±1 gray level;
    /// the estimate is smooth in the pixels) on skewed saddles off the
    /// window centre.
    #[test]
    fn covariance_matches_finite_differences() {
        const SIDE: usize = 15;
        let seed = [7.0f32, 7.0];
        for (angles, offset, width) in [
            ([0.3f32, 1.4f32], [0.3f32, -0.2f32], 1.0f32),
            ([-0.2, 1.05], [-0.35, 0.15], 1.6),
        ] {
            let mut img: Vec<u8> = tanh_saddle(SIDE, [7, 7], offset, angles, width, 2.0)
                .iter()
                .map(|v| v.round() as u8)
                .collect();
            let mut refiner = propagated(ForstnerConfig::default());
            let mut refine = |img: &[u8]| {
                let view = ImageView::from_u8_slice(SIDE, SIDE, img).unwrap();
                refiner.refine(seed, RefineContext::new(Some(view), None))
            };
            let base = refine(&img);
            assert_eq!(base.status, RefineStatus::Accepted);
            let cov = base.covariance_per_noise_variance.unwrap();
            let mut fd = [0.0f64; 3];
            for k in 0..img.len() {
                let v = img[k];
                img[k] = v + 1;
                let plus = refine(&img);
                img[k] = v - 1;
                let minus = refine(&img);
                img[k] = v;
                let gx = f64::from(plus.x - minus.x) / 2.0;
                let gy = f64::from(plus.y - minus.y) / 2.0;
                fd[0] += gx * gx;
                fd[1] += gx * gy;
                fd[2] += gy * gy;
            }
            let scale = (fd[0] * fd[2]).sqrt();
            assert!(
                fd[1].abs() > 0.03 * scale,
                "fixture must have xy ≠ 0: {fd:?}"
            );
            for (name, g, f) in [
                ("xx", cov.xx, fd[0]),
                ("xy", cov.xy, fd[1]),
                ("yy", cov.yy, fd[2]),
            ] {
                assert!(
                    (f64::from(g) - f).abs() <= 2e-3 * scale,
                    "{angles:?} {name}: analytic {g:e} vs finite-difference {f:e}"
                );
            }
        }
    }

    #[test]
    fn forstner_refines_toward_true_offset() {
        let img = synthetic_checkerboard(15, (7.35, 7.8), 40, 220);
        let view = ImageView::from_u8_slice(15, 15, &img).unwrap();
        let ctx = RefineContext {
            image: Some(view),
            response: None,
        };
        let mut refiner = propagated(ForstnerConfig::default());
        let res = refiner.refine([7.0, 8.0], ctx);
        assert_eq!(res.status, RefineStatus::Accepted);
        let true_xy = [7.35f32, 7.8f32];
        let seed_err = ((7.0 - true_xy[0]).powi(2) + (8.0 - true_xy[1]).powi(2)).sqrt();
        let refined_err = ((res.x - true_xy[0]).powi(2) + (res.y - true_xy[1]).powi(2)).sqrt();
        assert!(
            refined_err <= seed_err * 1.6 && refined_err < 1.0,
            "refined_err {refined_err} seed_err {seed_err} res {:?}",
            (res.x, res.y)
        );
    }
}
