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
use serde::{Deserialize, Serialize};

/// Configuration for the [`ForstnerRefiner`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ForstnerConfig {
    /// Half-size of the local gradient window (full window is
    /// `2*radius+1`). A radius of 2 gives a 5×5 patch — large enough to
    /// capture the gradient structure around a corner while staying
    /// local.
    pub radius: i32,
    /// Minimum trace of the structure tensor (sum of eigenvalues).
    /// Rejects flat regions where gradient energy is too low. The
    /// value 25.0 corresponds roughly to an average gradient magnitude
    /// of ~5 per pixel in a 5×5 window (5² = 25), filtering out
    /// textureless areas.
    pub min_trace: f32,
    /// Minimum determinant of the structure tensor (product of
    /// eigenvalues). Guards against singular or near-singular systems
    /// where the least-squares solution is numerically unstable. 1e-3
    /// is a conservative floor that rejects only truly degenerate
    /// cases.
    pub min_det: f32,
    /// Maximum ratio of the larger to the smaller eigenvalue. A high
    /// condition number indicates an edge rather than a corner (one
    /// dominant gradient direction). The threshold 50.0 is permissive
    /// — standard Harris/Förstner literature suggests values in the
    /// 10–100 range depending on noise level and corner sharpness.
    pub max_condition_number: f32,
    /// Maximum displacement (in pixels) from the initial integer seed
    /// to the refined subpixel location. Offsets larger than ~1.5 px
    /// suggest the seed was mislocated and the refinement is
    /// extrapolating rather than interpolating; such results are
    /// rejected.
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
}

impl ForstnerRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: ForstnerConfig) -> Self {
        Self { cfg }
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
            None => {
                return RefineResult {
                    x: seed_xy[0],
                    y: seed_xy[1],
                    score: 0.0,
                    status: RefineStatus::Rejected,
                }
            }
        };

        let cx = seed_xy[0].round() as i32;
        let cy = seed_xy[1].round() as i32;
        let patch_r = self.cfg.radius;

        if !img.supports_patch(cx, cy, patch_r + 1) {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: 0.0,
                status: RefineStatus::OutOfBounds,
            };
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
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det,
                status: RefineStatus::IllConditioned,
            };
        }

        let discr = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda_min = 0.5 * (trace - discr);
        let lambda_max = 0.5 * (trace + discr);

        if lambda_min <= 0.0 {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det,
                status: RefineStatus::IllConditioned,
            };
        }

        let cond = lambda_max / lambda_min;
        if !cond.is_finite() || cond > self.cfg.max_condition_number {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det,
                status: RefineStatus::IllConditioned,
            };
        }

        let inv_det = 1.0 / det;
        let ux = (a11 * bx - a01 * by) * inv_det;
        let uy = (-a01 * bx + a00 * by) * inv_det;

        let max_off = self.cfg.max_offset.min(self.cfg.radius as f32 + 0.5);
        if ux.abs() > max_off || uy.abs() > max_off {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: det,
                status: RefineStatus::Rejected,
            };
        }

        let score = det / (trace * trace + 1e-6);
        RefineResult::accepted([seed_xy[0] + ux, seed_xy[1] + uy], score)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_checkerboard;
    use super::*;
    use crate::imageview::ImageView;

    #[test]
    fn forstner_refines_toward_true_offset() {
        let img = synthetic_checkerboard(15, (7.35, 7.8), 40, 220);
        let view = ImageView::from_u8_slice(15, 15, &img).unwrap();
        let ctx = RefineContext {
            image: Some(view),
            response: None,
        };
        let mut refiner = ForstnerRefiner::new(ForstnerConfig::default());
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
