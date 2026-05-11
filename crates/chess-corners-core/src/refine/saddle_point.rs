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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SaddlePointConfig {
    pub radius: i32,
    pub det_margin: f32,
    pub max_offset: f32,
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

#[derive(Debug)]
pub struct SaddlePointRefiner {
    cfg: SaddlePointConfig,
    m: [f32; 36],
    rhs: [f32; 6],
}

impl SaddlePointRefiner {
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

        let coeffs = match self.solve_6x6() {
            Some(c) => c,
            None => {
                return RefineResult {
                    x: seed_xy[0],
                    y: seed_xy[1],
                    score: 0.0,
                    status: RefineStatus::IllConditioned,
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
            };
        }

        let score = (-det_h).sqrt();
        RefineResult::accepted([seed_xy[0] + ux, seed_xy[1] + uy], score)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_checkerboard;
    use super::*;
    use crate::imageview::ImageView;

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
