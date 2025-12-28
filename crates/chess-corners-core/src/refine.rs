//! Pluggable subpixel refinement backends for ChESS corners.
//!
//! The default pipeline uses a 5×5 center-of-mass refinement on the response
//! map (matching the legacy behavior). Alternative refiners operate on the
//! original image intensity patch and provide more discriminative scores and
//! acceptance logic.
use crate::imageview::ImageView;
use crate::ResponseMap;

/// Status of a refinement attempt.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RefineStatus {
    Accepted,
    Rejected,
    OutOfBounds,
    IllConditioned,
}

/// Result of refining a single corner candidate.
#[derive(Copy, Clone, Debug)]
pub struct RefineResult {
    pub xy: [f32; 2],
    pub score: f32,
    pub status: RefineStatus,
}

impl RefineResult {
    #[inline]
    pub fn accepted(xy: [f32; 2], score: f32) -> Self {
        Self {
            xy,
            score,
            status: RefineStatus::Accepted,
        }
    }
}

/// Inputs shared by refinement methods.
#[derive(Copy, Clone, Debug, Default)]
pub struct RefineContext<'a> {
    pub image: Option<ImageView<'a>>,
    pub response: Option<&'a ResponseMap>,
}

/// Trait implemented by pluggable refinement backends.
pub trait CornerRefiner {
    /// Half-width of the patch the refiner needs around the seed.
    fn radius(&self) -> i32;
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}

/// User-facing enum selecting a refinement backend.
#[derive(Clone, Debug)]
pub enum RefinerKind {
    CenterOfMass(CenterOfMassConfig),
    Forstner(ForstnerConfig),
    SaddlePoint(SaddlePointConfig),
}

impl Default for RefinerKind {
    fn default() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
}

/// Runtime refiner with reusable scratch buffers.
#[derive(Debug)]
pub enum Refiner {
    CenterOfMass(CenterOfMassRefiner),
    Forstner(ForstnerRefiner),
    SaddlePoint(SaddlePointRefiner),
}

impl Refiner {
    pub fn from_kind(kind: RefinerKind) -> Self {
        match kind {
            RefinerKind::CenterOfMass(cfg) => Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg)),
            RefinerKind::Forstner(cfg) => Refiner::Forstner(ForstnerRefiner::new(cfg)),
            RefinerKind::SaddlePoint(cfg) => Refiner::SaddlePoint(SaddlePointRefiner::new(cfg)),
        }
    }
}

impl CornerRefiner for Refiner {
    #[inline]
    fn radius(&self) -> i32 {
        match self {
            Refiner::CenterOfMass(r) => r.radius(),
            Refiner::Forstner(r) => r.radius(),
            Refiner::SaddlePoint(r) => r.radius(),
        }
    }

    #[inline]
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        match self {
            Refiner::CenterOfMass(r) => r.refine(seed_xy, ctx),
            Refiner::Forstner(r) => r.refine(seed_xy, ctx),
            Refiner::SaddlePoint(r) => r.refine(seed_xy, ctx),
        }
    }
}

/// Legacy center-of-mass refinement on the response map.
#[derive(Clone, Copy, Debug)]
pub struct CenterOfMassConfig {
    pub radius: i32,
}

impl Default for CenterOfMassConfig {
    fn default() -> Self {
        Self { radius: 2 }
    }
}

#[derive(Debug)]
pub struct CenterOfMassRefiner {
    cfg: CenterOfMassConfig,
}

impl CenterOfMassRefiner {
    pub fn new(cfg: CenterOfMassConfig) -> Self {
        Self { cfg }
    }
}

impl CornerRefiner for CenterOfMassRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        self.cfg.radius
    }

    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        let resp = match ctx.response {
            Some(r) => r,
            None => {
                return RefineResult {
                    xy: seed_xy,
                    score: 0.0,
                    status: RefineStatus::Rejected,
                }
            }
        };

        let x = seed_xy[0].round() as i32;
        let y = seed_xy[1].round() as i32;
        let r = self.cfg.radius;

        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sw = 0.0;

        let w = resp.w as i32;
        let h = resp.h as i32;

        if x < r || y < r || x >= w - r || y >= h - r {
            return RefineResult {
                xy: seed_xy,
                score: 0.0,
                status: RefineStatus::OutOfBounds,
            };
        }

        for dy in -r..=r {
            let yy = (y + dy).clamp(0, h - 1) as usize;
            for dx in -r..=r {
                let xx = (x + dx).clamp(0, w - 1) as usize;
                let w_px = resp.at(xx, yy).max(0.0);
                sx += (xx as f32) * w_px;
                sy += (yy as f32) * w_px;
                sw += w_px;
            }
        }

        if sw > 0.0 {
            RefineResult::accepted([sx / sw, sy / sw], sw)
        } else {
            RefineResult {
                xy: seed_xy,
                score: 0.0,
                status: RefineStatus::Accepted,
            }
        }
    }
}

/// Förstner-style gradient-based refiner.
#[derive(Clone, Copy, Debug)]
pub struct ForstnerConfig {
    pub radius: i32,
    pub min_trace: f32,
    pub min_det: f32,
    pub max_condition_number: f32,
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

#[derive(Debug)]
pub struct ForstnerRefiner {
    cfg: ForstnerConfig,
}

impl ForstnerRefiner {
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
                    xy: seed_xy,
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
                xy: seed_xy,
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
                xy: seed_xy,
                score: det,
                status: RefineStatus::IllConditioned,
            };
        }

        let discr = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda_min = 0.5 * (trace - discr);
        let lambda_max = 0.5 * (trace + discr);

        if lambda_min <= 0.0 {
            return RefineResult {
                xy: seed_xy,
                score: det,
                status: RefineStatus::IllConditioned,
            };
        }

        let cond = lambda_max / lambda_min;
        if !cond.is_finite() || cond > self.cfg.max_condition_number {
            return RefineResult {
                xy: seed_xy,
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
                xy: seed_xy,
                score: det,
                status: RefineStatus::Rejected,
            };
        }

        let score = det / (trace * trace + 1e-6);
        RefineResult::accepted([seed_xy[0] + ux, seed_xy[1] + uy], score)
    }
}

/// Quadratic saddle-point surface refiner.
#[derive(Clone, Copy, Debug)]
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
                    xy: seed_xy,
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
                xy: seed_xy,
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
                    xy: seed_xy,
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
                xy: seed_xy,
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
                xy: seed_xy,
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
    use super::*;

    fn synthetic_checkerboard(size: usize, offset: (f32, f32), dark: u8, bright: u8) -> Vec<u8> {
        let mut img = vec![0u8; size * size];
        let ox = offset.0;
        let oy = offset.1;
        for y in 0..size {
            for x in 0..size {
                let xf = x as f32 - ox;
                let yf = y as f32 - oy;
                let dark_quad = (xf >= 0.0 && yf >= 0.0) || (xf < 0.0 && yf < 0.0);
                img[y * size + x] = if dark_quad { dark } else { bright };
            }
        }
        // Mild blur to provide gradients.
        let mut blurred = img.clone();
        for y in 1..(size - 1) {
            for x in 1..(size - 1) {
                let mut acc = 0u32;
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        acc +=
                            img[(y as i32 + ky) as usize * size + (x as i32 + kx) as usize] as u32;
                    }
                }
                blurred[y * size + x] = (acc / 9) as u8;
            }
        }
        blurred
    }

    #[test]
    fn center_of_mass_matches_expected_centroid() {
        let mut resp = ResponseMap {
            w: 7,
            h: 7,
            data: vec![0.0; 49],
        };
        // Put asymmetric weights so the centroid is easy to predict.
        resp.data[3 * 7 + 3] = 10.0;
        resp.data[3 * 7 + 4] = 5.0;
        resp.data[4 * 7 + 3] = 5.0;
        resp.data[4 * 7 + 4] = 2.0;

        let mut refiner = CenterOfMassRefiner::new(CenterOfMassConfig { radius: 1 });
        let ctx = RefineContext {
            image: None,
            response: Some(&resp),
        };
        let res = refiner.refine([3.0, 3.0], ctx);
        assert_eq!(res.status, RefineStatus::Accepted);
        // Compute expected centroid explicitly.
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sw = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let xx = (3 + dx) as usize;
                let yy = (3 + dy) as usize;
                let w_px = resp.at(xx, yy).max(0.0);
                sx += xx as f32 * w_px;
                sy += yy as f32 * w_px;
                sw += w_px;
            }
        }
        let expected = [sx / sw, sy / sw];
        assert!((res.xy[0] - expected[0]).abs() < 1e-4);
        assert!((res.xy[1] - expected[1]).abs() < 1e-4);
    }

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
        let refined_err =
            ((res.xy[0] - true_xy[0]).powi(2) + (res.xy[1] - true_xy[1]).powi(2)).sqrt();
        assert!(
            refined_err <= seed_err * 1.6 && refined_err < 1.0,
            "refined_err {refined_err} seed_err {seed_err} res {:?}",
            res.xy
        );
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
        let refined_err =
            ((res.xy[0] - true_xy[0]).powi(2) + (res.xy[1] - true_xy[1]).powi(2)).sqrt();
        assert!(
            refined_err <= seed_err * 1.6 && refined_err < 1.0,
            "refined_err {refined_err} seed_err {seed_err} res {:?}",
            res.xy
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
