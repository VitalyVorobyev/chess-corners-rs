//! Saddle-point quadratic-surface refiner.
//!
//! Fits a 2nd-order surface `I(x, y) = a x² + b x y + c y² + d x + e y + f`
//! to a `(2·radius+1)²` grid of unit-spaced samples around the corner and
//! locates the stationary point of the quadratic. The Hessian
//! `[2a b; b 2c]` must have negative determinant (a saddle) for the corner
//! to be accepted.
//!
//! A quadratic fitted on a window that is not centred on the corner puts
//! the saddle partway back toward the window centre. The refiner therefore
//! refits on a patch resampled around the previous estimate until the
//! saddle lands on the window centre, where the window is point-symmetric
//! about the corner and the pull vanishes.

use super::{CornerRefiner, PositionCovariance, RefineContext, RefineResult, RefineStatus};
use crate::imageview::ImageView;
use crate::uncertainty::{edge_width, knot, PixelTile, SystematicModel, Uncertainty};
use serde::{Deserialize, Serialize};

/// Systematic error of the converged, re-centred estimate by edge width
/// (see [`SystematicModel`]): the window is centred on the estimate, so the
/// gain is 1 and only the residual remains — the error the fit keeps where
/// the corner sits between pixels (largest on sharp, rotated edges, which
/// alias) and on small, blurred cells. RMS per axis on camera-formed
/// synthetic boards from two generators (cells 6–12, rotations 0–45°,
/// shear, lens blur σ 0.3–3), fitted by
/// `tools/refiner_robustness/fit_systematic.py`.
const SYSTEMATIC: SystematicModel = SystematicModel {
    knots: &[
        knot(1.09, 1.000, 0.0066), // rms 0.0066, n 5700
        knot(1.14, 1.000, 0.0037), // rms 0.0037, n 7505
        knot(1.18, 1.000, 0.0041), // rms 0.0041, n 7591
        knot(1.24, 1.000, 0.0041), // rms 0.0041, n 6594
        knot(1.30, 1.000, 0.0078), // rms 0.0078, n 9476
        knot(1.35, 1.000, 0.0092), // rms 0.0092, n 7996
        knot(1.40, 1.000, 0.0108), // rms 0.0108, n 5794
        knot(1.47, 1.000, 0.0068), // rms 0.0068, n 5921
        knot(1.58, 1.000, 0.0061), // rms 0.0061, n 6357
        knot(1.68, 1.000, 0.0074), // rms 0.0074, n 8182
    ],
    // Real stereo calibration set: `tools/refiner_robustness/real_validation/fit_floor.py`.
    floor: 0.053,
};

/// Most re-centring steps per corner after the integer-grid fit.
const MAX_STEPS: usize = 16;
/// Re-centring has converged once the saddle fitted on the window lies
/// within this distance of the window centre, in pixels.
const STEP_TOL: f32 = 1e-3;
/// Integer-centred fits kept per corner; a converging corner needs 4–6.
const FIT_CACHE: usize = 9;
/// Step lengths tried (each half the previous) before re-centring stops.
const MAX_HALVINGS: usize = 4;
/// Smallest `det(−M)` of the window sensitivity `M` for which the Newton
/// step is taken; below it the plain step `o += u` is used. An isolated
/// corner has `−M ≈ g·I` with `g` in 0.4–1.
const MIN_GAIN_DET: f32 = 0.05;

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
/// The first fit samples the integer pixel grid around the rounded seed
/// and decides acceptance: its Hessian must pass the determinant
/// thresholds and its saddle must lie within
/// [`SaddlePointConfig::max_offset`] of the seed. The refiner then moves
/// the window, refitting on the patch resampled bilinearly around each new
/// position, until the fitted saddle lies within 0.001 px of the window
/// centre; this removes the pull toward the window centre that a single
/// off-centre fit has. Each move is a Newton step on the window position
/// (a plain step when the fit barely responds to moving the window),
/// halved until it brings the saddle closer to the window centre.
///
/// If re-centring does not converge — no shorter step helps, a refit is
/// not a clear saddle or leaves the image or the `max_offset` box, or 16
/// steps pass — the refiner returns the integer-grid fit, without a
/// covariance. A converged corner reports the covariance of the converged
/// position; where that position is only weakly determined by the window
/// (the fit hardly responds to moving it) the covariance is large but
/// finite.
///
/// On ~5 px cells with strong shear and blur the window also covers
/// neighbouring corners, and the converged position can be worse than the
/// integer-grid fit.
///
/// Resampling and the least-squares fit are both linear, so a refit is the
/// bilinear blend of the fits on the four surrounding integer-centred
/// patches. Those fits use one operator that depends only on
/// [`SaddlePointConfig::radius`] and is built at construction; a corner
/// needs four to six of them. Refining a corner allocates nothing.
///
/// A `radius < 1` patch cannot determine a quadratic; such a refiner
/// reports [`RefineStatus::IllConditioned`] for every in-bounds seed.
#[derive(Debug)]
pub struct SaddlePointRefiner {
    cfg: SaddlePointConfig,
    pub(crate) uncertainty: Uncertainty,
    /// The least-squares operator `R = (ΦᵀΦ)⁻¹Φᵀ` of the
    /// [`quadratic_basis`] over the `(2·radius+1)²` samples: the
    /// coefficients are `Σₖ Rₖ·zₖ`. `None` when `ΦᵀΦ` is singular
    /// (`radius < 1`).
    fit_op: Option<FitOperator>,
    scratch: Scratch,
}

/// Quadratic coefficients `(a, b, c, d, e)`; `f` is never needed.
type Coefs = [f32; 5];

/// Rows `a..e` of the least-squares fit operator `R`, stored by sample
/// (`columns[k] = Rₖ`), and its shifted self-overlaps.
#[derive(Debug)]
struct FitOperator {
    columns: Vec<Coefs>,
    /// `K(δ) = Σₖ Rₖ Rₖ₊δᵀ` over the samples `k` with `k + δ` on the grid,
    /// for `δ ∈ {−1, 0, 1}²` at index `3·(δy + 1) + δx + 1`.
    overlap: Box<[[Coefs; 5]; 9]>,
}

/// Per-corner working memory, sized at construction.
#[derive(Debug)]
struct Scratch {
    /// One `(2·radius+1)²` integer-centred patch, row-major.
    patch: Vec<f32>,
    /// Pixels around the estimate for the edge width.
    tile: PixelTile,
}

impl SaddlePointRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: SaddlePointConfig) -> Self {
        let side = (2 * cfg.radius.max(0) + 1) as usize;
        Self {
            cfg,
            uncertainty: Uncertainty::default(),
            fit_op: fit_operator(cfg.radius),
            scratch: Scratch {
                patch: vec![0.0; side * side],
                tile: PixelTile::default(),
            },
        }
    }
}

impl CornerRefiner for SaddlePointRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        self.cfg.radius
    }

    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        let unmoved =
            |score: f32, status: RefineStatus| RefineResult::unmoved(seed_xy, score, status);

        let Some(img) = ctx.image else {
            return unmoved(0.0, RefineStatus::Rejected);
        };

        let centre = [seed_xy[0].round() as i32, seed_xy[1].round() as i32];
        if !img.supports_patch(centre[0], centre[1], self.cfg.radius) {
            return unmoved(0.0, RefineStatus::OutOfBounds);
        }

        match self.estimate(&img, centre, seed_xy, MAX_STEPS, STEP_TOL) {
            Ok(est) => {
                let mut res = RefineResult {
                    converged: est.converged,
                    ..RefineResult::accepted(est.xy, est.score)
                };
                if let Some(noise) = est.covariance {
                    let at = est.xy.map(|v| v.round() as i32);
                    let width = edge_width(&img, at, &mut self.scratch.tile);
                    res.model = Some(SYSTEMATIC.model(noise, [0.0; 2], width, 1.0));
                }
                res
            }
            Err((score, status)) => unmoved(score, status),
        }
    }
}

/// Integer-pixel image access for the patch fits.
trait Pixels {
    /// Whether the square of half-size `radius` centred on pixel `(x, y)`
    /// lies inside the image.
    fn supports(&self, x: i32, y: i32, radius: i32) -> bool;
    /// Copy that square into `out`, row-major. Requires
    /// [`supports`](Pixels::supports).
    fn read_patch(&self, x: i32, y: i32, radius: i32, out: &mut [f32]);
}

impl Pixels for ImageView<'_> {
    #[inline]
    fn supports(&self, x: i32, y: i32, radius: i32) -> bool {
        self.supports_patch(x, y, radius)
    }

    #[inline]
    fn read_patch(&self, x: i32, y: i32, radius: i32, out: &mut [f32]) {
        let side = (2 * radius + 1) as usize;
        let x0 = (x + self.origin[0] - radius) as usize;
        let y0 = (y + self.origin[1] - radius) as usize;
        for (row, dst) in out.chunks_exact_mut(side).enumerate() {
            let start = (y0 + row) * self.width + x0;
            for (d, &v) in dst.iter_mut().zip(&self.data[start..start + side]) {
                *d = f32::from(v);
            }
        }
    }
}

/// Grid of unit-spaced samples centred `offset` px from an integer pixel.
///
/// Every sample shares the bilinear weights of the fractional part `t`:
/// sample `(dx, dy)` interpolates the pixels `base + (dx, dy) + {0, 1}²`.
#[derive(Clone, Copy, Debug)]
struct Window {
    base: [i32; 2],
    t: [f32; 2],
}

impl Window {
    #[inline]
    fn new(centre: [i32; 2], offset: [f32; 2]) -> Self {
        let floor = offset.map(f32::floor);
        Self {
            base: [centre[0] + floor[0] as i32, centre[1] + floor[1] as i32],
            t: [offset[0] - floor[0], offset[1] - floor[1]],
        }
    }

    /// Bilinear weight of pixel `base + (dx, dy) + (i, j)` in sample
    /// `(dx, dy)`, as `w[j][i]`. Exactly `[[1, 0], [0, 0]]` at `t = 0`.
    #[inline]
    fn weights(&self) -> [[f32; 2]; 2] {
        let [tx, ty] = self.t;
        [
            [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty)],
            [(1.0 - tx) * ty, tx * ty],
        ]
    }
}

/// Integer-centred patch fits of one corner, computed on first use and
/// kept by centre pixel (oldest overwritten first).
struct IntegerFits<'a, P: Pixels> {
    px: &'a P,
    fit_op: &'a FitOperator,
    radius: i32,
    patch: &'a mut [f32],
    centres: [[i32; 2]; FIT_CACHE],
    coefs: [Coefs; FIT_CACHE],
    len: usize,
    /// The four fits around `corners_base`, as `[j][i]`.
    corners: [[Coefs; 2]; 2],
    corners_base: Option<[i32; 2]>,
}

impl<'a, P: Pixels> IntegerFits<'a, P> {
    fn new(px: &'a P, fit_op: &'a FitOperator, radius: i32, patch: &'a mut [f32]) -> Self {
        Self {
            px,
            fit_op,
            radius,
            patch,
            centres: [[0; 2]; FIT_CACHE],
            coefs: [[0.0; 5]; FIT_CACHE],
            len: 0,
            corners: [[[0.0; 5]; 2]; 2],
            corners_base: None,
        }
    }

    /// Coefficients fitted on the patch centred on pixel `p`, or `None`
    /// when that patch leaves the image.
    fn at(&mut self, p: [i32; 2]) -> Option<Coefs> {
        let known = self.len.min(FIT_CACHE);
        if let Some(i) = self.centres[..known].iter().position(|&c| c == p) {
            return Some(self.coefs[i]);
        }
        if !self.px.supports(p[0], p[1], self.radius) {
            return None;
        }
        self.px.read_patch(p[0], p[1], self.radius, self.patch);
        // The operator rows annihilate constants; subtracting the centre
        // pixel only keeps the f32 sums small.
        let reference = self.patch[self.patch.len() / 2];
        let mut coefs = [0.0f32; 5];
        for (col, z) in self.fit_op.columns.iter().zip(self.patch.iter()) {
            let z = z - reference;
            for (c, r) in coefs.iter_mut().zip(col) {
                *c += r * z;
            }
        }
        let slot = self.len % FIT_CACHE;
        self.centres[slot] = p;
        self.coefs[slot] = coefs;
        self.len += 1;
        Some(coefs)
    }

    /// The integer-centred fits at `base + {0, 1}²`, as `c[j][i]`, or
    /// `None` when one leaves the image.
    fn corners(&mut self, base: [i32; 2]) -> Option<[[Coefs; 2]; 2]> {
        let mut out = [[[0.0; 5]; 2]; 2];
        for (j, row) in out.iter_mut().enumerate() {
            for (i, c) in row.iter_mut().enumerate() {
                *c = self.at([base[0] + i as i32, base[1] + j as i32])?;
            }
        }
        Some(out)
    }

    /// Saddle of the quadratic fitted on the patch resampled on `window`,
    /// or `None` when a fit leaves the image or is not a clear saddle.
    fn refit(&mut self, cfg: &SaddlePointConfig, window: &Window) -> Option<PatchSaddle> {
        if self.corners_base != Some(window.base) {
            self.corners = self.corners(window.base)?;
            self.corners_base = Some(window.base);
        }
        solve(cfg, blend(&self.corners, window)).ok()
    }

    /// [`slope`](Self::slope) from the four fits of the last
    /// [`refit`](Self::refit), or `None` when they are not around `window`.
    fn cached_slope(&self, window: &Window) -> Option<[Coefs; 2]> {
        if self.corners_base != Some(window.base) {
            return None;
        }
        let [[c00, c10], [c01, c11]] = self.corners;
        let [tx, ty] = window.t;
        Some([
            std::array::from_fn(|k| (1.0 - ty) * (c10[k] - c00[k]) + ty * (c11[k] - c01[k])),
            std::array::from_fn(|k| (1.0 - tx) * (c01[k] - c00[k]) + tx * (c11[k] - c10[k])),
        ])
    }

    /// `∂θ/∂t` of [`blend`] on `window`, one column per axis: the
    /// weighted difference of neighbouring integer-centred fits. Where `t`
    /// is 0 on an axis the forward difference is used, or the backward one
    /// when the forward fit leaves the image.
    fn slope(&mut self, window: &Window) -> Option<[Coefs; 2]> {
        let mut out = [[0.0f32; 5]; 2];
        for (axis, col) in out.iter_mut().enumerate() {
            let other = 1 - axis;
            let step = |mut p: [i32; 2], d: i32| {
                p[axis] += d;
                p
            };
            for (j, w) in [1.0 - window.t[other], window.t[other]]
                .into_iter()
                .enumerate()
            {
                if w == 0.0 {
                    continue;
                }
                let mut p = window.base;
                p[other] += j as i32;
                let here = self.at(p)?;
                let (lo, hi) = match self.at(step(p, 1)) {
                    Some(next) => (here, next),
                    None => (self.at(step(p, -1))?, here),
                };
                for ((c, h), l) in col.iter_mut().zip(hi).zip(lo) {
                    *c += w * (h - l);
                }
            }
        }
        Some(out)
    }
}

/// Saddle of the quadratic fitted to one patch.
#[derive(Clone, Copy, Debug)]
struct PatchSaddle {
    /// Stationary point relative to the window centre.
    u: [f32; 2],
    /// Quadratic coefficients `(a, b, c)`; the Hessian is `[2a b; b 2c]`.
    quad: [f32; 3],
    /// `det(H) = 4ac − b²`, negative for a saddle.
    det_h: f32,
}

/// Accepted refinement: absolute position, score and covariance.
#[derive(Clone, Copy, Debug)]
struct Estimate {
    xy: [f32; 2],
    score: f32,
    covariance: Option<PositionCovariance>,
    /// Re-centring converged to a fixed point with a regular window
    /// sensitivity; `false` for the integer-grid fallback.
    converged: bool,
}

impl SaddlePointRefiner {
    /// Re-centred saddle estimate for the seed `seed` (rounded to
    /// `centre`, whose patch must be inside the image): the integer-grid
    /// fit, then up to `max_steps` damped Newton steps on the window
    /// offset until the fitted saddle is within `tol` px of the window
    /// centre; the integer-grid fit, without covariance, when that fails
    /// (see [`SaddlePointRefiner`]). `Err((score, status))` when the first
    /// fit fails or leaves the `max_offset` box.
    fn estimate(
        &mut self,
        px: &impl Pixels,
        centre: [i32; 2],
        seed: [f32; 2],
        max_steps: usize,
        tol: f32,
    ) -> Result<Estimate, (f32, RefineStatus)> {
        let Self {
            cfg,
            uncertainty,
            fit_op,
            scratch,
        } = self;
        let Some(fit_op) = fit_op.as_ref() else {
            return Err((0.0, RefineStatus::IllConditioned));
        };
        let r = cfg.radius;
        let max_off = cfg.max_offset.min(r as f32 + 0.5);
        // `est` is relative to `centre`; the box is about the seed itself.
        let near_seed = |est: [f32; 2]| {
            (0..2).all(|k| ((centre[k] as f32 + est[k]) - seed[k]).abs() <= max_off)
        };

        let mut fits = IntegerFits::new(px, fit_op, r, &mut scratch.patch);
        let coefs = fits.at(centre).ok_or((0.0, RefineStatus::OutOfBounds))?;
        let mut fit = solve(cfg, coefs).map_err(|det_h| (det_h, RefineStatus::IllConditioned))?;
        if !near_seed(fit.u) {
            return Err((fit.det_h, RefineStatus::Rejected));
        }
        let single = Estimate {
            xy: [centre[0] as f32 + fit.u[0], centre[1] as f32 + fit.u[1]],
            score: (-fit.det_h).sqrt(),
            covariance: None,
            converged: false,
        };

        // Damped Newton on F(o) = u(o) = 0, with `o` the window offset from
        // `centre` and `u` the saddle offset fitted on that window. A step
        // is taken only if it shrinks |u|, so the iteration cannot cycle.
        let norm2 = |u: [f32; 2]| u[0] * u[0] + u[1] * u[1];
        let mut o = [0.0f32; 2];
        let mut window = Window::new(centre, o);
        for _ in 0..max_steps {
            if norm2(fit.u) < tol * tol {
                break;
            }
            // The first window is the integer grid: its slope would need two
            // extra fits, so it takes the plain step.
            let step = fits
                .cached_slope(&window)
                .and_then(|slope| newton_step(sensitivity(&fit, &slope).1, fit.u))
                .unwrap_or(fit.u);
            let mut scale = 1.0f32;
            let mut next = None;
            for _ in 0..MAX_HALVINGS {
                let o2 = [o[0] + scale * step[0], o[1] + scale * step[1]];
                let w2 = Window::new(centre, o2);
                if let Some(f2) = fits.refit(cfg, &w2) {
                    if norm2(f2.u) < norm2(fit.u) && near_seed([o2[0] + f2.u[0], o2[1] + f2.u[1]]) {
                        next = Some((o2, w2, f2));
                        break;
                    }
                }
                scale *= 0.5;
            }
            let Some(accepted) = next else {
                break;
            };
            (o, window, fit) = accepted;
        }
        if norm2(fit.u) >= tol * tol {
            return Ok(single);
        }

        let slope = match fits.cached_slope(&window) {
            Some(slope) => Some(slope),
            None => fits.slope(&window),
        };
        // The fixed point is regular when the window sensitivity `M` is
        // invertible: that alone decides convergence, under every tier. The
        // covariance (only under a reporting tier) is separate and may still
        // overflow to `None` on a nearly singular `M`.
        let regular = slope
            .map(|slope| sensitivity(&fit, &slope))
            .filter(|(_, m)| {
                let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
                det.is_finite() && det != 0.0
            });
        let converged = regular.is_some();
        let covariance = regular
            .filter(|_| uncertainty.reports())
            .and_then(|(jac, m)| covariance(fit_op, &window, &jac, m));
        Ok(Estimate {
            xy: [
                centre[0] as f32 + o[0] + fit.u[0],
                centre[1] as f32 + o[1] + fit.u[1],
            ],
            score: (-fit.det_h).sqrt(),
            covariance,
            converged,
        })
    }
}

/// `J = ∂u/∂θ` at `fit` and the window sensitivity `M = ∂u/∂o = J·slope`.
fn sensitivity(fit: &PatchSaddle, slope: &[Coefs; 2]) -> ([Coefs; 2], [[f32; 2]; 2]) {
    let jac = saddle_jacobian(fit);
    let dot = |a: &Coefs, b: &Coefs| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let m = jac.map(|row| [dot(&row, &slope[0]), dot(&row, &slope[1])]);
    (jac, m)
}

/// Newton step `−M⁻¹u` toward `u(o) = 0`, or `None` when the gain `−M` is
/// not clearly positive (the plain step `u` is used instead).
fn newton_step(m: [[f32; 2]; 2], u: [f32; 2]) -> Option<[f32; 2]> {
    let (g00, g01, g10, g11) = (-m[0][0], -m[0][1], -m[1][0], -m[1][1]);
    let det = g00 * g11 - g01 * g10;
    if !(det > MIN_GAIN_DET && g00 + g11 > 0.0) {
        return None;
    }
    // G⁻¹u.
    Some([
        (g11 * u[0] - g01 * u[1]) / det,
        (g00 * u[1] - g10 * u[0]) / det,
    ])
}

/// Coefficients fitted on the patch resampled on `window`: the bilinear
/// blend of the integer-centred fits `corners` around its base.
#[inline]
fn blend(corners: &[[Coefs; 2]; 2], window: &Window) -> Coefs {
    let w = window.weights();
    std::array::from_fn(|k| {
        w[0][0] * corners[0][0][k]
            + w[0][1] * corners[0][1][k]
            + w[1][0] * corners[1][0][k]
            + w[1][1] * corners[1][1][k]
    })
}

/// Saddle of the quadratic with coefficients `coefs`. `Err(det_h)` when
/// the surface is not a clear saddle or its stationary point is not finite.
fn solve(cfg: &SaddlePointConfig, [a, b, c, d, e]: Coefs) -> Result<PatchSaddle, f32> {
    let det_h = 4.0 * a * c - b * b;
    if det_h > -cfg.det_margin || det_h.abs() < cfg.min_abs_det {
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

/// First-order covariance of the re-centred estimate per unit image-noise
/// variance (px² per gray-level²), for `fit` on `window` with
/// `slope = ∂θ/∂t` of its coefficients.
///
/// The estimate is the fixed point `F(o, I) = 0` of the saddle offset
/// `F = u(θ(o, I))` fitted to the patch resampled at window offset `o`. By
/// the implicit-function theorem `∂ô/∂I = −M⁻¹·∂F/∂I` with
/// `M = ∂F/∂o = J·∂θ/∂t` and `J = ∂u/∂θ`. The coefficients are
/// `θ = R·W·I`, with `W` the bilinear resampling of the pixels `I`, so with
/// white pixel noise of variance `σₙ²` the position covariance is
/// `σₙ²·A·(R W Wᵀ Rᵀ)·Aᵀ` with `A = M⁻¹J`; this returns it per unit `σₙ²`.
/// A near-singular `M` gives a large, finite covariance; `None` only when
/// it overflows. `R W Wᵀ Rᵀ = Σ c_δ K(δ)`, where `c_δ`
/// sums the products of the bilinear weights of two pixels `δ` apart.
///
/// The fit's residual is not used for the scale: on a sharp corner it
/// measures how far the patch is from a quadratic, not the noise.
// Runs once per corner; kept out of line so the refit loop in `estimate`
// stays compact (~15% fewer instructions per corner, measured).
#[inline(never)]
fn covariance(
    fit_op: &FitOperator,
    window: &Window,
    jac: &[Coefs; 2],
    m: [[f32; 2]; 2],
) -> Option<PositionCovariance> {
    let dot = |a: &Coefs, b: &Coefs| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let inv_det = 1.0 / (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
    let m_inv = [
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ];
    let a: [Coefs; 2] =
        m_inv.map(|row| std::array::from_fn(|k| row[0] * jac[0][k] + row[1] * jac[1][k]));

    // c_δ = Σ w(p)·w(p + δ) over the weight pairs; pixel (i, j) has w[j][i].
    let w = window.weights();
    let mut c = [0.0f32; 9];
    for (j, wj) in w.iter().enumerate() {
        for (i, wij) in wj.iter().enumerate() {
            for (j2, wj2) in w.iter().enumerate() {
                for (i2, wij2) in wj2.iter().enumerate() {
                    c[3 * (1 + j2 - j) + 1 + i2 - i] += wij * wij2;
                }
            }
        }
    }
    let mut k = [[0.0f32; 5]; 5];
    for (cd, kd) in c.iter().zip(fit_op.overlap.iter()) {
        if *cd == 0.0 {
            continue;
        }
        for (krow, kdrow) in k.iter_mut().zip(kd) {
            for (v, d) in krow.iter_mut().zip(kdrow) {
                *v += cd * d;
            }
        }
    }

    let ak = a.map(|row| -> Coefs { std::array::from_fn(|s| dot(&row, &k.map(|kr| kr[s]))) });
    PositionCovariance::new(dot(&ak[0], &a[0]), dot(&ak[0], &a[1]), dot(&ak[1], &a[1])).finite()
}

/// Regressors of the surface `a x² + b x y + c y² + d x + e y + f`.
#[inline]
fn quadratic_basis(x: f32, y: f32) -> [f32; 6] {
    [x * x, x * y, y * y, x, y, 1.0]
}

/// `(ΦᵀΦ)⁻¹` of the quadratic fit over the integer offsets
/// `[-radius, radius]²`, or `None` when the grid cannot determine a
/// quadratic (`radius < 1`).
fn inverse_normal_matrix(radius: i32) -> Option<[[f64; 6]; 6]> {
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
        std::array::from_fn(|col| m[row][6 + col])
    }))
}

/// Rows `a..e` of `(ΦᵀΦ)⁻¹Φᵀ` for the [`quadratic_basis`] over the
/// integer offsets `[-radius, radius]²` (row-major) and their shifted
/// self-overlaps, or `None` when the grid cannot determine a quadratic
/// (`radius < 1`).
fn fit_operator(radius: i32) -> Option<FitOperator> {
    let normal_inv = inverse_normal_matrix(radius)?;
    let side = (2 * radius + 1) as usize;
    let n = side * side;
    let mut rows = Vec::with_capacity(5 * n);
    for coef in &normal_inv[..5] {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let phi = quadratic_basis(dx as f32, dy as f32).map(f64::from);
                rows.push(coef.iter().zip(phi).map(|(p, f)| p * f).sum::<f64>());
            }
        }
    }

    let mut overlap = Box::new([[[0.0f32; 5]; 5]; 9]);
    for (d, kd) in overlap.iter_mut().enumerate() {
        let (dx, dy) = (d as i64 % 3 - 1, d as i64 / 3 - 1);
        for (r, krow) in kd.iter_mut().enumerate() {
            for (s, v) in krow.iter_mut().enumerate() {
                let mut sum = 0.0f64;
                for y in 0..side as i64 {
                    for x in 0..side as i64 {
                        let (x2, y2) = (x + dx, y + dy);
                        if (0..side as i64).contains(&x2) && (0..side as i64).contains(&y2) {
                            let k = (y as usize) * side + x as usize;
                            let k2 = (y2 as usize) * side + x2 as usize;
                            sum += rows[r * n + k] * rows[s * n + k2];
                        }
                    }
                }
                *v = sum as f32;
            }
        }
    }
    Some(FitOperator {
        columns: (0..n)
            .map(|k| std::array::from_fn(|i| rows[i * n + k] as f32))
            .collect(),
        overlap,
    })
}

/// `J = ∂u/∂(a, b, c, d, e)` of the stationary point
/// `u = −H⁻¹·(d, e)`, `H = [2a b; b 2c]`, at `fit`:
/// `J = −H⁻¹·[[2uₓ, u_y, 0, 1, 0], [0, uₓ, 2u_y, 0, 1]]`.
fn saddle_jacobian(fit: &PatchSaddle) -> [[f32; 5]; 2] {
    let [a, b, c] = fit.quad;
    let [ux, uy] = fit.u;
    // −H⁻¹ for H = [2a b; b 2c].
    let inv_det = 1.0 / fit.det_h;
    let neg_h_inv = [
        [-2.0 * c * inv_det, b * inv_det],
        [b * inv_det, -2.0 * a * inv_det],
    ];
    let du = [[2.0 * ux, uy, 0.0, 1.0, 0.0], [0.0, ux, 2.0 * uy, 0.0, 1.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|k| neg_h_inv[i][0] * du[0][k] + neg_h_inv[i][1] * du[1][k])
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_checkerboard;
    use super::*;
    use crate::imageview::ImageView;
    use chess_corners_testutil::{
        aa_chessboard, add_gaussian_noise, area_chessboard, gaussian_blur,
    };

    /// A refiner reporting a covariance, which these tests examine.
    fn propagated(cfg: SaddlePointConfig) -> SaddlePointRefiner {
        let mut refiner = SaddlePointRefiner::new(cfg);
        refiner.uncertainty = Uncertainty::Propagated;
        refiner
    }

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
        propagated(SaddlePointConfig::default()).refine(
            seed,
            RefineContext {
                image: Some(view),
                response: None,
            },
        )
    }

    /// Empirical and predicted 1σ position scatter `([σx, σy], [σx, σy])`
    /// over `draws` noise realizations of one corner, seeded at the
    /// rounded true position. Counts only converged corners (those with a
    /// covariance); at least 90% of the draws must be.
    fn scatter(noise: f32, blur: f32, offset: (f32, f32), draws: u64) -> ([f64; 2], [f64; 2]) {
        let clean = board(offset, blur);
        // The total noise of the 8-bit image, Gaussian plus rounding: the σ²
        // `position_covariance` takes.
        let noise_var = f64::from(noise * noise) + f64::from(crate::uncertainty::ROUNDING_VAR);
        let mut xs = Vec::new();
        let mut pred = [0.0f64; 2];
        for seed in 0..draws {
            let mut img = clean.clone();
            add_gaussian_noise(&mut img, noise, seed + 1);
            let res = refine_default(&img, [offset.0.round(), offset.1.round()]);
            let Some(cov) = res.noise_covariance_per_unit_variance() else {
                continue;
            };
            pred[0] += f64::from(cov.xx);
            pred[1] += f64::from(cov.yy);
            xs.push([f64::from(res.x), f64::from(res.y)]);
        }
        let n = xs.len() as f64;
        assert!(
            n > 0.9 * draws as f64,
            "only {n} of {draws} draws converged"
        );
        let sigma = |k: usize| {
            let mean = xs.iter().map(|p| p[k]).sum::<f64>() / n;
            (xs.iter().map(|p| (p[k] - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        };
        let predicted = |k: usize| (noise_var * pred[k] / n).sqrt();
        ([sigma(0), sigma(1)], [predicted(0), predicted(1)])
    }

    /// Predicted (`σₙ²·noise_covariance_per_unit_variance()`) against empirical
    /// position scatter. 300 draws give the empirical σ a relative standard
    /// error of `1/sqrt(2·300) ≈ 4%`, so the ±15% band is ≈3.7 standard
    /// errors wide.
    #[test]
    fn covariance_predicts_noise_scatter() {
        for (offset, blur) in [
            ((10.3f32, 10.6f32), 0.8f32),
            ((10.45, 9.8), 0.8),
            ((10.3, 10.6), 2.5),
        ] {
            let (emp, pred) = scatter(4.0, blur, offset, 300);
            for k in 0..2 {
                let ratio = pred[k] / emp[k];
                assert!(
                    (0.85..=1.15).contains(&ratio),
                    "offset {offset:?} blur {blur} axis {k}: predicted σ {} vs empirical σ {} (ratio {ratio})",
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

    /// Noise-free error on corners rendered by exact pixel-area
    /// integration, over a 5×5 grid of subpixel phases. A single fit on the
    /// integer grid is pulled toward the rounded seed by up to 0.13 px
    /// here; the re-centred estimate is not.
    #[test]
    fn recentred_fit_is_unbiased_on_area_rendered_corners() {
        // (axis angle, blur σ, bound on the worst error in px)
        for (angle_deg, blur, bound) in [
            (90.0f32, 0.0f32, 0.005f32),
            (90.0, 1.5, 0.01),
            (60.0, 0.0, 0.03),
            (60.0, 1.5, 0.02),
        ] {
            let mut worst = 0.0f32;
            for k in 0..25 {
                let truth = (10.0 + (k % 5) as f32 / 5.0, 10.0 + (k / 5) as f32 / 5.0);
                let mut img = area_chessboard(SIZE, 8, truth, angle_deg.to_radians(), 40, 200);
                gaussian_blur(&mut img, SIZE, blur);
                let res = refine_default(&img, [truth.0.round(), truth.1.round()]);
                assert_eq!(res.status, RefineStatus::Accepted, "corner {truth:?}");
                worst = worst.max((res.x - truth.0).hypot(res.y - truth.1));
            }
            assert!(
                worst < bound,
                "angle {angle_deg}, blur {blur}: worst error {worst} px (bound {bound})"
            );
        }
    }

    /// `f32` image for exact pixel perturbations.
    struct FloatImage {
        side: usize,
        data: Vec<f32>,
    }

    impl Pixels for FloatImage {
        fn supports(&self, x: i32, y: i32, radius: i32) -> bool {
            let max = self.side as i32 - 1;
            x - radius >= 0 && y - radius >= 0 && x + radius <= max && y + radius <= max
        }

        fn read_patch(&self, x: i32, y: i32, radius: i32, out: &mut [f32]) {
            let side = (2 * radius + 1) as usize;
            for (row, dst) in out.chunks_exact_mut(side).enumerate() {
                let start = (y - radius) as usize * self.side + row * self.side;
                let start = start + (x - radius) as usize;
                dst.copy_from_slice(&self.data[start..start + side]);
            }
        }
    }

    /// `noise_covariance_per_unit_variance()` equals `Σₚ (∂ĉ/∂Iₚ)(∂ĉ/∂Iₚ)ᵀ` over
    /// the image pixels for the converged re-centred estimate `ĉ`, the
    /// noise-propagation sum it linearizes. Central differences of the full
    /// iterated estimate (run to a tight tolerance) on `f32` images of
    /// smooth, skewed (perspective-like) saddles off the pixel centre, so
    /// the window is resampled and `xy ≠ 0`.
    #[test]
    fn covariance_matches_finite_differences() {
        const SIDE: usize = 15;
        let centre = [7i32, 7];
        let mut refiner = propagated(SaddlePointConfig::default());
        let eps = 1.0f32;
        // (edge-normal angles rad, saddle position relative to `centre`, edge width px)
        for (angles, offset, width) in [
            ([0.3f32, 1.4f32], [0.3f32, -0.2f32], 1.0f32),
            ([-0.2, 1.05], [-0.35, 0.15], 1.4),
        ] {
            let normals = angles.map(|a| [a.cos(), a.sin()]);
            let mut img = FloatImage {
                side: SIDE,
                data: vec![0.0; SIDE * SIDE],
            };
            for (k, v) in img.data.iter_mut().enumerate() {
                let x = (k % SIDE) as f32 - centre[0] as f32 - offset[0];
                let y = (k / SIDE) as f32 - centre[1] as f32 - offset[1];
                let edge = |n: [f32; 2]| ((n[0] * x + n[1] * y) / width).tanh();
                *v = 120.0 + 80.0 * edge(normals[0]) * edge(normals[1]);
            }
            let seed = centre.map(|c| c as f32);
            let mut estimate = |img: &FloatImage| {
                refiner
                    .estimate(img, centre, seed, 200, 1e-7)
                    .expect("fixture is a clear saddle")
            };
            let base = estimate(&img);
            let cov = base.covariance.unwrap();
            assert!(
                (base.xy[0] - seed[0] - offset[0]).abs() < 0.01
                    && (base.xy[1] - seed[1] - offset[1]).abs() < 0.01,
                "{angles:?}: converged to {:?}, saddle at {offset:?} from {seed:?}",
                base.xy
            );

            let mut fd = [0.0f64; 3];
            for k in 0..img.data.len() {
                let v = img.data[k];
                img.data[k] = v + eps;
                let plus = estimate(&img).xy;
                img.data[k] = v - eps;
                let minus = estimate(&img).xy;
                img.data[k] = v;
                let gx = f64::from(plus[0] - minus[0]) / f64::from(2.0 * eps);
                let gy = f64::from(plus[1] - minus[1]) / f64::from(2.0 * eps);
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

    /// A seed at the minimum border margin whose re-centred window would
    /// need a pixel outside the image cannot converge: the refiner returns
    /// the integer-grid fit, without a covariance.
    #[test]
    fn early_exit_returns_single_fit_without_covariance() {
        // Corner 0.3 px left of the seed column x = 2: every refit window
        // toward it needs column −1.
        let img = board((1.7, 10.4), 0.8);
        let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
        let res = refine_default(&img, [2.0, 10.0]);
        assert_eq!(res.status, RefineStatus::Accepted);
        assert!(res.noise_covariance_per_unit_variance().is_none());
        let single = propagated(SaddlePointConfig::default())
            .estimate(&view, [2, 10], [2.0, 10.0], 0, STEP_TOL)
            .unwrap();
        assert_eq!([res.x, res.y], single.xy);
        // A refit would have moved it: the single fit is visibly biased.
        assert!((res.x - 1.7).hypot(res.y - 10.4) > 0.05);
    }

    /// Running out of steps before the saddle reaches the window centre is
    /// non-convergence too: integer-grid fit, no covariance. The same
    /// corner converges within the default step budget.
    #[test]
    fn step_cap_returns_single_fit_without_covariance() {
        let img = board((10.4, 9.65), 0.0);
        let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
        let mut refiner = propagated(SaddlePointConfig::default());
        let (centre, seed) = ([10, 10], [10.0, 10.0]);
        let single = refiner.estimate(&view, centre, seed, 0, STEP_TOL).unwrap();
        let capped = refiner.estimate(&view, centre, seed, 1, STEP_TOL).unwrap();
        assert!(capped.covariance.is_none());
        assert_eq!(capped.xy, single.xy);
        let full = refiner
            .estimate(&view, centre, seed, MAX_STEPS, STEP_TOL)
            .unwrap();
        assert!(full.covariance.is_some());
        let err = |xy: [f32; 2]| (xy[0] - 10.4).hypot(xy[1] - 9.65);
        assert!(
            err(full.xy) < 0.5 * err(single.xy),
            "{full:?} vs {single:?}"
        );
    }

    /// Convergence, position and status do not depend on the tier: one
    /// predicate (a regular window sensitivity) decides convergence under
    /// both, the covariance is computed separately. Sweeps blur, noise and
    /// subpixel phase, including corners near the image border.
    #[test]
    fn convergence_does_not_depend_on_the_tier() {
        let mut on = propagated(SaddlePointConfig::default());
        let mut off = propagated(SaddlePointConfig::default());
        off.uncertainty = Uncertainty::Off;
        let mut checked = 0;
        for k in 0..60u64 {
            let offset = (
                2.4 + (k % 8) as f32 * 2.1 + 0.13 * k as f32 % 1.0,
                10.0 + (k % 5) as f32 * 0.21,
            );
            let mut img = board(offset, [0.0, 0.8, 2.5][k as usize % 3]);
            add_gaussian_noise(&mut img, [0.0, 1.0, 8.0][k as usize % 3], k);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            for seed in [
                [offset.0.round(), offset.1.round()],
                [offset.0.floor(), offset.1.ceil()],
            ] {
                let ctx = RefineContext::new(Some(view), None);
                let (a, b) = (on.refine(seed, ctx), off.refine(seed, ctx));
                assert_eq!(
                    (a.x.to_bits(), a.y.to_bits(), a.status, a.converged),
                    (b.x.to_bits(), b.y.to_bits(), b.status, b.converged),
                    "seed {seed:?} offset {offset:?}"
                );
                assert!(b.noise_covariance_per_unit_variance().is_none());
                checked += usize::from(a.converged);
            }
        }
        assert!(checked > 60, "{checked} converged");
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
        let res = propagated(cfg).refine(
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
            let res = propagated(cfg).refine(
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
        let mut refiner = propagated(SaddlePointConfig::default());
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
