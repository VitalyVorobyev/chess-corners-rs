//! Center-of-mass refiner: 5×5 weighted centroid on the response map.
//!
//! The library default subpixel refiner. Operates on the dense ChESS
//! response map (not the image intensity patch); cheapest of the
//! built-in refiners.

use super::{CornerRefiner, RefineContext, RefineResult, RefineStatus};
use crate::detect::chess::response::ResponseStencil;
use crate::imageview::ImageView;
use crate::uncertainty::{
    knot, CovarianceModel, PixelGradients, PixelTile, SystematicModel, EDGE_REACH,
};
use crate::ResponseMap;
use serde::{Deserialize, Serialize};

/// Systematic error of the centroid by edge width (see
/// [`SystematicModel`]). The centroid's error does not scale with the
/// corner's offset from the window (gain 1): it comes from sampling the
/// response, varies with the board's rotation, and shrinks with blur. The
/// residual is its RMS per axis on camera-formed synthetic boards from two
/// generators (cells 6–12, rotations 0–45°, shear, lens blur σ 0.3–3),
/// fitted by `tools/refiner_robustness/fit_systematic.py`.
const SYSTEMATIC: SystematicModel = SystematicModel {
    knots: &[
        knot(1.09, 1.000, 0.0904), // rms 0.0863, n 5604
        knot(1.14, 1.000, 0.0877), // rms 0.0839, n 7561
        knot(1.18, 1.000, 0.0758), // rms 0.0733, n 7598
        knot(1.24, 1.000, 0.0706), // rms 0.0686, n 6581
        knot(1.30, 1.000, 0.0674), // rms 0.0656, n 9447
        knot(1.35, 1.000, 0.0664), // rms 0.0647, n 8027
        knot(1.40, 1.000, 0.0768), // rms 0.0743, n 5742
        knot(1.47, 1.000, 0.0633), // rms 0.0618, n 5999
        knot(1.58, 1.000, 0.0441), // rms 0.0436, n 6324
        knot(1.68, 1.000, 0.0445), // rms 0.0440, n 8222
    ],
    // Real stereo calibration set: `tools/refiner_robustness/real_validation/fit_floor.py`.
    floor: 0.074,
};

/// Configuration for the [`CenterOfMassRefiner`].
///
/// Controls the size of the weighted-centroid window. The centroid is
/// computed over a `(2·radius+1)²` patch on the ChESS response map
/// centered at the rounded seed position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct CenterOfMassConfig {
    /// Half-width of the centroid window, in response-map pixels (integer
    /// `>= 0`; the window is `(2·radius+1)²`). Default `2` (a 5×5 window);
    /// larger values smooth single-pixel noise but may merge energy from
    /// nearby corners.
    ///
    /// Advanced tuning. The default is appropriate for most scenes.
    #[cfg_attr(feature = "schemars", schemars(range(min = 0)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub radius: i32,
}

impl Default for CenterOfMassConfig {
    fn default() -> Self {
        Self { radius: 2 }
    }
}

/// Center-of-mass subpixel refiner.
///
/// Computes a response-weighted centroid in a `(2·radius+1)²` window
/// around the seed position on the ChESS response map. Only positive
/// response values contribute to the centroid; a window with no positive
/// response returns [`RefineStatus::IllConditioned`] at the unchanged
/// seed. The cheapest built-in refiner and the library default.
///
/// Its position covariance propagates image noise through the ChESS
/// response of every pixel in the window and through the centroid, so it
/// needs the image the response map was computed from in the
/// [`RefineContext`] and a map produced by
/// [`chess_response_u8`](crate::chess_response_u8) or
/// [`chess_response_u8_patch`](crate::chess_response_u8_patch).
#[derive(Debug)]
pub struct CenterOfMassRefiner {
    cfg: CenterOfMassConfig,
    pub(crate) reports_covariance: bool,
    grads: PixelGradients,
    tile: PixelTile,
}

impl CenterOfMassRefiner {
    /// Construct a refiner with the given configuration.
    pub fn new(cfg: CenterOfMassConfig) -> Self {
        Self {
            cfg,
            reports_covariance: false,
            grads: PixelGradients::default(),
            tile: PixelTile::default(),
        }
    }

    /// Covariance model of the centroid `xy` of the window centred on pixel
    /// `centre`, or `None` when the map is not a ChESS response.
    ///
    /// The centroid `x̂ = Σ wₚ xₚ / Σ wₚ` with `wₚ = max(Rₚ, 0)` has
    /// `∂x̂/∂Rₚ = (xₚ − x̂)/Σw` on every positive sample, and each `Rₚ` is
    /// piecewise linear in the ring and cross pixels around `p`.
    fn covariance(
        &mut self,
        img: &ImageView<'_>,
        resp: &ResponseMap,
        centre: [i32; 2],
        xy: [f32; 2],
        mass: f32,
    ) -> Option<CovarianceModel> {
        let ring_kind = resp.chess_ring?;
        let ring = ring_kind.offsets();
        let r = self.cfg.radius;
        let reach = r + ring_kind.radius() as i32;
        let side = (2 * reach + 1) as usize;
        // One copy of every pixel the response window and the edge width read.
        self.tile.load(img, centre, reach.max(EDGE_REACH));
        self.grads
            .reset([centre[0] - reach, centre[1] - reach], side);
        let stencil = ResponseStencil::new(ring, &self.tile, &self.grads);
        for dy in -r..=r {
            let yy = centre[1] + dy;
            for dx in -r..=r {
                let xx = centre[0] + dx;
                if resp.at(xx as usize, yy as usize) <= 0.0 {
                    continue;
                }
                let scale = [(xx as f32 - xy[0]) / mass, (yy as f32 - xy[1]) / mass];
                stencil.add_gradient(&self.tile, xx, yy, scale, &mut self.grads);
            }
        }
        let offset = [xy[0] - centre[0] as f32, xy[1] - centre[1] as f32];
        let width = self.tile.edge_width(centre);
        Some(SYSTEMATIC.model(self.grads.covariance()?, offset, width, 1.0))
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
            None => return RefineResult::unmoved(seed_xy, 0.0, RefineStatus::Rejected),
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
            return RefineResult::unmoved(seed_xy, 0.0, RefineStatus::OutOfBounds);
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
            let xy = [sx / sw, sy / sw];
            let mut res = RefineResult::accepted(xy, sw);
            if let (true, Some(img)) = (self.reports_covariance, ctx.image) {
                res.model = self.covariance(&img, resp, [x, y], xy, sw);
            }
            res
        } else {
            // No positive response in the window: there is no mass to
            // take a centroid of, so the seed cannot be refined.
            RefineResult::unmoved(seed_xy, 0.0, RefineStatus::IllConditioned)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::tanh_saddle;
    use super::*;
    use crate::detect::chess::ring::RING5;
    use crate::{chess_response_u8, ChessParams, ResponseMap};

    /// A refiner reporting a covariance, which these tests examine.
    fn propagated(cfg: CenterOfMassConfig) -> CenterOfMassRefiner {
        let mut refiner = CenterOfMassRefiner::new(cfg);
        refiner.reports_covariance = true;
        refiner
    }

    /// The ChESS response `SR − DR − 16·|μₙ − μₗ|` at `(x, y)` of an `f32`
    /// image: the reference the analytic gradient is checked against.
    fn response_f64(img: &[f64], side: usize, x: i32, y: i32) -> f64 {
        let at = |dx: i32, dy: i32| img[(y + dy) as usize * side + (x + dx) as usize];
        let s: [f64; 16] = core::array::from_fn(|k| at(RING5[k].0, RING5[k].1));
        let sr: f64 = (0..4)
            .map(|k| ((s[k] + s[k + 8]) - (s[k + 4] + s[k + 12])).abs())
            .sum();
        let dr: f64 = (0..8).map(|k| (s[k] - s[k + 8]).abs()).sum();
        let mu_n = s.iter().sum::<f64>() / 16.0;
        let mu_l = (at(0, 0) + at(0, -1) + at(0, 1) + at(1, 0) + at(-1, 0)) / 5.0;
        sr - dr - 16.0 * (mu_n - mu_l).abs()
    }

    /// The 5×5 centroid of [`response_f64`] around `centre`.
    fn centroid_f64(img: &[f64], side: usize, centre: [i32; 2]) -> [f64; 2] {
        let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (x, y) = (centre[0] + dx, centre[1] + dy);
                let w = response_f64(img, side, x, y).max(0.0);
                sx += f64::from(x) * w;
                sy += f64::from(y) * w;
                sw += w;
            }
        }
        [sx / sw, sy / sw]
    }

    /// `noise_covariance_per_unit_variance()` is `Σₚ (∂x̂/∂Iₚ)(∂x̂/∂Iₚ)ᵀ` over the
    /// image pixels: central differences (±0.01 gray level, inside the
    /// linear piece of every `|·|` of the response) of the centroid of a
    /// float re-implementation of the response, on 8-bit saddles off the
    /// window centre.
    #[test]
    fn covariance_matches_finite_differences() {
        const SIDE: usize = 21;
        let centre = [10i32, 10];
        for (angles, offset, width) in [
            ([0.3f32, 1.4f32], [0.3f32, -0.2f32], 1.0f32),
            ([-0.2, 1.05], [-0.35, 0.15], 1.6),
        ] {
            let img: Vec<u8> = tanh_saddle(SIDE, centre, offset, angles, width, 3.0)
                .iter()
                .map(|v| v.round() as u8)
                .collect();
            let resp = chess_response_u8(&img, SIDE, SIDE, &ChessParams::default());
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let v = resp.at((centre[0] + dx) as usize, (centre[1] + dy) as usize);
                    assert!(
                        v != 0.0,
                        "fixture has a zero response sample (kink of max(R, 0))"
                    );
                }
            }
            let view = ImageView::from_u8_slice(SIDE, SIDE, &img).unwrap();
            let res = propagated(CenterOfMassConfig::default()).refine(
                centre.map(|c| c as f32),
                RefineContext::new(Some(view), Some(&resp)),
            );
            assert_eq!(res.status, RefineStatus::Accepted);
            let cov = res.noise_covariance_per_unit_variance().unwrap();

            let mut f: Vec<f64> = img.iter().map(|&v| f64::from(v)).collect();
            let base = centroid_f64(&f, SIDE, centre);
            assert!(
                (base[0] - f64::from(res.x)).abs() < 1e-4
                    && (base[1] - f64::from(res.y)).abs() < 1e-4
            );
            let eps = 0.01f64;
            let mut fd = [0.0f64; 3];
            for k in 0..f.len() {
                let v = f[k];
                f[k] = v + eps;
                let plus = centroid_f64(&f, SIDE, centre);
                f[k] = v - eps;
                let minus = centroid_f64(&f, SIDE, centre);
                f[k] = v;
                let g = [0, 1].map(|a| (plus[a] - minus[a]) / (2.0 * eps));
                fd[0] += g[0] * g[0];
                fd[1] += g[0] * g[1];
                fd[2] += g[1] * g[1];
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
                    (f64::from(g) - f).abs() <= 1e-4 * scale,
                    "{angles:?} {name}: analytic {g:e} vs finite-difference {f:e}"
                );
            }
        }
    }

    /// Without the image, or on a map that is not a ChESS response, the
    /// centroid has no covariance; under `Off` neither.
    #[test]
    fn covariance_needs_image_and_chess_response() {
        const SIDE: usize = 21;
        let img: Vec<u8> = tanh_saddle(
            SIDE,
            [10, 10],
            [0.2, 0.1],
            [0.0, core::f32::consts::FRAC_PI_2],
            1.0,
            0.0,
        )
        .iter()
        .map(|v| v.round() as u8)
        .collect();
        let resp = chess_response_u8(&img, SIDE, SIDE, &ChessParams::default());
        let plain = ResponseMap::new(SIDE, SIDE, resp.data().to_vec());
        let view = ImageView::from_u8_slice(SIDE, SIDE, &img).unwrap();
        let mut refiner = propagated(CenterOfMassConfig::default());
        let with = refiner.refine([10.0, 10.0], RefineContext::new(Some(view), Some(&resp)));
        assert!(with.noise_covariance_per_unit_variance().is_some());
        for ctx in [
            RefineContext::new(None, Some(&resp)),
            RefineContext::new(Some(view), Some(&plain)),
        ] {
            let res = refiner.refine([10.0, 10.0], ctx);
            assert_eq!([res.x, res.y], [with.x, with.y]);
            assert!(res.noise_covariance_per_unit_variance().is_none());
        }
        refiner.reports_covariance = false;
        let off = refiner.refine([10.0, 10.0], RefineContext::new(Some(view), Some(&resp)));
        assert_eq!([off.x, off.y], [with.x, with.y]);
        assert!(off.noise_covariance_per_unit_variance().is_none());
    }

    #[test]
    fn center_of_mass_matches_expected_centroid() {
        let mut resp = ResponseMap::new(7, 7, vec![0.0; 49]);
        resp.data[3 * 7 + 3] = 10.0;
        resp.data[3 * 7 + 4] = 5.0;
        resp.data[4 * 7 + 3] = 5.0;
        resp.data[4 * 7 + 4] = 2.0;

        let mut refiner = propagated(CenterOfMassConfig { radius: 1 });
        let ctx = RefineContext {
            image: None,
            response: Some(&resp),
        };
        let res = refiner.refine([3.0, 3.0], ctx);
        assert_eq!(res.status, RefineStatus::Accepted);

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
        assert!((res.x - expected[0]).abs() < 1e-4);
        assert!((res.y - expected[1]).abs() < 1e-4);
    }

    #[test]
    fn window_without_positive_response_is_ill_conditioned() {
        // Zero and negative response everywhere in the window; a positive
        // value just outside it must not count.
        let mut resp = ResponseMap::new(7, 7, vec![-3.0; 49]);
        resp.data[3 * 7 + 3] = 0.0;
        resp.data[3 * 7 + 6] = 50.0;

        let mut refiner = propagated(CenterOfMassConfig { radius: 1 });
        let ctx = RefineContext {
            image: None,
            response: Some(&resp),
        };
        let res = refiner.refine([3.2, 2.9], ctx);
        assert_eq!(res.status, RefineStatus::IllConditioned);
        assert_eq!([res.x, res.y], [3.2, 2.9]);
        assert_eq!(res.score, 0.0);
        assert!(res.noise_covariance_per_unit_variance().is_none());
    }
}
