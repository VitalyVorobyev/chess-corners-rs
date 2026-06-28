//! Center-of-mass refiner: 5×5 weighted centroid on the response map.
//!
//! The library default subpixel refiner. Operates on the dense ChESS
//! response map (not the image intensity patch); cheapest of the
//! built-in refiners.

use super::{CornerRefiner, RefineContext, RefineResult, RefineStatus};
use serde::{Deserialize, Serialize};

/// Configuration for the [`CenterOfMassRefiner`].
///
/// Controls the size of the weighted-centroid window. The centroid is
/// computed over a `(2·radius+1)²` patch on the ChESS response map
/// centered at the rounded seed position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct CenterOfMassConfig {
    /// Half-width of the centroid window in response-map pixels.
    /// Default is `2` (a 5×5 window). Increasing this value makes the
    /// centroid less sensitive to single-pixel noise but may merge
    /// energy from nearby corners.
    ///
    /// Advanced tuning. The default is appropriate for most scenes.
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
/// response values contribute to the centroid. The cheapest built-in
/// refiner and the library default.
#[derive(Debug)]
pub struct CenterOfMassRefiner {
    cfg: CenterOfMassConfig,
}

impl CenterOfMassRefiner {
    /// Construct a refiner with the given configuration.
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
                    x: seed_xy[0],
                    y: seed_xy[1],
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
                x: seed_xy[0],
                y: seed_xy[1],
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
                x: seed_xy[0],
                y: seed_xy[1],
                score: 0.0,
                status: RefineStatus::Accepted,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResponseMap;

    #[test]
    fn center_of_mass_matches_expected_centroid() {
        let mut resp = ResponseMap {
            w: 7,
            h: 7,
            data: vec![0.0; 49],
        };
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
}
