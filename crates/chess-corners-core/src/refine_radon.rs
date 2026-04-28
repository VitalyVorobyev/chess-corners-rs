//! Local Radon-peak subpixel refiner.
//!
//! This is a per-candidate adaptation of the localized Radon transform
//! introduced in Duda & Frese 2018, *"Accurate Detection and
//! Localization of Checkerboard Corners for Calibration"*. The original
//! paper proposes a *whole-image* detector in which the response is
//! evaluated on every pixel of a 2× supersampled image, post-blurred
//! with a small box filter, and the subpixel corner is recovered by a
//! Gaussian peak fit. The refiner implemented here performs the same
//! three steps, but only inside a small region of interest around an
//! existing ChESS candidate.
//!
//! # Algorithm
//!
//! For each seed `(cx, cy)`:
//!
//! 1. Evaluate the localized Radon response `(max − min)²` over four
//!    discrete ray angles `α ∈ {0, π/4, π/2, 3π/4}` on a dense grid of
//!    samples around the seed. Ray integration and response-grid
//!    sampling both use step `1 / image_upsample` physical pixels,
//!    equivalent to operating on a 2×-supersampled image (paper §3.1
//!    step 2).
//! 2. Smooth the response map with a `(2·response_blur_radius+1)²` box
//!    filter (paper §3.1 step 7, default 3×3).
//! 3. Locate the discrete argmax. Reject border hits.
//! 4. Fit a parabola in `x` and `y` through the argmax and its two
//!    neighbours along each axis. By default the fit is performed on
//!    `log(response)` ("Gaussian peak fit", paper §3.1 step 8), which
//!    is robust to mild plateauing of the raw response.
//!
//! The returned offset is in the seed's pixel frame; callers do not
//! need to know about the response-grid density.
//!
//! # Status
//!
//! The refiner is expected to recover sub-0.1 px accuracy on clean
//! chessboard patches with the default configuration. Noise-tolerance
//! follows the paper's empirical behaviour — smoothing of the response
//! map is what makes the peak fit stable rather than the ray
//! integration alone.

use serde::{Deserialize, Serialize};

use crate::imageview::ImageView;
use crate::radon::{box_blur_inplace, fit_peak_frac, ANGLES, DIR_COS, DIR_SIN};
use crate::refine::{CornerRefiner, RefineContext, RefineResult, RefineStatus};

pub use crate::radon::PeakFitMode;

/// Configuration for [`RadonPeakRefiner`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RadonPeakConfig {
    /// Half-length of each ray, in physical image pixels. The ray is
    /// sampled with step `1 / image_upsample` and contains
    /// `2·ray_radius·image_upsample + 1` samples in total.
    ///
    /// Paper default is 2 (9 samples over 4 px at 2× supersampling).
    pub ray_radius: u32,
    /// Half-size of the response-search window around the rounded seed,
    /// in physical image pixels. The response map has side
    /// `2·patch_radius·image_upsample + 1`.
    pub patch_radius: u32,
    /// Supersampling factor. `1` operates on the original grid, `2`
    /// (paper default) is equivalent to computing the response on a
    /// bilinearly-upsampled image. Applies both to ray-sample spacing
    /// and to the response-map grid density.
    pub image_upsample: u32,
    /// Half-size of the box blur applied to the response map. `0`
    /// disables blurring; `1` (paper default) yields a 3×3 box.
    pub response_blur_radius: u32,
    /// Peak-fit mode. Defaults to [`PeakFitMode::Gaussian`].
    pub peak_fit: PeakFitMode,
    /// Reject candidates whose peak squared response is below this
    /// threshold. `0.0` disables the filter — ChESS already rejected
    /// non-corner candidates upstream.
    pub min_response: f32,
    /// Reject refinements whose displacement from the rounded seed
    /// exceeds this many pixels. Mirrors
    /// [`SaddlePointConfig::max_offset`](crate::refine::SaddlePointConfig).
    pub max_offset: f32,
}

impl Default for RadonPeakConfig {
    fn default() -> Self {
        Self {
            ray_radius: 2,
            patch_radius: 3,
            image_upsample: 2,
            response_blur_radius: 1,
            peak_fit: PeakFitMode::Gaussian,
            min_response: 0.0,
            max_offset: 1.5,
        }
    }
}

impl RadonPeakConfig {
    #[inline]
    fn image_upsample_clamped(&self) -> u32 {
        self.image_upsample.max(1)
    }

    #[inline]
    fn side(&self) -> usize {
        (2 * self.patch_radius as usize * self.image_upsample_clamped() as usize) + 1
    }
}

/// Subpixel refiner built on a local Duda-Frese-style Radon response.
///
/// See the [module docs](self) for the algorithm and attribution.
#[derive(Debug)]
pub struct RadonPeakRefiner {
    cfg: RadonPeakConfig,
    /// Flattened response map of side [`Self::side`]. Sized once at
    /// construction and reused across calls; no per-corner allocation.
    resp: Vec<f32>,
    /// Scratch buffer used by the box-blur pass. Same size as `resp`.
    blur_scratch: Vec<f32>,
    side: usize,
}

impl RadonPeakRefiner {
    /// Build a refiner with pre-allocated scratch for the configured
    /// response-map size.
    pub fn new(cfg: RadonPeakConfig) -> Self {
        let side = cfg.side();
        Self {
            cfg,
            resp: vec![0.0; side * side],
            blur_scratch: vec![0.0; side * side],
            side,
        }
    }

    /// Access the configuration, chiefly for tests and introspection.
    #[inline]
    pub fn config(&self) -> &RadonPeakConfig {
        &self.cfg
    }

    /// Access the current response map (post-blur). Exposed for tests
    /// and debugging. Caller must not rely on the internal layout beyond
    /// row-major `side × side`.
    #[inline]
    pub fn response(&self) -> (&[f32], usize) {
        (&self.resp, self.side)
    }

    /// Compute the localized Radon response at a continuous image
    /// coordinate. Returns `(max_ray − min_ray)²`.
    #[inline]
    fn response_at(&self, img: &ImageView<'_>, x: f32, y: f32, step: f32) -> f32 {
        // Mirror `image_upsample_clamped()` so ray length stays consistent
        // with the response-map grid when a config with `image_upsample == 0`
        // slips through serde (valid `u32`, clamped everywhere else).
        let samples_per_side =
            (self.cfg.ray_radius as i32) * (self.cfg.image_upsample_clamped() as i32);
        let samples_per_side = samples_per_side.max(1);
        let mut max_r = f32::NEG_INFINITY;
        let mut min_r = f32::INFINITY;
        for a in 0..ANGLES {
            let cx = DIR_COS[a];
            let sy = DIR_SIN[a];
            let mut sum = 0.0f32;
            for k in -samples_per_side..=samples_per_side {
                let kf = k as f32 * step;
                sum += img.sample_bilinear(x + kf * cx, y + kf * sy);
            }
            if sum > max_r {
                max_r = sum;
            }
            if sum < min_r {
                min_r = sum;
            }
        }
        let d = max_r - min_r;
        d * d
    }
}

impl CornerRefiner for RadonPeakRefiner {
    #[inline]
    fn radius(&self) -> i32 {
        // The caller must preserve enough margin so rays don't leave the
        // image. The outermost response sample sits `patch_radius` px
        // from the seed and its rays extend another `ray_radius` px.
        self.cfg.patch_radius as i32 + self.cfg.ray_radius as i32
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
        let patch_r = self.cfg.patch_radius as i32;
        let ray_r = self.cfg.ray_radius as i32;

        if !img.supports_patch(cx, cy, patch_r + ray_r) {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: 0.0,
                status: RefineStatus::OutOfBounds,
            };
        }

        let upsample = self.cfg.image_upsample_clamped() as i32;
        let step = 1.0f32 / upsample as f32;
        let side = self.side as i32;
        debug_assert_eq!(side, 2 * patch_r * upsample + 1);
        let center_i = patch_r * upsample;

        for iy in 0..side {
            let dy = (iy - center_i) as f32 * step;
            for ix in 0..side {
                let dx = (ix - center_i) as f32 * step;
                let r = self.response_at(&img, cx as f32 + dx, cy as f32 + dy, step);
                self.resp[(iy as usize) * self.side + ix as usize] = r;
            }
        }

        box_blur_inplace(
            &mut self.resp,
            &mut self.blur_scratch,
            self.side,
            self.side,
            self.cfg.response_blur_radius as usize,
        );

        let mut best = f32::NEG_INFINITY;
        let mut best_ix = 0i32;
        let mut best_iy = 0i32;
        for iy in 0..side {
            for ix in 0..side {
                let r = self.resp[(iy as usize) * self.side + ix as usize];
                if r > best {
                    best = r;
                    best_ix = ix;
                    best_iy = iy;
                }
            }
        }

        if best < self.cfg.min_response {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: best,
                status: RefineStatus::Rejected,
            };
        }

        // Border argmax: no valid parabolic neighborhood.
        if best_ix == 0 || best_iy == 0 || best_ix == side - 1 || best_iy == side - 1 {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: best,
                status: RefineStatus::IllConditioned,
            };
        }

        let at = |rx: i32, ry: i32, resp: &[f32], side: usize| -> f32 {
            resp[(ry as usize) * side + rx as usize]
        };
        let r_c = at(best_ix, best_iy, &self.resp, self.side);
        let r_xm = at(best_ix - 1, best_iy, &self.resp, self.side);
        let r_xp = at(best_ix + 1, best_iy, &self.resp, self.side);
        let r_ym = at(best_ix, best_iy - 1, &self.resp, self.side);
        let r_yp = at(best_ix, best_iy + 1, &self.resp, self.side);

        let fx = fit_peak_frac(r_xm, r_c, r_xp, self.cfg.peak_fit);
        let fy = fit_peak_frac(r_ym, r_c, r_yp, self.cfg.peak_fit);

        let dx = (best_ix - center_i) as f32 * step + fx * step;
        let dy = (best_iy - center_i) as f32 * step + fy * step;

        let max_off = self.cfg.max_offset.min(patch_r as f32 + 0.5);
        if dx.abs() > max_off || dy.abs() > max_off {
            return RefineResult {
                x: seed_xy[0],
                y: seed_xy[1],
                score: best,
                status: RefineStatus::Rejected,
            };
        }

        let score = best.sqrt();
        RefineResult::accepted([cx as f32 + dx, cy as f32 + dy], score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a proper periodic chessboard with a corner anchored at
    /// `offset`, anti-aliased via 8× supersampling so that subpixel
    /// offsets actually appear in the output. Each output pixel is the
    /// average of an 8×8 grid of sub-samples at the underlying hard
    /// chessboard pattern. Then apply a mild 3×3 box blur (simulates
    /// camera blur; without it the edges are too sharp for the 4-angle
    /// Radon peak to be well-behaved).
    ///
    /// This replaces the previous nearest-neighbour rasterisation,
    /// which quantised the apparent corner position to half-pixels and
    /// made sub-0.1 px assertions unphysical.
    fn synthetic_chessboard(
        size: usize,
        cell: usize,
        offset: (f32, f32),
        dark: u8,
        bright: u8,
    ) -> Vec<u8> {
        const SUPER: usize = 8;
        let (ox, oy) = offset;
        let c = cell as f32;
        let dark_f = dark as f32;
        let bright_f = bright as f32;
        let inv_super2 = 1.0 / (SUPER * SUPER) as f32;
        let mut img = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut acc = 0.0f32;
                for sy in 0..SUPER {
                    let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                    let cy = ((yf - oy) / c).floor() as i32;
                    for sx in 0..SUPER {
                        let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                        let cx = ((xf - ox) / c).floor() as i32;
                        let dark_cell = (cx + cy).rem_euclid(2) == 0;
                        acc += if dark_cell { dark_f } else { bright_f };
                    }
                }
                img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
            }
        }
        // Mild 3×3 blur (camera PSF simulation).
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

    /// Apply a separable Gaussian blur in-place. Used by robustness tests.
    fn gaussian_blur(img: &mut [u8], size: usize, sigma: f32) {
        let radius = ((3.0 * sigma).ceil() as usize).max(1);
        let klen = 2 * radius + 1;
        let mut kernel = vec![0f32; klen];
        let mut sum = 0f32;
        for (i, k) in kernel.iter_mut().enumerate() {
            let x = i as f32 - radius as f32;
            *k = (-(x * x) / (2.0 * sigma * sigma)).exp();
            sum += *k;
        }
        for k in kernel.iter_mut() {
            *k /= sum;
        }
        let mut tmp = vec![0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut acc = 0f32;
                for (ki, &k) in kernel.iter().enumerate() {
                    let sx =
                        (x as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                    acc += img[y * size + sx] as f32 * k;
                }
                tmp[y * size + x] = acc;
            }
        }
        for y in 0..size {
            for x in 0..size {
                let mut acc = 0f32;
                for (ki, &k) in kernel.iter().enumerate() {
                    let sy =
                        (y as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                    acc += tmp[sy * size + x] * k;
                }
                img[y * size + x] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    /// Deterministic additive Gaussian noise via PCG-style LCG + Box-Muller.
    fn add_gaussian_noise(img: &mut [u8], sigma: f32, seed: u64) {
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        let mut next_u32 = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut uniform = || -> f32 { (next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0) };
        for px in img.iter_mut() {
            let u1 = uniform();
            let u2 = uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
            let v = *px as f32 + z * sigma;
            *px = v.round().clamp(0.0, 255.0) as u8;
        }
    }

    fn error([x, y]: [f32; 2], [tx, ty]: [f32; 2]) -> f32 {
        ((x - tx).powi(2) + (y - ty).powi(2)).sqrt()
    }

    /// Image size and cell size chosen so ~5×5 cells are visible and a
    /// 4-physical-px ray (at any sample in the response window) crosses
    /// into neighbouring cells without straying past the image border.
    const TEST_SIZE: usize = 35;
    const TEST_CELL: usize = 6;
    const TEST_CENTER: f32 = 17.0;

    #[test]
    fn compare_clean_accuracy_vs_saddlepoint() {
        // Contract: RadonPeak must be competitive with (ideally better
        // than) SaddlePoint on clean inputs now that the paper's full
        // pipeline is implemented.
        use crate::refine::{SaddlePointConfig, SaddlePointRefiner};
        let truth = (TEST_CENTER + 0.35, TEST_CENTER + 0.8);
        let img = synthetic_chessboard(TEST_SIZE, TEST_CELL, truth, 30, 230);
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
        let seed = [truth.0.round(), truth.1.round()];
        let ctx = RefineContext {
            image: Some(view),
            response: None,
        };

        let mut rp = RadonPeakRefiner::new(RadonPeakConfig::default());
        let radon = rp.refine(seed, ctx);
        assert_eq!(radon.status, RefineStatus::Accepted);

        let mut sp = SaddlePointRefiner::new(SaddlePointConfig::default());
        let saddle = sp.refine(seed, ctx);

        let radon_err = error([radon.x, radon.y], [truth.0, truth.1]);
        let saddle_err = if saddle.status == RefineStatus::Accepted {
            error([saddle.x, saddle.y], [truth.0, truth.1])
        } else {
            f32::NAN
        };
        eprintln!(
            "clean-data accuracy: radon_peak={:.4} saddle_point={:.4}",
            radon_err, saddle_err
        );
        assert!(radon_err < 0.1, "radon_err {radon_err} exceeds 0.1 px");
    }

    #[test]
    fn recovers_ideal_subpixel_offset() {
        let truth = (TEST_CENTER + 0.35, TEST_CENTER + 0.8);
        let img = synthetic_chessboard(TEST_SIZE, TEST_CELL, truth, 30, 230);
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
        let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
        let res = refiner.refine(
            [truth.0.round(), truth.1.round()],
            RefineContext {
                image: Some(view),
                response: None,
            },
        );
        assert_eq!(res.status, RefineStatus::Accepted);
        let err = error([res.x, res.y], [truth.0, truth.1]);
        assert!(err < 0.1, "err={} res=({},{})", err, res.x, res.y);
    }

    #[test]
    fn subpixel_sweep_mean_error_bounded() {
        let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
        let mut total = 0.0f32;
        let mut worst = 0.0f32;
        let mut count = 0.0f32;
        for k in 0..8 {
            let off = TEST_CENTER + (k as f32) / 8.0;
            let img = synthetic_chessboard(TEST_SIZE, TEST_CELL, (off, off), 30, 230);
            let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
            let res = refiner.refine(
                [off.round(), off.round()],
                RefineContext {
                    image: Some(view),
                    response: None,
                },
            );
            assert_eq!(res.status, RefineStatus::Accepted, "k={}", k);
            let err = error([res.x, res.y], [off, off]);
            total += err;
            worst = worst.max(err);
            count += 1.0;
        }
        let mean = total / count;
        assert!(
            mean < 0.1 && worst < 0.2,
            "mean {mean} worst {worst} over 8 offsets"
        );
    }

    #[test]
    fn gaussian_fit_beats_parabolic_on_clean_inputs() {
        // Sanity check for the Gaussian peak-fit mode: it should be at
        // least as accurate as parabolic on a sweep of subpixel offsets.
        let mut gauss_total = 0.0f32;
        let mut parab_total = 0.0f32;
        let cfg_gauss = RadonPeakConfig::default();
        let cfg_parab = RadonPeakConfig {
            peak_fit: PeakFitMode::Parabolic,
            ..cfg_gauss
        };
        let mut gauss = RadonPeakRefiner::new(cfg_gauss);
        let mut parab = RadonPeakRefiner::new(cfg_parab);
        for k in 0..8 {
            let off = TEST_CENTER + (k as f32) / 8.0;
            let img = synthetic_chessboard(TEST_SIZE, TEST_CELL, (off, off), 30, 230);
            let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
            let seed = [off.round(), off.round()];
            let ctx = RefineContext {
                image: Some(view),
                response: None,
            };
            let rg = gauss.refine(seed, ctx);
            let rp = parab.refine(seed, ctx);
            assert_eq!(rg.status, RefineStatus::Accepted);
            assert_eq!(rp.status, RefineStatus::Accepted);
            gauss_total += error([rg.x, rg.y], [off, off]);
            parab_total += error([rp.x, rp.y], [off, off]);
        }
        eprintln!("gauss_mean={} parab_mean={}", gauss_total, parab_total);
        assert!(
            gauss_total <= parab_total + 1e-3,
            "Gaussian fit regressed vs parabolic: {} > {}",
            gauss_total,
            parab_total
        );
    }

    #[test]
    fn refined_beats_seed_under_blur() {
        let truth = (TEST_CENTER + 0.3, TEST_CENTER + 0.7);
        for sigma in [1.0f32, 2.0] {
            let mut img = synthetic_chessboard(TEST_SIZE, TEST_CELL, truth, 30, 230);
            gaussian_blur(&mut img, TEST_SIZE, sigma);
            let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
            let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
            let seed = [truth.0.round(), truth.1.round()];
            let res = refiner.refine(
                seed,
                RefineContext {
                    image: Some(view),
                    response: None,
                },
            );
            assert_eq!(res.status, RefineStatus::Accepted, "sigma={}", sigma);
            let seed_err = error(seed, [truth.0, truth.1]);
            let ref_err = error([res.x, res.y], [truth.0, truth.1]);
            assert!(
                ref_err <= seed_err,
                "sigma={}: refined {} not better than seed {}",
                sigma,
                ref_err,
                seed_err
            );
        }
    }

    #[test]
    fn refined_beats_seed_under_moderate_noise() {
        let truth = (TEST_CENTER + 0.3, TEST_CENTER + 0.7);
        let mut img = synthetic_chessboard(TEST_SIZE, TEST_CELL, truth, 30, 230);
        add_gaussian_noise(&mut img, 5.0, 0xC0FFEE);
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
        let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
        let seed = [truth.0.round(), truth.1.round()];
        let res = refiner.refine(
            seed,
            RefineContext {
                image: Some(view),
                response: None,
            },
        );
        assert_eq!(res.status, RefineStatus::Accepted);
        let seed_err = error(seed, [truth.0, truth.1]);
        let ref_err = error([res.x, res.y], [truth.0, truth.1]);
        assert!(
            ref_err <= seed_err,
            "refined {} not better than seed {}",
            ref_err,
            seed_err
        );
    }

    #[test]
    fn rejects_flat_region() {
        let flat = vec![128u8; TEST_SIZE * TEST_SIZE];
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &flat).unwrap();
        let cfg = RadonPeakConfig {
            min_response: 1.0,
            ..RadonPeakConfig::default()
        };
        let mut refiner = RadonPeakRefiner::new(cfg);
        let res = refiner.refine(
            [TEST_CENTER, TEST_CENTER],
            RefineContext {
                image: Some(view),
                response: None,
            },
        );
        assert_ne!(res.status, RefineStatus::Accepted);
    }

    #[test]
    fn out_of_bounds_near_image_edge() {
        let img = synthetic_chessboard(TEST_SIZE, TEST_CELL, (2.0, 2.0), 30, 230);
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
        let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
        let res = refiner.refine(
            [1.0, 1.0],
            RefineContext {
                image: Some(view),
                response: None,
            },
        );
        assert_eq!(res.status, RefineStatus::OutOfBounds);
    }

    #[test]
    fn deterministic_repeated_calls() {
        let img = synthetic_chessboard(
            TEST_SIZE,
            TEST_CELL,
            (TEST_CENTER + 0.2, TEST_CENTER + 0.6),
            30,
            230,
        );
        let view = ImageView::from_u8_slice(TEST_SIZE, TEST_SIZE, &img).unwrap();
        let mut refiner = RadonPeakRefiner::new(RadonPeakConfig::default());
        let ctx = RefineContext {
            image: Some(view),
            response: None,
        };
        let a = refiner.refine([TEST_CENTER, TEST_CENTER + 1.0], ctx);
        let b = refiner.refine([TEST_CENTER, TEST_CENTER + 1.0], ctx);
        assert_eq!(a.status, b.status);
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
        assert_eq!(a.score.to_bits(), b.score.to_bits());
    }

    #[test]
    fn config_round_trips_through_json() {
        let cfg = RadonPeakConfig {
            ray_radius: 5,
            patch_radius: 3,
            image_upsample: 2,
            response_blur_radius: 2,
            peak_fit: PeakFitMode::Parabolic,
            min_response: 2.5,
            max_offset: 1.25,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RadonPeakConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn scratch_buffer_sized_correctly() {
        for up in [1u32, 2, 4] {
            let cfg = RadonPeakConfig {
                patch_radius: 3,
                image_upsample: up,
                ..RadonPeakConfig::default()
            };
            let refiner = RadonPeakRefiner::new(cfg);
            let expected_side = 2 * 3 * up as usize + 1;
            assert_eq!(refiner.side, expected_side);
            assert_eq!(refiner.resp.len(), expected_side * expected_side);
            assert_eq!(refiner.blur_scratch.len(), expected_side * expected_side);
        }
    }
}
