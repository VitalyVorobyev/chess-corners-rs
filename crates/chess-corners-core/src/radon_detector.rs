//! Whole-image Duda-Frese Radon detector.
//!
//! An alternative to the ChESS detector in [`response`](crate::response)
//! for frames where ChESS's 16-sample ring fails — heavy motion blur,
//! strong defocus, low-contrast scenes, or cells smaller than
//! `~2·ring_radius`. The detector computes a dense 4-angle localized
//! Radon response `R(x, y) = (max_α S_α − min_α S_α)²` using
//! summed-area tables for O(1)-per-pixel ray sums, then applies the
//! same peak-fit pipeline as [`RadonPeakRefiner`](crate::refine_radon):
//! threshold / NMS / box-blur / 3-point Gaussian fit in x and y.
//!
//! # Pipeline
//!
//! ```text
//!  u8 ─► [optional 2× bilinear upsample]
//!                 │
//!                 ▼
//!        [4 summed-area tables: row / col / +diag / −diag]
//!                 │
//!                 ▼
//!        [pointwise Radon response = (max − min)²]
//!                 │
//!                 ▼
//!        [box blur, threshold, NMS, min-cluster]
//!                 │
//!                 ▼
//!        [3-point Gaussian peak fit]
//!                 │
//!                 ▼
//!         Corner list in input-pixel coordinates
//! ```
//!
//! Shares [`DIR_COS`](crate::radon::DIR_COS), [`DIR_SIN`](crate::radon::DIR_SIN),
//! [`PeakFitMode`], [`fit_peak_frac`], and [`box_blur_inplace`] with the
//! refiner so both paths agree on the underlying primitives.
//!
//! # SAT element type
//!
//! Summed-area-tables default to `i64`, which is always safe. Enable
//! the `radon-sat-u32` crate feature to switch to `u32`, which halves
//! SAT memory and widens SIMD lanes at the cost of a ~16 MP image-size
//! cap (`255 · W · H ≤ u32::MAX`).

use serde::{Deserialize, Serialize};

use crate::descriptor::Corner;
use crate::detect::{count_positive_neighbors, is_local_max};
use crate::radon::{box_blur_inplace, fit_peak_frac, PeakFitMode};
use crate::ResponseMap;

/// Summed-area-table element type. Gated by the `radon-sat-u32`
/// crate feature.
#[cfg(not(feature = "radon-sat-u32"))]
pub type SatElem = i64;

/// Summed-area-table element type (feature `radon-sat-u32`).
#[cfg(feature = "radon-sat-u32")]
pub type SatElem = u32;

/// Configuration for the whole-image Radon detector.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RadonDetectorParams {
    /// Half-length of each ray in **working-resolution** pixels (i.e.
    /// post-upsample). The ray has `2·ray_radius + 1` samples. Paper
    /// default at `image_upsample=2` is 4 working pixels ⇒ 2 physical.
    pub ray_radius: u32,
    /// Image-level supersampling factor. `1` operates on the input
    /// pixel grid; `2` bilinearly upsamples first (paper default).
    /// M1 supports the set `{1, 2}`; values `>= 3` are clamped to `2`
    /// (see [`MAX_IMAGE_UPSAMPLE`]). Higher factors are future work.
    pub image_upsample: u32,
    /// Half-size of the box blur applied to the response map. `0`
    /// disables blurring; `1` yields a 3×3 box.
    pub response_blur_radius: u32,
    /// Peak-fit mode for the 3-point subpixel refinement. Gaussian
    /// (log-space) is more stable at near-plateau peaks.
    pub peak_fit: PeakFitMode,
    /// Relative response threshold as a fraction of the map's max
    /// value. Used when `threshold_abs` is `None`.
    pub threshold_rel: f32,
    /// Absolute response threshold. Overrides `threshold_rel` when set.
    /// The paper's `(max−min)²` response is always ≥ 0, so the strict
    /// inequality `R > 0` that ChESS uses is not by itself selective
    /// enough — use a positive absolute floor in practice.
    pub threshold_abs: Option<f32>,
    /// Non-maximum-suppression half-radius (in **working-resolution**
    /// pixels).
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbours in the NMS window
    /// required to accept a peak. Rejects isolated noise.
    pub min_cluster_size: u32,
}

impl Default for RadonDetectorParams {
    fn default() -> Self {
        Self {
            ray_radius: 4,
            image_upsample: 2,
            response_blur_radius: 1,
            peak_fit: PeakFitMode::Gaussian,
            // (max−min)² is always non-negative, so the ChESS `R > 0`
            // contract doesn't carry a useful signal here. Default to a
            // 1 % relative floor; callers who want "any peak" can set
            // `threshold_abs = Some(0.0)` explicitly.
            threshold_rel: 0.01,
            threshold_abs: None,
            nms_radius: 4,
            min_cluster_size: 2,
        }
    }
}

/// Supported image-upsample factors in M1: `{1, 2}`. Anything higher
/// would need a different upsampler; values `>= 3` are clamped to `2`
/// at the entry points rather than silently producing mismatched
/// buffer sizes downstream.
pub const MAX_IMAGE_UPSAMPLE: u32 = 2;

impl RadonDetectorParams {
    /// Clamp `image_upsample` into the supported set `{1, 2}`.
    /// Values outside that range are silently clamped — callers can
    /// detect truncation by comparing against [`MAX_IMAGE_UPSAMPLE`].
    #[inline]
    fn image_upsample_clamped(&self) -> u32 {
        self.image_upsample.clamp(1, MAX_IMAGE_UPSAMPLE)
    }

    #[inline]
    fn ray_radius_clamped(&self) -> u32 {
        self.ray_radius.max(1)
    }
}

/// Reusable scratch for the whole-image Radon detector. Holds the
/// upsampled image buffer, the four summed-area tables, the response
/// map, and the box-blur scratch. All buffers grow on demand and are
/// reused across frames — same pattern as `PyramidBuffers`.
#[derive(Debug, Default)]
pub struct RadonBuffers {
    /// Upsampled image at `working_w × working_h`. Empty when
    /// `image_upsample == 1` (the caller's image is used directly).
    upsampled: Vec<u8>,
    /// Working resolution (input dims × `image_upsample`).
    working_w: usize,
    working_h: usize,
    /// Row-wise prefix sums: `row_cumsum[y][x] = Σ img[y][0..=x]`.
    row_cumsum: Vec<SatElem>,
    /// Column-wise prefix sums.
    col_cumsum: Vec<SatElem>,
    /// NW-SE diagonal prefix sums: value at `(y, x)` is the sum along
    /// `(x-k, y-k)` for `k ∈ [0, min(x, y)]`.
    diag_pos_cumsum: Vec<SatElem>,
    /// NE-SW diagonal prefix sums: value at `(y, x)` is the sum along
    /// `(x+k, y-k)` for `k ∈ [0, min(W-1-x, y)]`.
    diag_neg_cumsum: Vec<SatElem>,
    /// Dense response map at working resolution (row-major).
    response: Vec<f32>,
    /// Temporary storage for the box blur. Same size as `response`.
    blur_scratch: Vec<f32>,
}

impl RadonBuffers {
    /// Create an empty set of buffers. They grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure all buffers are sized for a `(input_w × upsample,
    /// input_h × upsample)` working resolution. Also re-sizes the
    /// upsampled-image buffer when needed.
    fn ensure_capacity(&mut self, input_w: usize, input_h: usize, upsample: u32) {
        let up = upsample.max(1) as usize;
        let ww = input_w * up;
        let wh = input_h * up;
        let n = ww * wh;
        self.working_w = ww;
        self.working_h = wh;
        if up > 1 {
            self.upsampled.resize(n, 0);
        } else {
            self.upsampled.clear();
        }
        self.row_cumsum.resize(n, SatElem::default());
        self.col_cumsum.resize(n, SatElem::default());
        self.diag_pos_cumsum.resize(n, SatElem::default());
        self.diag_neg_cumsum.resize(n, SatElem::default());
        self.response.resize(n, 0.0);
        self.blur_scratch.resize(n, 0.0);
    }
}

/// Corner-aligned bilinear 2× upsample. Output pixel `(iu, iv)`
/// samples source coordinate `(iu/2, iv/2)` — so output `(0, 0)`
/// aligns with source `(0, 0)`, and the inverse transform from
/// working to input is simply `x_in = x_work / upsample`. `out`
/// must already be sized `(2W × 2H)`.
fn upsample_bilinear_2x(src: &[u8], w: usize, h: usize, out: &mut [u8]) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(out.len(), 4 * w * h);
    let ww = 2 * w;
    let wh = 2 * h;
    for iy in 0..wh {
        let sy = iy as f32 * 0.5;
        let y0f = sy.floor();
        let y0 = (y0f as isize).max(0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let ty = (sy - y0f).clamp(0.0, 1.0);
        for ix in 0..ww {
            let sx = ix as f32 * 0.5;
            let x0f = sx.floor();
            let x0 = (x0f as isize).max(0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let tx = (sx - x0f).clamp(0.0, 1.0);
            let i00 = src[y0 * w + x0] as f32;
            let i10 = src[y0 * w + x1] as f32;
            let i01 = src[y1 * w + x0] as f32;
            let i11 = src[y1 * w + x1] as f32;
            let a = i00 + (i10 - i00) * tx;
            let b = i01 + (i11 - i01) * tx;
            let v = a + (b - a) * ty;
            out[iy * ww + ix] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Build the four cumulative-sum tables from a working-resolution u8
/// image slice. Each table has exactly `w·h` elements in row-major
/// order.
fn build_cumsums(
    img: &[u8],
    w: usize,
    h: usize,
    row_cumsum: &mut [SatElem],
    col_cumsum: &mut [SatElem],
    diag_pos_cumsum: &mut [SatElem],
    diag_neg_cumsum: &mut [SatElem],
) {
    debug_assert_eq!(img.len(), w * h);
    debug_assert_eq!(row_cumsum.len(), w * h);
    debug_assert_eq!(col_cumsum.len(), w * h);
    debug_assert_eq!(diag_pos_cumsum.len(), w * h);
    debug_assert_eq!(diag_neg_cumsum.len(), w * h);

    // Row-wise prefix sums.
    for y in 0..h {
        let mut acc: SatElem = SatElem::default();
        for x in 0..w {
            acc += SatElem::from(img[y * w + x]);
            row_cumsum[y * w + x] = acc;
        }
    }

    // Column-wise prefix sums.
    for x in 0..w {
        let mut acc: SatElem = SatElem::default();
        for y in 0..h {
            acc += SatElem::from(img[y * w + x]);
            col_cumsum[y * w + x] = acc;
        }
    }

    // NW-SE diagonal prefix sums: diag_pos[y][x] = I[y][x] + diag_pos[y-1][x-1].
    for y in 0..h {
        for x in 0..w {
            let prev = if y > 0 && x > 0 {
                diag_pos_cumsum[(y - 1) * w + (x - 1)]
            } else {
                SatElem::default()
            };
            diag_pos_cumsum[y * w + x] = prev + SatElem::from(img[y * w + x]);
        }
    }

    // NE-SW diagonal prefix sums: diag_neg[y][x] = I[y][x] + diag_neg[y-1][x+1].
    for y in 0..h {
        for x in 0..w {
            let prev = if y > 0 && x + 1 < w {
                diag_neg_cumsum[(y - 1) * w + (x + 1)]
            } else {
                SatElem::default()
            };
            diag_neg_cumsum[y * w + x] = prev + SatElem::from(img[y * w + x]);
        }
    }
}

/// Bundle of cumsum tables + geometry, passed to response kernels.
struct Cumsums<'a> {
    row: &'a [SatElem],
    col: &'a [SatElem],
    diag_pos: &'a [SatElem],
    diag_neg: &'a [SatElem],
    w: usize,
    h: usize,
}

/// Compute the dense Radon response in the interior `[r..w-r) ×
/// [r..h-r)`. Border pixels (where ray samples would leave the image)
/// receive 0.0 so thresholding / NMS naturally rejects them.
fn compute_response(cs: &Cumsums<'_>, ray_radius: usize, out: &mut [f32]) {
    let w = cs.w;
    let h = cs.h;
    debug_assert_eq!(out.len(), w * h);
    if w <= 2 * ray_radius || h <= 2 * ray_radius {
        out.fill(0.0);
        return;
    }
    let r = ray_radius;
    // Zero the border rows / cols we won't compute.
    for y in 0..h {
        for x in 0..w {
            let at_border = x < r || y < r || x + r >= w || y + r >= h;
            if at_border {
                out[y * w + x] = 0.0;
            }
        }
    }

    for y in r..(h - r) {
        for x in r..(w - r) {
            // Horizontal ray: row_cumsum[y][x+r] - row_cumsum[y][x-r-1].
            let s_h_hi = cs.row[y * w + (x + r)];
            let s_h_lo = if x > r {
                cs.row[y * w + (x - r - 1)]
            } else {
                SatElem::default()
            };
            let s_h = s_h_hi - s_h_lo;

            // Vertical ray.
            let s_v_hi = cs.col[(y + r) * w + x];
            let s_v_lo = if y > r {
                cs.col[(y - r - 1) * w + x]
            } else {
                SatElem::default()
            };
            let s_v = s_v_hi - s_v_lo;

            // NW-SE diagonal ray.
            let s_d1_hi = cs.diag_pos[(y + r) * w + (x + r)];
            let s_d1_lo = if x > r && y > r {
                cs.diag_pos[(y - r - 1) * w + (x - r - 1)]
            } else {
                SatElem::default()
            };
            let s_d1 = s_d1_hi - s_d1_lo;

            // NE-SW diagonal ray.
            let s_d2_hi = cs.diag_neg[(y + r) * w + (x - r)];
            let s_d2_lo = if y > r && x + r + 1 < w {
                cs.diag_neg[(y - r - 1) * w + (x + r + 1)]
            } else {
                SatElem::default()
            };
            let s_d2 = s_d2_hi - s_d2_lo;

            // (max − min)², cast to f32 for the peak-fit pipeline.
            let s = [s_h, s_v, s_d1, s_d2];
            let (mut mx, mut mn) = (s[0], s[0]);
            for &v in &s[1..] {
                if v > mx {
                    mx = v;
                }
                if v < mn {
                    mn = v;
                }
            }
            let d = sat_to_f32(mx - mn);
            out[y * w + x] = d * d;
        }
    }
}

#[inline]
fn sat_to_f32(v: SatElem) -> f32 {
    v as f32
}

/// Compute the dense Radon response into `buffers.response` and return
/// a read-only `ResponseMap` view at **working resolution** (i.e.
/// `input_dim × image_upsample`).
///
/// The returned response is in `buffers`'s backing storage — do not
/// mix with the borrow of `buffers` that follows.
pub fn radon_response_u8<'a>(
    img: &[u8],
    w: usize,
    h: usize,
    params: &RadonDetectorParams,
    buffers: &'a mut RadonBuffers,
) -> RadonResponseView<'a> {
    assert_eq!(img.len(), w * h, "img len must equal w*h");
    let up = params.image_upsample_clamped();
    buffers.ensure_capacity(w, h, up);
    let ww = buffers.working_w;
    let wh = buffers.working_h;

    // Under `radon-sat-u32` the SAT accumulator is `u32`, and the
    // largest prefix-sum value is bounded by `255 * ww * wh`. Beyond
    // that the `build_cumsums` additions wrap silently in release, so
    // we reject the input up-front rather than corrupting the response.
    // The default `i64` accumulator has no practical ceiling on any
    // image that fits in host memory; this check is a no-op there.
    #[cfg(feature = "radon-sat-u32")]
    {
        let pixels = (ww as u64) * (wh as u64);
        let max_sum = 255u64.checked_mul(pixels);
        assert!(
            matches!(max_sum, Some(v) if v <= u32::MAX as u64),
            "radon-sat-u32: 255*W*H ({}*{}) exceeds u32::MAX; \
             either rebuild without the radon-sat-u32 feature or \
             downsample the input",
            ww,
            wh,
        );
    }

    // Produce the working-resolution u8 image.
    let working_img: &[u8] = if up > 1 {
        upsample_bilinear_2x_if_needed(img, w, h, up, &mut buffers.upsampled);
        &buffers.upsampled
    } else {
        img
    };

    build_cumsums(
        working_img,
        ww,
        wh,
        &mut buffers.row_cumsum,
        &mut buffers.col_cumsum,
        &mut buffers.diag_pos_cumsum,
        &mut buffers.diag_neg_cumsum,
    );

    let cs = Cumsums {
        row: &buffers.row_cumsum,
        col: &buffers.col_cumsum,
        diag_pos: &buffers.diag_pos_cumsum,
        diag_neg: &buffers.diag_neg_cumsum,
        w: ww,
        h: wh,
    };
    compute_response(
        &cs,
        params.ray_radius_clamped() as usize,
        &mut buffers.response,
    );

    box_blur_inplace(
        &mut buffers.response,
        &mut buffers.blur_scratch,
        ww,
        params.response_blur_radius as usize,
    );
    // Response row-stride: width. Treat the row-major response as a
    // square-ish ResponseMap borrowed view.
    RadonResponseView {
        data: &buffers.response,
        w: ww,
        h: wh,
    }
}

/// Borrow of the dense working-resolution response map. Cheaply
/// convertible to a [`ResponseMap`] via [`Self::to_response_map`]
/// when ownership is required (e.g. for the classic
/// [`detect_corners_from_response`](crate::detect::detect_corners_from_response)).
#[derive(Debug)]
pub struct RadonResponseView<'a> {
    data: &'a [f32],
    w: usize,
    h: usize,
}

impl<'a> RadonResponseView<'a> {
    /// Width at working resolution.
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    /// Height at working resolution.
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    /// Raw response data, row-major.
    #[inline]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Response value at a working-resolution integer coordinate.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    /// Materialize into an owned `ResponseMap` (copies the data).
    pub fn to_response_map(&self) -> ResponseMap {
        ResponseMap::new(self.w, self.h, self.data.to_vec())
    }
}

#[inline]
fn upsample_bilinear_2x_if_needed(img: &[u8], w: usize, h: usize, up: u32, out: &mut Vec<u8>) {
    // Callers must pre-clamp via `RadonDetectorParams::image_upsample_clamped()`,
    // so `up` is always in `{1, 2}` here. Assert in debug for sanity.
    debug_assert_eq!(up, 2, "image_upsample must be 1 or 2 in M1");
    let _ = up;
    out.resize(4 * w * h, 0);
    upsample_bilinear_2x(img, w, h, out);
}

/// Detect corners from the working-resolution response map produced
/// by [`radon_response_u8`]. Applies threshold / NMS / min-cluster
/// rejection at working resolution, then a 3-point peak fit on the
/// blurred response to get subpixel offsets. Output coordinates are
/// divided by `image_upsample` so they sit in the **input pixel
/// frame**.
pub fn detect_corners_from_radon(
    resp: &RadonResponseView<'_>,
    params: &RadonDetectorParams,
) -> Vec<Corner> {
    let w = resp.w;
    let h = resp.h;
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut max_r = f32::NEG_INFINITY;
    for &v in resp.data {
        if v > max_r {
            max_r = v;
        }
    }
    if !max_r.is_finite() {
        return Vec::new();
    }

    let thr = params
        .threshold_abs
        .unwrap_or(params.threshold_rel * max_r)
        .max(0.0);

    let nms_r = params.nms_radius as i32;
    let ray_r = params.ray_radius_clamped() as i32;
    // We already zeroed pixels < ray_r from the border; leave 1 extra
    // pixel for the 3-point peak fit.
    let border = (ray_r + nms_r + 1).max(0) as usize;
    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    // Borrow the working-resolution response slice directly. The
    // shared NMS helpers are slice-based (see `detect::is_local_max`)
    // so we no longer allocate a full-frame `ResponseMap` clone in
    // the hot path — a noticeable win at `image_upsample=2` on HD
    // frames.
    let data = resp.data;
    let mut out = Vec::new();
    let inv_up = 1.0 / (params.image_upsample_clamped() as f32);

    #[inline(always)]
    fn at(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
        data[y * w + x]
    }

    for y in border..(h - border) {
        for x in border..(w - border) {
            let v = at(data, w, x, y);
            if v <= thr {
                continue;
            }
            if !is_local_max(data, w, h, x, y, nms_r, v) {
                continue;
            }
            if count_positive_neighbors(data, w, h, x, y, nms_r) < params.min_cluster_size {
                continue;
            }

            // 3-point peak fit. Neighbours are guaranteed in-bounds by
            // the border clamp.
            let r_c = v;
            let r_xm = at(data, w, x - 1, y);
            let r_xp = at(data, w, x + 1, y);
            let r_ym = at(data, w, x, y - 1);
            let r_yp = at(data, w, x, y + 1);
            let fx = fit_peak_frac(r_xm, r_c, r_xp, params.peak_fit);
            let fy = fit_peak_frac(r_ym, r_c, r_yp, params.peak_fit);

            // Working-resolution coordinates then back to input frame.
            let gx = (x as f32 + fx) * inv_up;
            let gy = (y as f32 + fy) * inv_up;
            out.push(Corner {
                x: gx,
                y: gy,
                strength: v,
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- cumsum correctness ----

    #[test]
    fn row_cumsum_matches_naive_sum() {
        let w = 5usize;
        let h = 3usize;
        let img: Vec<u8> = (0..(w * h) as u8).collect();
        let mut r = vec![SatElem::default(); w * h];
        let mut c = vec![SatElem::default(); w * h];
        let mut d1 = vec![SatElem::default(); w * h];
        let mut d2 = vec![SatElem::default(); w * h];
        build_cumsums(&img, w, h, &mut r, &mut c, &mut d1, &mut d2);
        for y in 0..h {
            let mut expected: SatElem = SatElem::default();
            for x in 0..w {
                expected += SatElem::from(img[y * w + x]);
                assert_eq!(r[y * w + x], expected);
            }
        }
    }

    #[test]
    fn col_cumsum_matches_naive_sum() {
        let w = 4usize;
        let h = 5usize;
        let img: Vec<u8> = (0..(w * h) as u8).collect();
        let mut r = vec![SatElem::default(); w * h];
        let mut c = vec![SatElem::default(); w * h];
        let mut d1 = vec![SatElem::default(); w * h];
        let mut d2 = vec![SatElem::default(); w * h];
        build_cumsums(&img, w, h, &mut r, &mut c, &mut d1, &mut d2);
        for x in 0..w {
            let mut expected: SatElem = SatElem::default();
            for y in 0..h {
                expected += SatElem::from(img[y * w + x]);
                assert_eq!(c[y * w + x], expected);
            }
        }
    }

    #[test]
    fn diag_pos_cumsum_matches_naive_sum() {
        let w = 4usize;
        let h = 4usize;
        let img: Vec<u8> = (0..(w * h) as u8).collect();
        let mut r = vec![SatElem::default(); w * h];
        let mut c = vec![SatElem::default(); w * h];
        let mut d1 = vec![SatElem::default(); w * h];
        let mut d2 = vec![SatElem::default(); w * h];
        build_cumsums(&img, w, h, &mut r, &mut c, &mut d1, &mut d2);
        for y in 0..h {
            for x in 0..w {
                // Naive: walk diagonal back to either left or top border.
                let mut expected: SatElem = SatElem::default();
                let mut xi = x as isize;
                let mut yi = y as isize;
                while xi >= 0 && yi >= 0 {
                    expected += SatElem::from(img[yi as usize * w + xi as usize]);
                    xi -= 1;
                    yi -= 1;
                }
                assert_eq!(d1[y * w + x], expected, "at ({},{})", x, y);
            }
        }
    }

    #[test]
    fn diag_neg_cumsum_matches_naive_sum() {
        let w = 4usize;
        let h = 4usize;
        let img: Vec<u8> = (0..(w * h) as u8).collect();
        let mut r = vec![SatElem::default(); w * h];
        let mut c = vec![SatElem::default(); w * h];
        let mut d1 = vec![SatElem::default(); w * h];
        let mut d2 = vec![SatElem::default(); w * h];
        build_cumsums(&img, w, h, &mut r, &mut c, &mut d1, &mut d2);
        for y in 0..h {
            for x in 0..w {
                // Naive: walk diagonal back to either right or top border.
                let mut expected: SatElem = SatElem::default();
                let mut xi = x as isize;
                let mut yi = y as isize;
                while xi < w as isize && yi >= 0 {
                    expected += SatElem::from(img[yi as usize * w + xi as usize]);
                    xi += 1;
                    yi -= 1;
                }
                assert_eq!(d2[y * w + x], expected, "at ({},{})", x, y);
            }
        }
    }

    #[test]
    fn ray_sums_via_sat_match_direct_sums() {
        let w = 15usize;
        let h = 15usize;
        let img: Vec<u8> = (0..(w * h)).map(|i| (i % 251) as u8).collect();
        let mut r = vec![SatElem::default(); w * h];
        let mut c = vec![SatElem::default(); w * h];
        let mut d1 = vec![SatElem::default(); w * h];
        let mut d2 = vec![SatElem::default(); w * h];
        build_cumsums(&img, w, h, &mut r, &mut c, &mut d1, &mut d2);

        let ray_r = 3usize;
        for y in ray_r..(h - ray_r) {
            for x in ray_r..(w - ray_r) {
                // Horizontal.
                let mut h_sum: SatElem = SatElem::default();
                for k in 0..=(2 * ray_r) {
                    let xx = x + k - ray_r;
                    h_sum += SatElem::from(img[y * w + xx]);
                }
                let h_hi = r[y * w + (x + ray_r)];
                let h_lo = if x > ray_r {
                    r[y * w + (x - ray_r - 1)]
                } else {
                    SatElem::default()
                };
                assert_eq!(h_hi - h_lo, h_sum, "horiz at ({},{})", x, y);

                // Vertical.
                let mut v_sum: SatElem = SatElem::default();
                for k in 0..=(2 * ray_r) {
                    let yy = y + k - ray_r;
                    v_sum += SatElem::from(img[yy * w + x]);
                }
                let v_hi = c[(y + ray_r) * w + x];
                let v_lo = if y > ray_r {
                    c[(y - ray_r - 1) * w + x]
                } else {
                    SatElem::default()
                };
                assert_eq!(v_hi - v_lo, v_sum, "vert at ({},{})", x, y);

                // NW-SE diagonal.
                let mut d1_sum: SatElem = SatElem::default();
                for k in 0..=(2 * ray_r) {
                    let xx = x + k - ray_r;
                    let yy = y + k - ray_r;
                    d1_sum += SatElem::from(img[yy * w + xx]);
                }
                let d1_hi = d1[(y + ray_r) * w + (x + ray_r)];
                let d1_lo = if x > ray_r && y > ray_r {
                    d1[(y - ray_r - 1) * w + (x - ray_r - 1)]
                } else {
                    SatElem::default()
                };
                assert_eq!(d1_hi - d1_lo, d1_sum, "diag+ at ({},{})", x, y);

                // NE-SW diagonal.
                let mut d2_sum: SatElem = SatElem::default();
                for k in 0..=(2 * ray_r) {
                    let xx = x + ray_r - k; // decreasing x
                    let yy = y + k - ray_r; // increasing y (from -R to +R)
                    d2_sum += SatElem::from(img[yy * w + xx]);
                }
                let d2_hi = d2[(y + ray_r) * w + (x - ray_r)];
                let d2_lo = if y > ray_r && x + ray_r + 1 < w {
                    d2[(y - ray_r - 1) * w + (x + ray_r + 1)]
                } else {
                    SatElem::default()
                };
                assert_eq!(d2_hi - d2_lo, d2_sum, "diag- at ({},{})", x, y);
            }
        }
    }

    // ---- end-to-end detector on synthetic chessboard ----

    fn synthetic_chessboard_aa(
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
        img
    }

    #[test]
    fn detector_recovers_interior_corner_without_seed() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        // Corner near the center of the image; the offset chooses where
        // the cell grid anchors.
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);

        let params = RadonDetectorParams {
            image_upsample: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_corners_from_radon(&resp, &params);

        assert!(
            !corners.is_empty(),
            "expected at least one corner, got none"
        );

        // There is a lattice of corners; find the one closest to the
        // cell junction near the image center.
        // Cell boundaries lie at (offset.x + k·CELL, offset.y + m·CELL)
        // for integer k, m. Closest to (SIZE/2, SIZE/2) = (32.5, 32.5):
        let expected_x =
            offset.0 + (((SIZE as f32 / 2.0 - offset.0) / CELL as f32).round() * CELL as f32);
        let expected_y =
            offset.1 + (((SIZE as f32 / 2.0 - offset.1) / CELL as f32).round() * CELL as f32);

        let (best, err) = corners
            .iter()
            .map(|c| {
                let dx = c.x - expected_x;
                let dy = c.y - expected_y;
                (c, (dx * dx + dy * dy).sqrt())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty");
        assert!(
            err < 0.2,
            "closest corner err={:.4} at ({:.3}, {:.3}), expected ({:.3}, {:.3}); {} corners found",
            err,
            best.x,
            best.y,
            expected_x,
            expected_y,
            corners.len(),
        );
    }

    #[test]
    fn detector_upsample_1_still_produces_corners() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 1,
            ray_radius: 2,
            nms_radius: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_corners_from_radon(&resp, &params);
        assert!(!corners.is_empty(), "upsample=1 produced no corners");
    }

    #[test]
    fn response_map_is_non_negative_everywhere() {
        const SIZE: usize = 29;
        const CELL: usize = 6;
        let img = synthetic_chessboard_aa(SIZE, CELL, (14.2, 14.6), 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 1,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        for &v in resp.data() {
            assert!(v >= 0.0, "negative response value: {v}");
        }
    }

    #[test]
    fn image_upsample_above_cap_is_clamped_not_panicked() {
        // `image_upsample >= 3` is unsupported in M1; the entry points
        // must clamp to the cap instead of panicking in
        // `upsample_bilinear_2x_if_needed` or mis-sizing downstream
        // buffers (Codex P1 on PR #40).
        const SIZE: usize = 29;
        const CELL: usize = 6;
        let img = synthetic_chessboard_aa(SIZE, CELL, (14.2, 14.6), 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 5,
            ..RadonDetectorParams::default()
        };
        assert_eq!(params.image_upsample_clamped(), MAX_IMAGE_UPSAMPLE);

        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        assert_eq!(resp.width(), SIZE * MAX_IMAGE_UPSAMPLE as usize);
        assert_eq!(resp.height(), SIZE * MAX_IMAGE_UPSAMPLE as usize);
    }

    #[test]
    fn image_upsample_zero_is_clamped_to_one() {
        // `image_upsample = 0` is valid `u32` but nonsensical. Clamp to
        // the min supported value so downstream code doesn't divide by
        // zero or size buffers to zero.
        const SIZE: usize = 21;
        const CELL: usize = 5;
        let img = synthetic_chessboard_aa(SIZE, CELL, (10.1, 10.4), 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 0,
            ..RadonDetectorParams::default()
        };
        assert_eq!(params.image_upsample_clamped(), 1);

        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        assert_eq!(resp.width(), SIZE);
        assert_eq!(resp.height(), SIZE);
    }
}
