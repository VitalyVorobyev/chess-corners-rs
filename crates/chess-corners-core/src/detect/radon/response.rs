//! Whole-image Duda-Frese Radon response computation.
//!
//! Computes the dense `R(x, y) = (max_α S_α − min_α S_α)²` response
//! over four ray angles using summed-area tables for `O(1)` per-pixel
//! ray sums. Optional 2× bilinear upsampling matches the paper-style
//! working grid used by the facade preset.
//!
//! # SAT element type
//!
//! Summed-area tables default to `i64`, which leaves far more headroom
//! than `u32` for normal image sizes. Enable the `radon-sat-u32` crate
//! feature to switch to `u32`, which halves SAT memory and widens SIMD
//! lanes at the cost of a ~16 MP image-size cap
//! (`255 · W · H ≤ u32::MAX`).

use serde::{Deserialize, Serialize};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
use core::simd::Simd;

#[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
use std::simd::cmp::SimdOrd;

use super::primitives::{box_blur_inplace, PeakFitMode};
use crate::ResponseMap;

/// Number of pixels processed per SIMD iteration in
/// `compute_response_row_simd`. Eight `i64` lanes is the natural
/// width on AVX-512 / NEON-pair / SVE machines; smaller-width
/// architectures fall back to the scalar tail handler in the same
/// kernel.
#[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
const RADON_LANES: usize = 8;

/// Summed-area-table element type. Gated by the `radon-sat-u32`
/// crate feature.
#[cfg(not(feature = "radon-sat-u32"))]
pub(crate) type SatElem = i64;

/// Summed-area-table element type (feature `radon-sat-u32`).
#[cfg(feature = "radon-sat-u32")]
pub(crate) type SatElem = u32;

/// Configuration for the whole-image Radon detector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RadonDetectorParams {
    /// Half-length of each ray in **working-resolution** pixels (i.e.
    /// post-upsample). The ray has `2·ray_radius + 1` samples. Paper
    /// default at `image_upsample=2` is 4 working pixels ⇒ 2 physical.
    pub ray_radius: u32,
    /// Image-level supersampling factor. `1` operates on the input
    /// pixel grid; `2` bilinearly upsamples first (paper default).
    /// M1 supports the set `{1, 2}`; values `>= 3` are clamped to `2`
    /// (see [`MAX_IMAGE_UPSAMPLE`](crate::unstable::MAX_IMAGE_UPSAMPLE)). Higher factors are future work.
    pub image_upsample: u32,
    /// Half-size of the box blur applied to the response map. `0`
    /// disables blurring; `1` yields a 3×3 box.
    pub response_blur_radius: u32,
    /// Peak-fit mode for the 3-point subpixel refinement.
    pub peak_fit: PeakFitMode,
    /// Response threshold as a fraction of the per-frame maximum
    /// response. The Radon `(max−min)²` response is unnormalized (it
    /// scales with ray length and contrast), so a relative floor adapts
    /// across frames where a fixed absolute cutoff would not.
    pub threshold_rel: f32,
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
            threshold_rel: 0.01,
            nms_radius: 4,
            min_cluster_size: 2,
        }
    }
}

/// Supported image-upsample factors: `{1, 2}`. Anything higher would
/// need a different upsampler; values `>= 3` are clamped to `2` at the
/// entry points rather than silently producing mismatched buffer sizes
/// downstream.
pub const MAX_IMAGE_UPSAMPLE: u32 = 2;

impl RadonDetectorParams {
    /// Clamp `image_upsample` into the supported set `{1, 2}`.
    /// Values outside that range are silently clamped — callers can
    /// detect truncation by comparing against
    /// [`MAX_IMAGE_UPSAMPLE`](crate::unstable::MAX_IMAGE_UPSAMPLE).
    #[inline]
    pub(crate) fn image_upsample_clamped(&self) -> u32 {
        self.image_upsample.clamp(1, MAX_IMAGE_UPSAMPLE)
    }

    #[inline]
    pub(crate) fn ray_radius_clamped(&self) -> u32 {
        self.ray_radius.max(1)
    }
}

/// Reusable scratch for the whole-image Radon detector. Holds the
/// upsampled image buffer, the four summed-area tables, the response
/// map, and the box-blur scratch. All buffers grow on demand and are
/// reused across frames — same pattern as `PyramidBuffers`.
#[derive(Debug, Default)]
pub struct RadonBuffers {
    upsampled: Vec<u8>,
    working_w: usize,
    working_h: usize,
    row_cumsum: Vec<SatElem>,
    col_cumsum: Vec<SatElem>,
    diag_pos_cumsum: Vec<SatElem>,
    diag_neg_cumsum: Vec<SatElem>,
    response: Vec<f32>,
    blur_scratch: Vec<f32>,
}

impl RadonBuffers {
    /// Create an empty set of buffers. They grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

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

fn upsample_bilinear_2x(src: &[u8], w: usize, h: usize, out: &mut [u8]) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(out.len(), 4 * w * h);
    if w == 0 || h == 0 {
        return;
    }
    let ww = 2 * w;

    let row_kernel = |iy: usize, dst: &mut [u8]| {
        let sy = iy as f32 * 0.5;
        let y0f = sy.floor();
        let y0 = (y0f as isize).max(0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let ty = (sy - y0f).clamp(0.0, 1.0);
        for (ix, out_px) in dst.iter_mut().enumerate() {
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
            *out_px = v.round().clamp(0.0, 255.0) as u8;
        }
    };

    #[cfg(feature = "rayon")]
    {
        out.par_chunks_mut(ww)
            .enumerate()
            .for_each(|(iy, row)| row_kernel(iy, row));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (iy, row) in out.chunks_mut(ww).enumerate() {
            row_kernel(iy, row);
        }
    }
}

#[inline]
fn sat_row(img: &[u8], w: usize, h: usize, row_cumsum: &mut [SatElem]) {
    debug_assert_eq!(row_cumsum.len(), w * h);
    for y in 0..h {
        let mut acc: SatElem = SatElem::default();
        for x in 0..w {
            acc += SatElem::from(img[y * w + x]);
            row_cumsum[y * w + x] = acc;
        }
    }
}

#[inline]
fn sat_col(img: &[u8], w: usize, h: usize, col_cumsum: &mut [SatElem]) {
    debug_assert_eq!(col_cumsum.len(), w * h);
    for x in 0..w {
        let mut acc: SatElem = SatElem::default();
        for y in 0..h {
            acc += SatElem::from(img[y * w + x]);
            col_cumsum[y * w + x] = acc;
        }
    }
}

#[inline]
fn sat_diag_pos(img: &[u8], w: usize, h: usize, diag_pos_cumsum: &mut [SatElem]) {
    debug_assert_eq!(diag_pos_cumsum.len(), w * h);
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
}

#[inline]
fn sat_diag_neg(img: &[u8], w: usize, h: usize, diag_neg_cumsum: &mut [SatElem]) {
    debug_assert_eq!(diag_neg_cumsum.len(), w * h);
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

    #[cfg(feature = "rayon")]
    {
        rayon::join(
            || {
                rayon::join(
                    || sat_row(img, w, h, row_cumsum),
                    || sat_col(img, w, h, col_cumsum),
                );
            },
            || {
                rayon::join(
                    || sat_diag_pos(img, w, h, diag_pos_cumsum),
                    || sat_diag_neg(img, w, h, diag_neg_cumsum),
                );
            },
        );
    }
    #[cfg(not(feature = "rayon"))]
    {
        sat_row(img, w, h, row_cumsum);
        sat_col(img, w, h, col_cumsum);
        sat_diag_pos(img, w, h, diag_pos_cumsum);
        sat_diag_neg(img, w, h, diag_neg_cumsum);
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

#[inline(always)]
fn radon_response_at(cs: &Cumsums<'_>, r: usize, x: usize, y: usize) -> f32 {
    let w = cs.w;
    let s_h_hi = cs.row[y * w + (x + r)];
    let s_h_lo = if x > r {
        cs.row[y * w + (x - r - 1)]
    } else {
        SatElem::default()
    };
    let s_h = s_h_hi - s_h_lo;

    let s_v_hi = cs.col[(y + r) * w + x];
    let s_v_lo = if y > r {
        cs.col[(y - r - 1) * w + x]
    } else {
        SatElem::default()
    };
    let s_v = s_v_hi - s_v_lo;

    let s_d1_hi = cs.diag_pos[(y + r) * w + (x + r)];
    let s_d1_lo = if x > r && y > r {
        cs.diag_pos[(y - r - 1) * w + (x - r - 1)]
    } else {
        SatElem::default()
    };
    let s_d1 = s_d1_hi - s_d1_lo;

    let s_d2_hi = cs.diag_neg[(y + r) * w + (x - r)];
    let s_d2_lo = if y > r && x + r + 1 < w {
        cs.diag_neg[(y - r - 1) * w + (x + r + 1)]
    } else {
        SatElem::default()
    };
    let s_d2 = s_d2_hi - s_d2_lo;

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
    d * d
}

#[inline]
fn compute_response_row(cs: &Cumsums<'_>, ray_radius: usize, y: usize, row: &mut [f32]) {
    let w = cs.w;
    let h = cs.h;
    let r = ray_radius;
    if y < r || y + r >= h {
        for v in row.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    for v in row[..r].iter_mut() {
        *v = 0.0;
    }

    #[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
    {
        compute_response_row_simd(cs, r, y, row);
    }
    #[cfg(not(all(feature = "simd", not(feature = "radon-sat-u32"))))]
    {
        for (x, out_px) in row.iter_mut().enumerate().take(w - r).skip(r) {
            *out_px = radon_response_at(cs, r, x, y);
        }
    }

    for v in row[(w - r)..].iter_mut() {
        *v = 0.0;
    }
}

#[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
#[inline]
fn compute_response_row_simd(cs: &Cumsums<'_>, r: usize, y: usize, row: &mut [f32]) {
    type S = Simd<i64, RADON_LANES>;

    let w = cs.w;

    row[r] = radon_response_at(cs, r, r, y);

    let interior_start = r + 1;
    let interior_end = w - r;
    let mut x = interior_start;

    if y <= r {
        for (x, cell) in row
            .iter_mut()
            .enumerate()
            .take(interior_end)
            .skip(interior_start)
        {
            *cell = radon_response_at(cs, r, x, y);
        }
        return;
    }

    let h_row_base = y * w;
    let v_hi_base = (y + r) * w;
    let v_lo_base = (y - r - 1) * w;
    let d1_hi_base = (y + r) * w;
    let d1_lo_base = (y - r - 1) * w;
    let d2_hi_base = (y + r) * w;
    let d2_lo_base = (y - r - 1) * w;

    while x + RADON_LANES < interior_end {
        let s_h_hi = S::from_slice(&cs.row[h_row_base + x + r..h_row_base + x + r + RADON_LANES]);
        let s_h_lo =
            S::from_slice(&cs.row[h_row_base + x - r - 1..h_row_base + x - r - 1 + RADON_LANES]);
        let s_h = s_h_hi - s_h_lo;

        let s_v_hi = S::from_slice(&cs.col[v_hi_base + x..v_hi_base + x + RADON_LANES]);
        let s_v_lo = S::from_slice(&cs.col[v_lo_base + x..v_lo_base + x + RADON_LANES]);
        let s_v = s_v_hi - s_v_lo;

        let s_d1_hi =
            S::from_slice(&cs.diag_pos[d1_hi_base + x + r..d1_hi_base + x + r + RADON_LANES]);
        let s_d1_lo = S::from_slice(
            &cs.diag_pos[d1_lo_base + x - r - 1..d1_lo_base + x - r - 1 + RADON_LANES],
        );
        let s_d1 = s_d1_hi - s_d1_lo;

        let s_d2_hi =
            S::from_slice(&cs.diag_neg[d2_hi_base + x - r..d2_hi_base + x - r + RADON_LANES]);
        let s_d2_lo = S::from_slice(
            &cs.diag_neg[d2_lo_base + x + r + 1..d2_lo_base + x + r + 1 + RADON_LANES],
        );
        let s_d2 = s_d2_hi - s_d2_lo;

        let mx = s_h.simd_max(s_v).simd_max(s_d1.simd_max(s_d2));
        let mn = s_h.simd_min(s_v).simd_min(s_d1.simd_min(s_d2));
        let diff = mx - mn;
        let arr = diff.to_array();
        for (lane, v) in arr.iter().enumerate() {
            let f = *v as f32;
            row[x + lane] = f * f;
        }

        x += RADON_LANES;
    }

    while x < interior_end {
        row[x] = radon_response_at(cs, r, x, y);
        x += 1;
    }
}

fn compute_response(cs: &Cumsums<'_>, ray_radius: usize, out: &mut [f32]) {
    let w = cs.w;
    let h = cs.h;
    debug_assert_eq!(out.len(), w * h);
    if w <= 2 * ray_radius || h <= 2 * ray_radius {
        out.fill(0.0);
        return;
    }

    #[cfg(feature = "rayon")]
    {
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            compute_response_row(cs, ray_radius, y, row);
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (y, row) in out.chunks_mut(w).enumerate() {
            compute_response_row(cs, ray_radius, y, row);
        }
    }
}

#[inline]
fn sat_to_f32(v: SatElem) -> f32 {
    v as f32
}

/// Compute the dense Radon response into `buffers.response` and return
/// a read-only [`RadonResponseView`] at **working resolution** (i.e.
/// `input_dim × image_upsample`).
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
        wh,
        params.response_blur_radius as usize,
    );
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
    pub(super) data: &'a [f32],
    pub(super) w: usize,
    pub(super) h: usize,
}

impl<'a> RadonResponseView<'a> {
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    pub fn to_response_map(&self) -> ResponseMap {
        ResponseMap {
            w: self.w,
            h: self.h,
            data: self.data.to_vec(),
        }
    }
}

#[inline]
fn upsample_bilinear_2x_if_needed(img: &[u8], w: usize, h: usize, up: u32, out: &mut Vec<u8>) {
    debug_assert_eq!(up, 2, "image_upsample must be 1 or 2");
    let _ = up;
    out.resize(4 * w * h, 0);
    upsample_bilinear_2x(img, w, h, out);
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_chessboard_aa;
    use super::*;

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

                let mut d2_sum: SatElem = SatElem::default();
                for k in 0..=(2 * ray_r) {
                    let xx = x + ray_r - k;
                    let yy = y + k - ray_r;
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

    #[test]
    fn radon_response_u8_handles_zero_extent_image() {
        let img: Vec<u8> = Vec::new();
        let params = RadonDetectorParams::default();
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, 0, 0, &params, &mut buffers);
        assert_eq!(resp.width(), 0);
        assert_eq!(resp.height(), 0);
        assert!(resp.data().is_empty());

        let params_no_upsample = RadonDetectorParams {
            image_upsample: 1,
            ..RadonDetectorParams::default()
        };
        let resp = radon_response_u8(&img, 0, 0, &params_no_upsample, &mut buffers);
        assert_eq!(resp.width(), 0);
        assert_eq!(resp.height(), 0);
    }

    #[cfg(all(feature = "simd", not(feature = "radon-sat-u32")))]
    #[test]
    fn simd_kernel_matches_scalar_at_every_interior_pixel() {
        for w in [16usize, 17, 18, 23, 24, 25, 32, 33] {
            let h = 24usize;
            let mut img = vec![0u8; w * h];
            for (i, p) in img.iter_mut().enumerate() {
                *p = ((i.wrapping_mul(37) ^ (i >> 3)) & 0xff) as u8;
            }
            let params = RadonDetectorParams {
                image_upsample: 1,
                response_blur_radius: 0,
                ..RadonDetectorParams::default()
            };
            let mut buffers = RadonBuffers::new();
            let response_snapshot: Vec<f32> = {
                let resp = radon_response_u8(&img, w, h, &params, &mut buffers);
                resp.data().to_vec()
            };

            let r = params.ray_radius_clamped() as usize;
            let cs = Cumsums {
                row: &buffers.row_cumsum,
                col: &buffers.col_cumsum,
                diag_pos: &buffers.diag_pos_cumsum,
                diag_neg: &buffers.diag_neg_cumsum,
                w,
                h,
            };
            for y in r..(h - r) {
                for x in r..(w - r) {
                    let expected = radon_response_at(&cs, r, x, y);
                    let got = response_snapshot[y * w + x];
                    assert!(
                        (expected - got).abs() < 1e-3,
                        "mismatch at w={w}, (x={x}, y={y}): scalar={expected}, simd={got}",
                    );
                }
            }
        }
    }
}
