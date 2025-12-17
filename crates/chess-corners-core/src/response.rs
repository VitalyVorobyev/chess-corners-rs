//! Dense ChESS response computation for 8-bit grayscale inputs.
use crate::ring::RingOffsets;
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use core::simd::Simd;

#[cfg(feature = "simd")]
use std::simd::prelude::{SimdInt, SimdUint};

#[cfg(feature = "simd")]
const LANES: usize = 16;

#[cfg(feature = "simd")]
type U8s = Simd<u8, LANES>;

#[cfg(feature = "simd")]
type I16s = Simd<i16, LANES>;

#[cfg(feature = "simd")]
type I32s = Simd<i32, LANES>;

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Rectangular region of interest, in image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

#[inline]
fn ring_from_params(params: &ChessParams) -> (RingOffsets, &'static [(i32, i32); 16]) {
    let ring = params.ring();
    (ring, ring.offsets())
}

/// Compute the dense ChESS response for an 8-bit grayscale image.
///
/// The response at each valid pixel center is computed from a 16‑sample ring
/// around the pixel and a 5‑pixel cross at the center. For a given center
/// `c`, let `s[0..16)` be the ring samples in the canonical order:
///
/// - `SR` (sum of “square” responses) compares four opposite quadrants on
///   the ring:
///
///   ```text
///   SR = sum_{k=0..3} | (s[k] + s[k+8]) - (s[k+4] + s[k+12]) |
///   ```
///
/// - `DR` (sum of “difference” responses) enforces edge‑like structure:
///
///   ```text
///   DR = sum_{k=0..7} | s[k] - s[k+8] |
///   ```
///
/// - `μₙ` is the mean of all 16 ring samples.
/// - `μₗ` is the local mean of the 5‑pixel cross at the center
///   (`c`, `north`, `south`, `east`, `west`).
///
/// The final ChESS response is:
///
/// ```text
/// R = SR - DR - 16 * |μₙ - μₗ|
/// ```
///
/// where high positive values correspond to chessboard‑like corners.
///
/// # Implementation strategy
///
/// Internally the image is processed row‑by‑row, but only pixels whose
/// full ring lies inside the image bounds are evaluated; border pixels
/// are left at zero in the returned [`ResponseMap`].
///
/// - Without any features, `chess_response_u8` uses a straightforward
///   nested `for y { for x { ... } }` scalar loop and relies on the
///   compiler’s auto‑vectorization in release builds.
/// - With the `rayon` feature, the work is split into independent row
///   slices and processed in parallel using `rayon::par_chunks_mut`.
/// - With the `simd` feature, the inner loop over `x` is rewritten to
///   operate on `LANES` pixels at a time using portable SIMD vectors
///   (currently 16 lanes of `u8`). The ring samples are gathered into
///   SIMD registers, `SR`/`DR`/`μₙ` are accumulated in integer vectors,
///   and the final response is written back per lane.
/// - With both `rayon` and `simd` enabled, each row is processed in
///   parallel *and* each row uses the SIMD‑accelerated inner loop.
///
/// All feature combinations produce the same output values (within a
/// small tolerance for floating‑point rounding), and differ only in
/// performance characteristics.
#[cfg_attr(
    feature = "tracing",
    instrument(level = "info", skip(img, params), fields(w, h))
)]
pub fn chess_response_u8(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    // rayon path compiled only when feature is enabled
    #[cfg(feature = "rayon")]
    {
        compute_response_parallel(img, w, h, params)
    }
    #[cfg(not(feature = "rayon"))]
    {
        compute_response_sequential(img, w, h, params)
    }
}

/// Always uses the scalar implementation (no rayon, no SIMD),
/// useful for reference/golden testing.
pub fn chess_response_u8_scalar(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ResponseMap {
    compute_response_sequential_scalar(img, w, h, params)
}

/// Compute the ChESS response only inside a rectangular ROI of the image.
///
/// The ROI is given in image coordinates [x0, x1) × [y0, y1) via [`Roi`]. The
/// returned [`ResponseMap`] has width (x1 - x0) and height (y1 - y0), with
/// coordinates relative to (x0, y0).
///
/// Pixels where the ChESS ring would go out of bounds (w.r.t. the *full*
/// image) are left at 0.0, and will be ignored by the detector because they
/// lie inside the border margin. Internally this reuses the same scalar,
/// SIMD, and optional `rayon` row kernels as [`chess_response_u8`], so ROI
/// refinement benefits from the same feature combinations as the full-frame
/// response path.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "debug",
        skip(img, params),
        fields(img_w, img_h, roi_w = roi.x1 - roi.x0, roi_h = roi.y1 - roi.y0)
    )
)]
pub fn chess_response_u8_patch(
    img: &[u8],
    img_w: usize,
    img_h: usize,
    params: &ChessParams,
    roi: Roi,
) -> ResponseMap {
    let Roi { x0, y0, x1, y1 } = roi;

    // Clamp ROI to the image bounds
    let x0 = x0.min(img_w);
    let y0 = y0.min(img_h);
    let x1 = x1.min(img_w);
    let y1 = y1.min(img_h);

    if x1 <= x0 || y1 <= y0 {
        return ResponseMap {
            w: 0,
            h: 0,
            data: Vec::new(),
        };
    }

    let patch_w = x1 - x0;
    let patch_h = y1 - y0;
    let mut data = vec![0.0f32; patch_w * patch_h];

    let (ring_kind, ring) = ring_from_params(params);
    let r = ring_kind.radius() as i32;

    // Safe region for ring centers in *global* image coordinates
    let gx0 = r as usize;
    let gy0 = r as usize;
    let gx1 = img_w - r as usize;
    let gy1 = img_h - r as usize;

    for py in 0..patch_h {
        let gy = y0 + py;
        if gy < gy0 || gy >= gy1 {
            continue;
        }

        // Global x-range where the ring is valid on this row.
        let row_gx0 = x0.max(gx0);
        let row_gx1 = x1.min(gx1);
        if row_gx0 >= row_gx1 {
            continue;
        }

        let row = &mut data[py * patch_w..(py + 1) * patch_w];
        let rel_start = row_gx0 - x0;
        let rel_end = row_gx1 - x0;
        let dst_row = &mut row[rel_start..rel_end];

        #[cfg(feature = "simd")]
        {
            compute_row_range_simd(img, img_w, gy as i32, ring, dst_row, row_gx0, row_gx1);
        }

        #[cfg(not(feature = "simd"))]
        {
            compute_row_range_scalar(img, img_w, gy as i32, ring, dst_row, row_gx0, row_gx1);
        }
    }

    ResponseMap {
        w: patch_w,
        h: patch_h,
        data,
    }
}

#[cfg(not(feature = "rayon"))]
fn compute_response_sequential(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ResponseMap {
    let (ring_kind, ring) = ring_from_params(params);
    let r = ring_kind.radius() as i32;

    let mut data = vec![0.0f32; w * h];

    // only evaluate where full ring fits
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    for y in y0..y1 {
        let row = &mut data[y * w..(y + 1) * w];
        let dst_row = &mut row[x0..x1];

        #[cfg(feature = "simd")]
        {
            compute_row_range_simd(img, w, y as i32, ring, dst_row, x0, x1);
        }

        #[cfg(not(feature = "simd"))]
        {
            compute_row_range_scalar(img, w, y as i32, ring, dst_row, x0, x1);
        }
    }

    ResponseMap { w, h, data }
}

fn compute_response_sequential_scalar(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ResponseMap {
    let (ring_kind, ring) = ring_from_params(params);
    let r = ring_kind.radius() as i32;

    let mut data = vec![0.0f32; w * h];

    // only evaluate where full ring fits
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    for y in y0..y1 {
        let row = &mut data[y * w..(y + 1) * w];
        let dst_row = &mut row[x0..x1];
        compute_row_range_scalar(img, w, y as i32, ring, dst_row, x0, x1);
    }

    ResponseMap { w, h, data }
}

#[cfg(feature = "rayon")]
fn compute_response_parallel(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    let (ring_kind, ring) = ring_from_params(params);
    let r = ring_kind.radius() as i32;
    let mut data = vec![0.0f32; w * h];

    // ring margin
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    // Parallelize over rows. We keep the exact same logic and write
    // each row's slice independently.
    data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y_i = y as i32;
        if y_i < y0 as i32 || y_i >= y1 as i32 {
            return;
        }

        let dst_row = &mut row[x0..x1];

        #[cfg(feature = "simd")]
        {
            compute_row_range_simd(img, w, y_i, ring, dst_row, x0, x1);
        }

        #[cfg(not(feature = "simd"))]
        {
            compute_row_range_scalar(img, w, y_i, ring, dst_row, x0, x1);
        }
    });

    ResponseMap { w, h, data }
}

// Fallback stub when rayon feature is off so the name still exists
#[cfg(not(feature = "rayon"))]
#[allow(dead_code)]
fn compute_response_parallel(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    compute_response_sequential(img, w, h, params)
}

/// Low-level ChESS response at a single pixel center.
///
/// This is the scalar reference implementation used by both the sequential
/// and SIMD paths:
///
/// - gathers 16 ring samples around `(x, y)` using the offsets defined in
///   [`crate::ring`],
/// - computes `SR`, `DR`, the ring mean `μₙ`, and the 5‑pixel local mean
///   `μₗ`, and
/// - returns `R = SR - DR - 16 * |μₙ - μₗ|` as a `f32`.
///
/// Callers are responsible for ensuring that `(x, y)` is far enough from the
/// image border so that all ring and 5‑pixel cross accesses are in‑bounds.
#[inline(always)]
fn chess_response_at_u8(img: &[u8], w: usize, x: i32, y: i32, ring: &[(i32, i32); 16]) -> f32 {
    // gather ring samples into i32
    let mut s = [0i32; 16];
    for k in 0..16 {
        let (dx, dy) = ring[k];
        let xx = (x + dx) as usize;
        let yy = (y + dy) as usize;
        s[k] = img[yy * w + xx] as i32;
    }

    // SR
    let mut sr = 0i32;
    for k in 0..4 {
        let a = s[k] + s[k + 8];
        let b = s[k + 4] + s[k + 12];
        sr += (a - b).abs();
    }

    // DR
    let mut dr = 0i32;
    for k in 0..8 {
        dr += (s[k] - s[k + 8]).abs();
    }

    // neighbor mean
    let sum_ring: i32 = s.iter().sum();
    let mu_n = sum_ring as f32 / 16.0;

    // local mean (5 px cross)
    let c = img[(y as usize) * w + (x as usize)] as f32;
    let n = img[((y - 1) as usize) * w + (x as usize)] as f32;
    let s0 = img[((y + 1) as usize) * w + (x as usize)] as f32;
    let e = img[(y as usize) * w + ((x + 1) as usize)] as f32;
    let w0 = img[(y as usize) * w + ((x - 1) as usize)] as f32;
    let mu_l = (c + n + s0 + e + w0) / 5.0;

    let mr = (mu_n - mu_l).abs();

    (sr as f32) - (dr as f32) - 16.0 * mr
}

fn compute_row_range_scalar(
    img: &[u8],
    w: usize,
    y: i32,
    ring: &[(i32, i32); 16],
    dst_row: &mut [f32],
    x_start: usize,
    x_end: usize,
) {
    for (offset, x) in (x_start..x_end).enumerate() {
        dst_row[offset] = chess_response_at_u8(img, w, x as i32, y, ring);
    }
}

#[cfg(feature = "simd")]
fn compute_row_range_simd(
    img: &[u8],
    w: usize,
    y: i32,
    ring: &[(i32, i32); 16],
    dst_row: &mut [f32],
    x_start: usize,
    x_end: usize,
) {
    let y_usize = y as usize;

    // Precompute row bases for each ring sample to avoid recomputing (y+dy)*w.
    let mut ring_bases: [isize; 16] = [0; 16];
    for k in 0..16 {
        let (dx, dy) = ring[k];
        let yy = (y + dy) as usize;
        ring_bases[k] = (yy * w) as isize + dx as isize;
    }

    let mut x = x_start;

    while x + LANES <= x_end {
        // Gather ring samples for LANES pixels starting at x
        let mut s: [I16s; 16] = [I16s::splat(0); 16];

        for k in 0..16 {
            let base = (ring_bases[k] + x as isize) as usize;

            // SAFETY: x range + radius guarantees we stay in bounds
            let v_u8 = U8s::from_slice(&img[base..base + LANES]);
            s[k] = v_u8.cast::<i16>();
        }

        // Sum of ring values (for neighbor mean)
        let mut sum_ring_v = I32s::splat(0);
        for &v in &s {
            sum_ring_v += v.cast::<i32>();
        }

        // SR
        let mut sr_v = I32s::splat(0);
        for k in 0..4 {
            let a = s[k].cast::<i32>() + s[k + 8].cast::<i32>();
            let b = s[k + 4].cast::<i32>() + s[k + 12].cast::<i32>();
            sr_v += (a - b).abs();
        }

        // DR
        let mut dr_v = I32s::splat(0);
        for k in 0..8 {
            let a = s[k].cast::<i32>();
            let b = s[k + 8].cast::<i32>();
            dr_v += (a - b).abs();
        }

        // Convert vectors to scalar arrays for the MR step
        let sr_arr = sr_v.cast::<f32>().to_array();
        let dr_arr = dr_v.cast::<f32>().to_array();
        let sum_ring_arr = sum_ring_v.cast::<f32>().to_array();

        // Per-lane local mean + final response
        for lane in 0..LANES {
            let xx = x + lane;
            let px = xx - x_start;

            // center + 4-neighborhood (scalar) at base resolution
            let c = img[y_usize * w + xx] as f32;
            let n = img[(y_usize - 1) * w + xx] as f32;
            let s0 = img[(y_usize + 1) * w + xx] as f32;
            let e = img[y_usize * w + (xx + 1)] as f32;
            let w0 = img[y_usize * w + (xx - 1)] as f32;

            let mu_n = sum_ring_arr[lane] / 16.0;
            let mu_l = (c + n + s0 + e + w0) / 5.0;
            let mr = (mu_n - mu_l).abs();

            dst_row[px] = sr_arr[lane] - dr_arr[lane] - 16.0 * mr;
        }

        x += LANES;
    }

    // Tail: scalar for remaining pixels
    while x < x_end {
        let px = x - x_start;
        let resp = chess_response_at_u8(img, w, x as i32, y, ring);
        dst_row[px] = resp;
        x += 1;
    }
}
