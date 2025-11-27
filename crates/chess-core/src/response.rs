//! Dense ChESS response computation for 8-bit grayscale inputs.
use crate::ring::ring_offsets;
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use core::simd::Simd;

#[cfg(feature = "simd")]
use std::simd::prelude::SimdUint;

#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;

#[cfg(feature = "simd")]
const LANES: usize = 16;

#[cfg(feature = "simd")]
type U8s = Simd<u8, LANES>;

#[cfg(feature = "simd")]
type F32s = Simd<f32, LANES>;

/// Compute the dense ChESS response for an 8-bit grayscale image.
///
/// Automatically parallelizes over rows when built with the `rayon` feature.
pub fn chess_response_u8(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    // rayon path compiled only when feature is enabled
    #[cfg(feature = "rayon")]
    {
        return compute_response_parallel(img, w, h, params);
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
/// The ROI is given in image coordinates [x0, x1) × [y0, y1). The returned
/// ResponseMap has width (x1 - x0) and height (y1 - y0), with coordinates
/// relative to (x0, y0).
///
/// Pixels where the ChESS ring would go out of bounds (w.r.t. the *full*
/// image) are left at 0.0, and will be ignored by the detector because they
/// lie inside the border margin.
pub fn chess_response_u8_patch(
    img: &[u8],
    img_w: usize,
    img_h: usize,
    params: &ChessParams,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> ResponseMap {
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

    let r = params.radius as i32;
    let ring = ring_offsets(params.radius);

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
        for px in 0..patch_w {
            let gx = x0 + px;
            if gx < gx0 || gx >= gx1 {
                continue;
            }

            let resp = chess_response_at_u8(img, img_w, gx as i32, gy as i32, ring);
            data[py * patch_w + px] = resp;
        }
    }

    ResponseMap {
        w: patch_w,
        h: patch_h,
        data,
    }
}


fn compute_response_sequential(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ResponseMap {
    let r = params.radius as i32;
    let ring = ring_offsets(params.radius);

    let mut data = vec![0.0f32; w * h];

    // only evaluate where full ring fits
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    for y in y0..y1 {
        let row = &mut data[y * w..(y + 1) * w];
        compute_row(img, w, y as i32, &ring, row, x0, x1);
    }

    ResponseMap { w, h, data }
}

fn compute_response_sequential_scalar(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ResponseMap {
    let r = params.radius as i32;
    let ring = ring_offsets(params.radius);

    let mut data = vec![0.0f32; w * h];

    // only evaluate where full ring fits
    let x0 = r as usize;
    let y0 = r as usize;
    let x1 = w - r as usize;
    let y1 = h - r as usize;

    for y in y0..y1 {
        let row = &mut data[y * w..(y + 1) * w];
        compute_row_scalar(img, w, y as i32, &ring, row, x0, x1);
    }

    ResponseMap { w, h, data }
}

#[cfg(feature = "rayon")]
fn compute_response_parallel(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    let r = params.radius as i32;
    let ring = ring_offsets(params.radius);
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
        compute_row(img, w, y_i, &ring, row, x0, x1);
    });

    ResponseMap { w, h, data }
}

// Fallback stub when rayon feature is off so the name still exists
#[cfg(not(feature = "rayon"))]
#[allow(dead_code)]
fn compute_response_parallel(img: &[u8], w: usize, h: usize, params: &ChessParams) -> ResponseMap {
    compute_response_sequential(img, w, h, params)
}

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

fn compute_row(
    img: &[u8],
    w: usize,
    y: i32,
    ring: &[(i32, i32); 16],
    row: &mut [f32],
    x0: usize,
    x1: usize,
) {
    #[cfg(feature = "simd")]
    {
        compute_row_simd(img, w, y, ring, row, x0, x1);
        return;
    }

    // fallback
    #[cfg(not(feature = "simd"))]
    compute_row_scalar(img, w, y, ring, row, x0, x1);
}

fn compute_row_scalar(
    img: &[u8],
    w: usize,
    y: i32,
    ring: &[(i32, i32); 16],
    row: &mut [f32],
    x0: usize,
    x1: usize,
) {
    for x in x0..x1 {
        let resp = chess_response_at_u8(img, w, x as i32, y, ring);
        row[x] = resp;
    }
}

#[cfg(feature = "simd")]
fn compute_row_simd(
    img: &[u8],
    w: usize,
    y: i32,
    ring: &[(i32, i32); 16],
    row: &mut [f32],
    x0: usize,
    x1: usize,
) {
    let y_usize = y as usize;

    let mut x = x0;

    while x + LANES <= x1 {
        // Gather ring samples for LANES pixels starting at x
        let mut s: [U8s; 16] = [U8s::splat(0); 16];

        for k in 0..16 {
            let (dx, dy) = ring[k];
            let yy = (y + dy) as usize;
            let xx = (x as i32 + dx) as usize;
            let base = yy * w + xx;

            // SAFETY: x range + radius guarantees we stay in bounds
            s[k] = U8s::from_slice(&img[base..base + LANES]);
        }

        // Sum of ring values (for neighbor mean)
        let mut sum_ring_v = F32s::splat(0.0);
        for k in 0..16 {
            sum_ring_v += s[k].cast::<f32>();
        }

        // SR
        let mut sr_v = F32s::splat(0.0);
        for k in 0..4 {
            let a = s[k].cast::<f32>() + s[k + 8].cast::<f32>();
            let b = s[k + 4].cast::<f32>() + s[k + 12].cast::<f32>();
            let diff = (a - b).abs();
            sr_v += diff;
        }

        // DR
        let mut dr_v = F32s::splat(0.0);
        for k in 0..8 {
            let a = s[k].cast::<f32>();
            let b = s[k + 8].cast::<f32>();
            dr_v += (a - b).abs();
        }

        // Convert vectors to scalar arrays for the MR step
        let sr_arr = sr_v.to_array();
        let dr_arr = dr_v.to_array();
        let sum_ring_arr = sum_ring_v.to_array();

        // Per-lane local mean + final response
        for lane in 0..LANES {
            let xx = x + lane;

            // center + 4-neighborhood (scalar) at base resolution
            let c  = img[y_usize * w + xx] as f32;
            let n  = img[(y_usize - 1) * w + xx] as f32;
            let s0 = img[(y_usize + 1) * w + xx] as f32;
            let e  = img[y_usize * w + (xx + 1)] as f32;
            let w0 = img[y_usize * w + (xx - 1)] as f32;

            let mu_n = sum_ring_arr[lane] / 16.0;
            let mu_l = (c + n + s0 + e + w0) / 5.0;
            let mr = (mu_n - mu_l).abs();

            row[xx] = sr_arr[lane] - dr_arr[lane] - 16.0 * mr;
        }

        x += LANES;
    }

    // Tail: scalar for remaining pixels
    while x < x1 {
        let resp = chess_response_at_u8(img, w, x as i32, y, ring);
        row[x] = resp;
        x += 1;
    }
}
