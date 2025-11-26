//! Corner detection utilities built on top of the dense ChESS response map.
use crate::response::chess_response_u8;
use crate::{ChessParams, ResponseMap};
use std::time::Instant;

/// A detected ChESS corner (subpixel).
#[derive(Clone, Debug)]
pub struct Corner {
    /// Subpixel location in image coordinates (x, y).
    pub xy: [f32; 2],
    /// Raw ChESS response at the integer peak (before COM refinement).
    pub strength: f32,
    /// Pyramid level / scale (0 for full-res; reserved for future multi-scale).
    pub scale: u8,
}

/// Timed detection outcome containing corners and profiling data.
pub struct ChessResult {
    /// Refined corners (in image coordinates).
    pub corners: Vec<Corner>,
    /// Time spent computing the dense response (milliseconds).
    pub resp_ms: f64,
    /// Time spent on thresholding, NMS, and refinement (milliseconds).
    pub detect_ms: f64,
}

/// Compute corners starting from an 8-bit grayscale image.
///
/// This is a convenience that combines:
/// - chess_response_u8 (dense response map)
/// - thresholding + NMS
/// - 5x5 center-of-mass subpixel refinement
pub fn find_corners_u8_with_trace(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> ChessResult {
    let resp_started = Instant::now();
    let resp = chess_response_u8(img, w, h, params);
    let resp_ms = resp_started.elapsed().as_secs_f64() * 1000.0;

    let detect_started = Instant::now();
    let corners = detect_corners_from_response(&resp, params);
    let detect_ms = detect_started.elapsed().as_secs_f64() * 1000.0;

    ChessResult {
        corners,
        resp_ms,
        detect_ms,
    }
}

/// Compute corners starting from an 8-bit grayscale image.
///
/// This is a convenience that combines:
/// - chess_response_u8 (dense response map)
/// - thresholding + NMS
/// - 5x5 center-of-mass subpixel refinement
pub fn find_corners_u8(img: &[u8], w: usize, h: usize, params: &ChessParams) -> Vec<Corner> {
    let resp = chess_response_u8(img, w, h, params);
    detect_corners_from_response(&resp, params)
}

/// Core detector: run NMS + refinement on an existing response map.
///
/// Useful if you want to reuse the response map for debugging or tuning. Honors
/// relative vs absolute thresholds, enforces the configurable NMS radius, and
/// rejects isolated responses via `min_cluster_size`.
pub fn detect_corners_from_response(resp: &ResponseMap, params: &ChessParams) -> Vec<Corner> {
    let w = resp.w;
    let h = resp.h;

    if w == 0 || h == 0 {
        return Vec::new();
    }

    // Compute global max response to derive relative threshold
    let mut max_r = f32::NEG_INFINITY;
    for &v in &resp.data {
        if v > max_r {
            max_r = v;
        }
    }
    if !max_r.is_finite() {
        return Vec::new();
    }

    let mut thr = params.threshold_abs.unwrap_or(params.threshold_rel * max_r);

    if thr < 0.0 {
        // Don’t use a negative threshold; that would accept noise.
        thr = 0.0;
    }

    let nms_r = params.nms_radius as i32;
    let refine_r = 2i32; // 5x5 window
    let ring_r = params.radius as i32;

    // We need to stay away from the borders enough to:
    // - have a full NMS window
    // - have a full 5x5 refinement window
    // The response map itself is valid in [ring_r .. w-ring_r), but
    // we don't want to sample outside [0..w/h) during refinement.
    let border = (ring_r + nms_r + refine_r).max(0) as usize;

    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    let mut corners = Vec::new();

    for y in border..(h - border) {
        for x in border..(w - border) {
            let v = resp.at(x, y);
            if v < thr {
                continue;
            }

            // Local maximum in NMS window
            if !is_local_max(resp, x, y, nms_r, v) {
                continue;
            }

            // Reject isolated pixels: require a minimum number of positive
            // neighbors in the same NMS window.
            let cluster_size = count_positive_neighbors(resp, x, y, nms_r);
            if cluster_size < params.min_cluster_size {
                continue;
            }

            let sub_xy = refine_com_5x5(resp, x, y);

            corners.push(Corner {
                xy: sub_xy,
                strength: v,
                scale: 0,
            });
        }
    }

    corners
}

fn is_local_max(resp: &ResponseMap, x: usize, y: usize, r: i32, v: f32) -> bool {
    let w = resp.w as i32;
    let h = resp.h as i32;
    let cx = x as i32;
    let cy = y as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = cx + dx;
            let yy = cy + dy;
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                continue;
            }
            let vv = resp.at(xx as usize, yy as usize);
            if vv > v {
                return false;
            }
        }
    }
    true
}

fn count_positive_neighbors(resp: &ResponseMap, x: usize, y: usize, r: i32) -> u32 {
    let w = resp.w as i32;
    let h = resp.h as i32;
    let cx = x as i32;
    let cy = y as i32;
    let mut count = 0;

    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = cx + dx;
            let yy = cy + dy;
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                continue;
            }
            let vv = resp.at(xx as usize, yy as usize);
            if vv > 0.0 {
                count += 1;
            }
        }
    }

    count
}

/// 5x5 center-of-mass refinement around an integer peak.
///
/// We use only non-negative responses (max(0, R)) so that negative sidelobes
/// don’t bias the estimate.
fn refine_com_5x5(resp: &ResponseMap, x: usize, y: usize) -> [f32; 2] {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sw = 0.0;

    let w = resp.w;
    let h = resp.h;

    // We assume caller has ensured x,y are at least 2 pixels away from borders.
    // Still, we clamp indices defensively in case params are mis-set.
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let xx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
            let yy = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;

            let w_px = resp.at(xx, yy).max(0.0);
            sx += (xx as f32) * w_px;
            sy += (yy as f32) * w_px;
            sw += w_px;
        }
    }

    if sw > 0.0 {
        [sx / sw, sy / sw]
    } else {
        [x as f32, y as f32]
    }
}
