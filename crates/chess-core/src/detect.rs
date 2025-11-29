//! Corner detection utilities built on top of the dense ChESS response map.
use crate::response::chess_response_u8;
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Describes a detected chessboard corner in full-resolution image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct ChessCornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    pub y: f32,

    /// Effective scale (e.g. from pyramid level, 1.0 for full-res).
    pub scale: f32,

    /// ChESS response / strength at this corner (in the full-res image).
    pub response: f32,

    /// Orientation of the local grid axis at the corner, in radians.
    ///
    /// Convention:
    /// - in [0, PI)
    /// - one of the two orthogonal grid axes; the other is theta + PI/2.
    pub orientation: f32,

    /// A small discrete “phase” that encodes which quadrants are darker/brighter.
    ///
    /// Values 0..3, defined as:
    /// - phase_bit0: which diagonal is darker (0 or 1)
    /// - phase_bit1: orientation of the darker diagonal / local contrast
    pub phase: u8,

    /// Optional quality measure of how corner-like vs blob-like the local structure is.
    /// For now, use something like λ1/λ2 ratio or any monotone function of the structure tensor.
    pub anisotropy: f32,
}

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

/// Compute corners starting from an 8-bit grayscale image.
///
/// This is a convenience that combines:
/// - chess_response_u8 (dense response map)
/// - thresholding + NMS
/// - 5x5 center-of-mass subpixel refinement
pub fn find_corners_u8(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> Vec<ChessCornerDescriptor> {
    find_corners_u8_with_scale(img, w, h, params, 1.0)
}

/// Same as [`find_corners_u8`], but lets callers override the descriptor scale.
pub fn find_corners_u8_with_scale(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
    scale: f32,
) -> Vec<ChessCornerDescriptor> {
    let resp = chess_response_u8(img, w, h, params);
    let corners = detect_corners_from_response(&resp, params);
    corners_to_descriptors(img, w, h, scale, corners)
}

/// Core detector: run NMS + refinement on an existing response map.
///
/// Useful if you want to reuse the response map for debugging or tuning. Honors
/// relative vs absolute thresholds, enforces the configurable NMS radius, and
/// rejects isolated responses via `min_cluster_size`.
#[cfg_attr(
    feature = "tracing",
    instrument(level = "debug", skip(resp, params), fields(w = resp.w, h = resp.h))
)]
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

/// Convert raw corner candidates into full descriptors by sampling the source image.
///
/// Orientation, phase, and anisotropy follow the conventions documented on
/// [`ChessCornerDescriptor`].
pub fn corners_to_descriptors(
    img: &[u8],
    w: usize,
    h: usize,
    scale: f32,
    corners: Vec<Corner>,
) -> Vec<ChessCornerDescriptor> {
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let (orientation, anisotropy) =
            estimate_corner_orientation_anisotropy(img, w, h, c.xy[0], c.xy[1]);
        let phase = estimate_corner_phase(img, w, h, c.xy[0], c.xy[1], orientation);

        out.push(ChessCornerDescriptor {
            x: c.xy[0],
            y: c.xy[1],
            scale,
            response: c.strength,
            orientation,
            phase,
            anisotropy,
        });
    }
    out
}

/// Estimate local orientation and anisotropy of a corner using the image gradients
/// in a small window around (x, y) in full-res coordinates.
///
/// - `img` is a grayscale image in row-major layout.
/// - `x`, `y` are subpixel positions; we round to the nearest integer for the window center.
/// - The function returns (orientation, anisotropy).
///
/// The orientation is normalized to [0, PI); the anisotropy is a simple
/// det/trace² proxy where higher values indicate more corner-like structure.
fn estimate_corner_orientation_anisotropy(
    img: &[u8],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
) -> (f32, f32) {
    if w == 0 || h == 0 {
        return (0.0, 0.0);
    }

    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let max_x = w.saturating_sub(1) as i32;
    let max_y = h.saturating_sub(1) as i32;

    // Use a 7x7 window (radius 3) around the corner.
    let r = 3i32;
    let mut s_xx = 0.0f32;
    let mut s_xy = 0.0f32;
    let mut s_yy = 0.0f32;

    for dy in -r..=r {
        let yy = (cy + dy).clamp(0, max_y);
        let y_idx = yy as usize;
        for dx in -r..=r {
            let xx = (cx + dx).clamp(0, max_x);
            let x_idx = xx as usize;

            let x_plus = (xx + 1).clamp(0, max_x) as usize;
            let x_minus = (xx - 1).clamp(0, max_x) as usize;
            let y_plus = (yy + 1).clamp(0, max_y) as usize;
            let y_minus = (yy - 1).clamp(0, max_y) as usize;

            let ix = img[y_idx * w + x_plus] as f32 - img[y_idx * w + x_minus] as f32;
            let iy = img[y_plus * w + x_idx] as f32 - img[y_minus * w + x_idx] as f32;

            s_xx += ix * ix;
            s_xy += ix * iy;
            s_yy += iy * iy;
        }
    }

    let theta_grad = 0.5 * (2.0 * s_xy).atan2(s_xx - s_yy);
    let mut theta = theta_grad;
    if theta < 0.0 {
        theta += core::f32::consts::PI;
    }

    let trace = s_xx + s_yy;
    let det = s_xx * s_yy - s_xy * s_xy;
    let eps = 1e-6_f32;
    let anisotropy = det / (trace * trace + eps);

    (theta, anisotropy)
}

/// Estimate a small discrete phase code (0..3) describing which quadrants
/// around the corner are darker/brighter relative to the local grid axes. The
/// two bits encode which diagonal is darker and its orientation.
fn estimate_corner_phase(img: &[u8], w: usize, h: usize, x: f32, y: f32, theta: f32) -> u8 {
    if w == 0 || h == 0 {
        return 0;
    }

    let (st, ct) = theta.sin_cos();
    let u = (ct, st);
    let v = (-st, ct);
    let r = 1.5_f32;

    let p00x = x - r * u.0 - r * v.0;
    let p00y = y - r * u.1 - r * v.1;
    let p01x = x - r * u.0 + r * v.0;
    let p01y = y - r * u.1 + r * v.1;
    let p10x = x + r * u.0 - r * v.0;
    let p10y = y + r * u.1 - r * v.1;
    let p11x = x + r * u.0 + r * v.0;
    let p11y = y + r * u.1 + r * v.1;

    let i00 = sample_bilinear(img, w, h, p00x, p00y);
    let i01 = sample_bilinear(img, w, h, p01x, p01y);
    let i10 = sample_bilinear(img, w, h, p10x, p10y);
    let i11 = sample_bilinear(img, w, h, p11x, p11y);

    let d0 = i00 + i11;
    let d1 = i01 + i10;

    let phase_bit0 = (d0 < d1) as u8;
    let phase_bit1 = (i00 < i11) as u8;
    phase_bit0 | (phase_bit1 << 1)
}

fn sample_bilinear(img: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }

    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    let xf = x.clamp(0.0, max_x);
    let yf = y.clamp(0.0, max_y);

    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let wx = xf - x0 as f32;
    let wy = yf - y0 as f32;

    let i00 = img[y0 * w + x0] as f32;
    let i10 = img[y0 * w + x1] as f32;
    let i01 = img[y1 * w + x0] as f32;
    let i11 = img[y1 * w + x1] as f32;

    let i0 = i00 * (1.0 - wx) + i10 * wx;
    let i1 = i01 * (1.0 - wx) + i11 * wx;
    i0 * (1.0 - wy) + i1 * wy
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn make_quadrant_corner(size: u32, dark: u8, bright: u8) -> GrayImage {
        let mut img = GrayImage::from_pixel(size, size, Luma([dark]));
        let mid = size / 2;
        for y in 0..size {
            for x in 0..size {
                let in_top = y < mid;
                let in_left = x < mid;
                if in_top ^ in_left {
                    img.put_pixel(x, y, Luma([bright]));
                }
            }
        }
        img
    }

    #[test]
    fn descriptors_report_orientation_and_phase() {
        let size = 32u32;
        let params = ChessParams {
            threshold_rel: 0.01,
            ..Default::default()
        };

        let img = make_quadrant_corner(size, 20, 220);
        let corners = find_corners_u8(img.as_raw(), size as usize, size as usize, &params);
        assert!(!corners.is_empty(), "expected at least one descriptor");

        let best = corners
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .expect("non-empty");

        // Expect orientation roughly aligned with the image axes.
        let near_axis = best.orientation.abs() < 0.35
            || (best.orientation - core::f32::consts::FRAC_PI_2).abs() < 0.35;
        assert!(near_axis, "unexpected orientation {}", best.orientation);

        let mut brighter = img.clone();
        for p in brighter.pixels_mut() {
            p[0] = p[0].saturating_add(5);
        }

        let brighter_corners =
            find_corners_u8(brighter.as_raw(), size as usize, size as usize, &params);
        assert!(!brighter_corners.is_empty());
        let best_brighter = brighter_corners
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .expect("non-empty brighter");

        assert!((best.x - best_brighter.x).abs() < 0.5 && (best.y - best_brighter.y).abs() < 0.5);
        assert_eq!(best.phase, best_brighter.phase);
    }
}
