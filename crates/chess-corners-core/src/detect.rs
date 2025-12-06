//! Corner detection utilities built on top of the dense ChESS response map.
use crate::descriptor::{corners_to_descriptors, Corner, CornerDescriptor};
use crate::response::chess_response_u8;
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "tracing")]
use tracing::instrument;

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
) -> Vec<CornerDescriptor> {
    let resp = chess_response_u8(img, w, h, params);
    let corners = detect_corners_from_response(&resp, params);
    let desc_radius = params.descriptor_ring_radius();
    corners_to_descriptors(img, w, h, desc_radius, corners)
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
    let ring_r = params.ring_radius() as i32;

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

/// Merge corners within a given radius, keeping the strongest response.
#[cfg_attr(feature = "tracing", instrument(level = "info", skip(corners)))]
pub fn merge_corners_simple(corners: &mut Vec<Corner>, radius: f32) -> Vec<Corner> {
    let r2 = radius * radius;
    let mut out: Vec<Corner> = Vec::new();

    // naive O(N^2) for now; N is small for a single chessboard frame
    'outer: for c in corners.drain(..) {
        for o in &mut out {
            let dx = c.xy[0] - o.xy[0];
            let dy = c.xy[1] - o.xy[1];
            if dx * dx + dy * dy <= r2 {
                // keep the stronger
                if c.strength > o.strength {
                    *o = c;
                }
                continue 'outer;
            }
        }
        out.push(c);
    }

    out
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

        // Expect orientation roughly aligned with a 45° grid (multiples of PI/4).
        let k = (best.orientation / core::f32::consts::FRAC_PI_4).round();
        let nearest = k * core::f32::consts::FRAC_PI_4;
        let near_axis = (best.orientation - nearest).abs() < 0.35;
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

    #[test]
    fn merge_corners_prefers_stronger_entries() {
        let mut corners = vec![
            Corner {
                xy: [10.0, 10.0],
                strength: 1.0,
            },
            Corner {
                xy: [11.0, 11.0],
                strength: 5.0,
            },
            Corner {
                xy: [20.0, 20.0],
                strength: 3.0,
            },
        ];
        let merged = merge_corners_simple(&mut corners, 2.5);
        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|c| (c.xy[0] - 11.0).abs() < 1e-6
            && (c.xy[1] - 11.0).abs() < 1e-6
            && (c.strength - 5.0).abs() < 1e-6));
        assert!(merged
            .iter()
            .any(|c| (c.xy[0] - 20.0).abs() < 1e-6 && (c.xy[1] - 20.0).abs() < 1e-6));
    }
}
