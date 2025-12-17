//! Corner detection utilities built on top of the dense ChESS response map.
use crate::descriptor::{corners_to_descriptors, Corner, CornerDescriptor};
use crate::imageview::ImageView;
use crate::refine::{CornerRefiner, RefineContext, RefineStatus, Refiner};
use crate::response::chess_response_u8;
use crate::{ChessParams, ResponseMap};

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Compute corners starting from an 8-bit grayscale image.
///
/// This is a convenience that combines:
/// - chess_response_u8 (dense response map)
/// - thresholding + NMS
/// - subpixel refinement driven by [`ChessParams::refiner`]
pub fn find_corners_u8(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
) -> Vec<CornerDescriptor> {
    let mut refiner = Refiner::from_kind(params.refiner.clone());
    find_corners_u8_with_refiner(img, w, h, params, &mut refiner)
}

/// Compute corners starting from an 8-bit grayscale image using a custom refiner.
pub fn find_corners_u8_with_refiner(
    img: &[u8],
    w: usize,
    h: usize,
    params: &ChessParams,
    refiner: &mut dyn CornerRefiner,
) -> Vec<CornerDescriptor> {
    let resp = chess_response_u8(img, w, h, params);
    let image =
        ImageView::from_u8_slice(w, h, img).expect("image dimensions must match buffer length");
    let corners = detect_corners_from_response_with_refiner(&resp, params, Some(image), refiner);
    let desc_radius = params.descriptor_ring_radius();
    corners_to_descriptors(img, w, h, desc_radius, corners)
}

/// Core detector: run NMS + refinement on an existing response map.
///
/// Useful if you want to reuse the response map for debugging or tuning. Honors
/// relative vs absolute thresholds, enforces the configurable NMS radius, and
/// rejects isolated responses via `min_cluster_size`.
pub fn detect_corners_from_response(resp: &ResponseMap, params: &ChessParams) -> Vec<Corner> {
    let mut refiner = Refiner::from_kind(params.refiner.clone());
    detect_corners_from_response_with_refiner(resp, params, None, &mut refiner)
}

/// Detector variant that accepts a user-provided refiner implementation.
pub fn detect_corners_from_response_with_refiner(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    refiner: &mut dyn CornerRefiner,
) -> Vec<Corner> {
    detect_corners_from_response_impl(resp, params, image, refiner)
}

#[cfg_attr(
    feature = "tracing",
    instrument(level = "debug", skip(resp, params, image, refiner), fields(w = resp.w, h = resp.h))
)]
fn detect_corners_from_response_impl(
    resp: &ResponseMap,
    params: &ChessParams,
    image: Option<ImageView<'_>>,
    refiner: &mut dyn CornerRefiner,
) -> Vec<Corner> {
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
    let refine_r = refiner.radius();
    let ring_r = params.ring_radius() as i32;

    // We need to stay away from the borders enough to:
    // - have a full NMS window
    // - have a full refinement window
    // The response map itself is valid in [ring_r .. w-ring_r), but
    // we don't want to sample outside [0..w/h) during refinement.
    let border = (ring_r + nms_r + refine_r).max(0) as usize;

    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    let mut corners = Vec::new();
    let ctx = RefineContext {
        image,
        response: Some(resp),
    };

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

            let seed_xy = [x as f32, y as f32];
            let res = refiner.refine(seed_xy, ctx);

            if matches!(res.status, RefineStatus::Accepted) {
                corners.push(Corner {
                    xy: res.xy,
                    strength: v,
                });
            }
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
    use crate::refine::{
        CenterOfMassConfig, CenterOfMassRefiner, RefineContext, RefineStatus, RefinerKind,
    };
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
    fn descriptors_report_orientation_is_stable() {
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

        let dtheta = (best.orientation - best_brighter.orientation).abs();
        let dtheta = dtheta.min(core::f32::consts::PI - dtheta);
        assert!(
            dtheta < 0.35,
            "unexpected orientation delta after brightness shift: {dtheta}"
        );
    }

    #[test]
    fn default_refiner_matches_center_of_mass() {
        let mut resp = ResponseMap {
            w: 32,
            h: 32,
            data: vec![0.0; 32 * 32],
        };

        let cx = 16usize;
        let cy = 16usize;
        let w = resp.w;

        resp.data[cy * w + cx] = 10.0;
        resp.data[cy * w + (cx + 1)] = 6.0;
        resp.data[(cy + 1) * w + cx] = 5.0;
        resp.data[(cy + 1) * w + (cx + 1)] = 4.0;

        let params = ChessParams {
            threshold_rel: 0.01,
            ..Default::default()
        };

        let mut refiner = CenterOfMassRefiner::new(CenterOfMassConfig::default());
        let ctx = RefineContext {
            image: None,
            response: Some(&resp),
        };
        let expected = refiner.refine([cx as f32, cy as f32], ctx);
        assert_eq!(expected.status, RefineStatus::Accepted);

        let corners = detect_corners_from_response(&resp, &params);
        assert_eq!(corners.len(), 1);
        let c = &corners[0];
        assert!((c.xy[0] - expected.xy[0]).abs() < 1e-6);
        assert!((c.xy[1] - expected.xy[1]).abs() < 1e-6);
    }

    #[test]
    fn params_refiner_controls_margin() {
        let mut resp = ResponseMap {
            w: 30,
            h: 30,
            data: vec![0.0; 30 * 30],
        };

        let cx = 10usize;
        let cy = 10usize;
        let w = resp.w;

        resp.data[cy * w + cx] = 10.0;
        resp.data[cy * w + (cx + 1)] = 1.0;
        resp.data[(cy + 1) * w + cx] = 1.0;

        let mut params = ChessParams {
            threshold_abs: Some(0.5),
            threshold_rel: 0.0,
            ..Default::default()
        };

        let baseline = detect_corners_from_response(&resp, &params);
        assert_eq!(baseline.len(), 1, "expected baseline detection");

        params.refiner = RefinerKind::CenterOfMass(CenterOfMassConfig { radius: 4 });
        let shrunk = detect_corners_from_response(&resp, &params);
        assert!(
            shrunk.is_empty(),
            "larger refiner radius should increase border and skip the corner"
        );
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
