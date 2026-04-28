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

    // The paper's acceptance criterion is "R > 0", so we use a strict
    // comparison below. With `threshold_abs = Some(0.0)` (the default)
    // this reduces to "accept any strictly positive response", which is
    // the paper's contract.

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
            if v <= thr {
                continue;
            }

            // Local maximum in NMS window
            if !is_local_max(resp.data(), resp.w, resp.h, x, y, nms_r, v) {
                continue;
            }

            // Reject isolated pixels: require a minimum number of positive
            // neighbors in the same NMS window.
            let cluster_size = count_positive_neighbors(resp.data(), resp.w, resp.h, x, y, nms_r);
            if cluster_size < params.min_cluster_size {
                continue;
            }

            let seed_xy = [x as f32, y as f32];
            let res = refiner.refine(seed_xy, ctx);

            if matches!(res.status, RefineStatus::Accepted) {
                corners.push(Corner {
                    x: res.x,
                    y: res.y,
                    strength: v,
                });
            }
        }
    }

    corners
}

/// Local-max NMS check over a `(2r+1)²` window on a row-major
/// response slice. Slice-based so borrowed views (e.g. the Radon
/// detector's working-resolution buffer) can call it without cloning
/// into a [`ResponseMap`].
pub(crate) fn is_local_max(
    data: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    r: i32,
    v: f32,
) -> bool {
    let wi = w as i32;
    let hi = h as i32;
    let cx = x as i32;
    let cy = y as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = cx + dx;
            let yy = cy + dy;
            if xx < 0 || yy < 0 || xx >= wi || yy >= hi {
                continue;
            }
            let vv = data[(yy as usize) * w + (xx as usize)];
            if vv > v {
                return false;
            }
        }
    }
    true
}

/// Count strictly-positive neighbors in the same window as
/// [`is_local_max`]. See that function for the slice contract.
pub(crate) fn count_positive_neighbors(
    data: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    r: i32,
) -> u32 {
    let wi = w as i32;
    let hi = h as i32;
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
            if xx < 0 || yy < 0 || xx >= wi || yy >= hi {
                continue;
            }
            let vv = data[(yy as usize) * w + (xx as usize)];
            if vv > 0.0 {
                count += 1;
            }
        }
    }

    count
}

/// Merge corners within a given radius, keeping the strongest response.
///
/// Uses a uniform spatial grid with cell size equal to `radius`, so any
/// two corners within `radius` of each other land in the same or
/// neighbouring cells. For `N` input corners the expected cost is
/// `O(N)` with small constants (vs the naive `O(N²)` pairwise scan),
/// which matters on Radon-detected frames where `N` can run into the
/// thousands.
///
/// The output is order-equivalent to the naive scan: when an incoming
/// corner is within `radius` of multiple existing corners, the
/// **first-seen** existing corner wins the merge — same as the
/// previous implementation. When the incoming corner is stronger, it
/// replaces that existing corner's position in-place.
#[cfg_attr(feature = "tracing", instrument(level = "info", skip(corners)))]
pub fn merge_corners_simple(corners: &mut Vec<Corner>, radius: f32) -> Vec<Corner> {
    let n = corners.len();
    if n == 0 {
        return Vec::new();
    }
    if !radius.is_finite() || radius <= 0.0 {
        return std::mem::take(corners);
    }

    let r2 = radius * radius;

    // Bounding box of the input corner cloud.
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for c in corners.iter() {
        if c.x < min_x {
            min_x = c.x;
        }
        if c.y < min_y {
            min_y = c.y;
        }
        if c.x > max_x {
            max_x = c.x;
        }
        if c.y > max_y {
            max_y = c.y;
        }
    }

    // Guard: any non-finite coordinate or zero-extent cloud falls back
    // to the naive scan rather than building a degenerate grid.
    if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
        return merge_corners_naive(corners, r2);
    }

    let cell = radius;
    let inv_cell = 1.0 / cell;
    let grid_w = ((max_x - min_x) * inv_cell).floor() as usize + 1;
    let grid_h = ((max_y - min_y) * inv_cell).floor() as usize + 1;

    // Each grid cell stores indices into `out`. `u32` is plenty for
    // realistic candidate counts and halves the per-cell footprint
    // vs `usize` on 64-bit targets.
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); grid_w * grid_h];
    let mut out: Vec<Corner> = Vec::with_capacity(n);

    let cell_of = |c: &Corner| -> (usize, usize) {
        let gx = (((c.x - min_x) * inv_cell) as usize).min(grid_w - 1);
        let gy = (((c.y - min_y) * inv_cell) as usize).min(grid_h - 1);
        (gx, gy)
    };

    for c in corners.drain(..) {
        let (gx, gy) = cell_of(&c);

        // Scan the 3×3 cell neighbourhood for the first existing
        // corner within `radius`. Iteration order matches the naive
        // scan (cells visited in (y, x) order; within a cell, indices
        // are stored in insertion order, which is the order
        // corners were pushed into `out`).
        let y0 = gy.saturating_sub(1);
        let y1 = (gy + 1).min(grid_h - 1);
        let x0 = gx.saturating_sub(1);
        let x1 = (gx + 1).min(grid_w - 1);

        let mut hit: Option<usize> = None;
        let mut best_idx = u32::MAX;
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                for &idx in &grid[ny * grid_w + nx] {
                    if idx >= best_idx {
                        continue;
                    }
                    let i = idx as usize;
                    let dx = c.x - out[i].x;
                    let dy = c.y - out[i].y;
                    if dx * dx + dy * dy <= r2 {
                        best_idx = idx;
                        hit = Some(i);
                    }
                }
            }
        }

        if let Some(i) = hit {
            if c.strength > out[i].strength {
                let (old_gx, old_gy) = cell_of(&out[i]);
                out[i] = c;
                let (new_gx, new_gy) = cell_of(&out[i]);
                if old_gx != new_gx || old_gy != new_gy {
                    let id = i as u32;
                    let old_cell = old_gy * grid_w + old_gx;
                    let new_cell = new_gy * grid_w + new_gx;
                    grid[old_cell].retain(|&j| j != id);
                    grid[new_cell].push(id);
                }
            }
        } else {
            let new_idx = out.len() as u32;
            let (gx, gy) = cell_of(&c);
            out.push(c);
            grid[gy * grid_w + gx].push(new_idx);
        }
    }

    out
}

/// Naive O(N²) merge, used only as a fallback when the spatial-grid
/// path can't build a valid bounding box (e.g. NaN / Inf coordinates).
fn merge_corners_naive(corners: &mut Vec<Corner>, r2: f32) -> Vec<Corner> {
    let mut out: Vec<Corner> = Vec::new();
    'outer: for c in corners.drain(..) {
        for o in &mut out {
            let dx = c.x - o.x;
            let dy = c.y - o.y;
            if dx * dx + dy * dy <= r2 {
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
    fn descriptors_report_two_axes_stable() {
        use core::f32::consts::{FRAC_PI_2, PI};

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

        // axes[0] in [0, π), axes[1] in (axes[0], axes[0] + π)
        assert!(best.axes[0].angle >= 0.0 && best.axes[0].angle < PI);
        assert!(
            best.axes[1].angle > best.axes[0].angle && best.axes[1].angle < best.axes[0].angle + PI
        );

        // The quadrant corner has one axis horizontal (line angle 0)
        // and one vertical (line angle π/2). Accept a generous tolerance
        // because the 32×32 synthetic image is aliased.
        let near_line = |x: f32, target: f32| -> f32 {
            let xr = x.rem_euclid(PI);
            let tr = target.rem_euclid(PI);
            let d = (xr - tr).abs();
            d.min(PI - d)
        };
        // One of the two axes matches horizontal (line 0), the other vertical (line π/2).
        let horiz = near_line(best.axes[0].angle, 0.0).min(near_line(best.axes[1].angle, 0.0));
        let vert =
            near_line(best.axes[0].angle, FRAC_PI_2).min(near_line(best.axes[1].angle, FRAC_PI_2));
        assert!(
            horiz < 0.35,
            "horiz line miss: {horiz}, axes {:?}",
            best.axes
        );
        assert!(vert < 0.35, "vert line miss: {vert}, axes {:?}", best.axes);
        assert!(best.contrast > 0.0);

        // Brightness shift stability: both axes survive a global
        // intensity offset.
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

        let da0 = near_line(best.axes[0].angle, best_brighter.axes[0].angle);
        let da1 = near_line(best.axes[1].angle, best_brighter.axes[1].angle);
        assert!(da0 < 0.35, "axis0 delta after brightness shift: {da0}");
        assert!(da1 < 0.35, "axis1 delta after brightness shift: {da1}");
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
        assert!((c.x - expected.x).abs() < 1e-6);
        assert!((c.y - expected.y).abs() < 1e-6);
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
                x: 10.0,
                y: 10.0,
                strength: 1.0,
            },
            Corner {
                x: 11.0,
                y: 11.0,
                strength: 5.0,
            },
            Corner {
                x: 20.0,
                y: 20.0,
                strength: 3.0,
            },
        ];
        let merged = merge_corners_simple(&mut corners, 2.5);
        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|c| (c.x - 11.0).abs() < 1e-6
            && (c.y - 11.0).abs() < 1e-6
            && (c.strength - 5.0).abs() < 1e-6));
        assert!(merged
            .iter()
            .any(|c| (c.x - 20.0).abs() < 1e-6 && (c.y - 20.0).abs() < 1e-6));
    }

    /// Cheap LCG so we can run a deterministic randomized equivalence
    /// test without pulling in `rand` as a dev-dep.
    fn lcg(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as u32
    }

    fn rand_corners(seed: u64, n: usize, span: f32) -> Vec<Corner> {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let rx = lcg(&mut state) as f32 / u32::MAX as f32;
            let ry = lcg(&mut state) as f32 / u32::MAX as f32;
            let rs = lcg(&mut state) as f32 / u32::MAX as f32;
            out.push(Corner {
                x: rx * span,
                y: ry * span,
                strength: rs * 100.0,
            });
        }
        out
    }

    fn corners_eq_unordered(a: &[Corner], b: &[Corner]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut used = vec![false; b.len()];
        'outer: for ca in a.iter() {
            for (i, cb) in b.iter().enumerate() {
                if used[i] {
                    continue;
                }
                if (ca.x - cb.x).abs() < 1e-5
                    && (ca.y - cb.y).abs() < 1e-5
                    && (ca.strength - cb.strength).abs() < 1e-3
                {
                    used[i] = true;
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    #[test]
    fn merge_corners_grid_matches_naive_scan() {
        // Equivalence check: the spatial-grid `merge_corners_simple`
        // must produce the same kept-corner set as the naive O(N²)
        // pairwise scan on randomized inputs. Order is allowed to
        // differ — both implementations are deterministic but the
        // grid visits cells in a different order.
        for seed in [1u64, 7, 42, 123, 1729] {
            for &n in &[16usize, 64, 256, 1024] {
                for &radius in &[1.0f32, 2.5, 6.0] {
                    let corners = rand_corners(seed, n, 50.0);
                    let mut a = corners.clone();
                    let mut b = corners.clone();
                    let r2 = radius * radius;
                    let merged_grid = merge_corners_simple(&mut a, radius);
                    let merged_naive = merge_corners_naive(&mut b, r2);
                    assert!(
                        corners_eq_unordered(&merged_grid, &merged_naive),
                        "mismatch at seed={seed}, n={n}, radius={radius}: grid={} naive={}",
                        merged_grid.len(),
                        merged_naive.len(),
                    );
                }
            }
        }
    }

    #[test]
    fn merge_corners_handles_empty_and_zero_radius() {
        let mut empty: Vec<Corner> = Vec::new();
        let merged = merge_corners_simple(&mut empty, 1.0);
        assert!(merged.is_empty());

        let mut few = vec![
            Corner {
                x: 1.0,
                y: 1.0,
                strength: 1.0,
            },
            Corner {
                x: 1.5,
                y: 1.5,
                strength: 2.0,
            },
        ];
        // Zero radius is a no-op merge — every input survives.
        let merged = merge_corners_simple(&mut few, 0.0);
        assert_eq!(merged.len(), 2);
    }
}
