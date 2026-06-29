//! Corner merging: spatial-grid and naive-fallback implementations.
use super::Corner;

#[cfg(feature = "tracing")]
use tracing::instrument;

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
