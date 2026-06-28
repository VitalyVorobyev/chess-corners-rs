//! NMS helper predicates shared by the ChESS and Radon detectors.

/// Local-max NMS check over a `(2r+1)²` window on a row-major
/// response slice. Slice-based so borrowed views (e.g. the Radon
/// detector's working-resolution buffer) can call it without cloning
/// into a [`crate::ResponseMap`].
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
