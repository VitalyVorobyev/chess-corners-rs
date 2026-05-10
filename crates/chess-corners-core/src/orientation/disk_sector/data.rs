//! Disk data extraction: sampling pixel values and gradients from the
//! support disk around the candidate corner.

use core::f32::consts::PI;

use super::geometry::wrap_pi;

/// Physical size limits for the support disk.
pub(super) const MAX_SIDE: usize = 34;
pub(super) const MAX_SUPPORT: usize = MAX_SIDE * MAX_SIDE;

const SUPPORT_SCALE: f32 = 1.6;
const MAX_SUPPORT_RADIUS: f32 = 8.0;
const INNER_RADIUS: f32 = 1.0;
pub(super) const MIN_SUPPORT: usize = 64;

/// All pixel-level data sampled from the support disk.
#[derive(Clone)]
pub(super) struct DiskData {
    pub(super) n: usize,
    pub(super) xs: [f32; MAX_SUPPORT],
    pub(super) ys: [f32; MAX_SUPPORT],
    pub(super) vals: [f32; MAX_SUPPORT],
    pub(super) vals_centered: [f32; MAX_SUPPORT],
    pub(super) grad_angles: [f32; MAX_SUPPORT],
    pub(super) grad_weights: [f32; MAX_SUPPORT],
    pub(super) mean: f32,
    pub(super) val_energy: f32,
}

impl DiskData {
    pub(super) fn new() -> Self {
        Self {
            n: 0,
            xs: [0.0; MAX_SUPPORT],
            ys: [0.0; MAX_SUPPORT],
            vals: [0.0; MAX_SUPPORT],
            vals_centered: [0.0; MAX_SUPPORT],
            grad_angles: [0.0; MAX_SUPPORT],
            grad_weights: [0.0; MAX_SUPPORT],
            mean: 0.0,
            val_energy: 0.0,
        }
    }
}

/// Sample the support disk around `(cx, cy)` using `radius`. Returns
/// `None` if the disk clips the image boundary, the support pixel count
/// is below [`MIN_SUPPORT`], or any other input precondition fails.
pub(super) fn extract_disk(
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
) -> Option<DiskData> {
    if w == 0 || h == 0 || img.len() < w.saturating_mul(h) || !cx.is_finite() || !cy.is_finite() {
        return None;
    }

    let support_radius = (radius as f32 * SUPPORT_SCALE).min(MAX_SUPPORT_RADIUS);
    if support_radius < 1.0 {
        return None;
    }

    let x0 = (cx - support_radius).floor() as i32;
    let x1 = (cx + support_radius).ceil() as i32;
    let y0 = (cy - support_radius).floor() as i32;
    let y1 = (cy + support_radius).ceil() as i32;
    if x0 < 0 || y0 < 0 || x1 >= w as i32 || y1 >= h as i32 {
        return None;
    }
    if (x1 - x0 + 1) as usize > MAX_SIDE || (y1 - y0 + 1) as usize > MAX_SIDE {
        return None;
    }

    let mut data = DiskData::new();
    let r2 = support_radius * support_radius;
    let inner2 = INNER_RADIUS * INNER_RADIUS;
    let mut sum = 0.0f32;

    for yy in y0..=y1 {
        let dy = yy as f32 - cy;
        for xx in x0..=x1 {
            let dx = xx as f32 - cx;
            let rr = dx * dx + dy * dy;
            if rr > r2 || rr < inner2 {
                continue;
            }
            if data.n >= MAX_SUPPORT {
                return None;
            }
            let idx = yy as usize * w + xx as usize;
            let val = img[idx] as f32;
            let (gx, gy) = sobel_at(img, w, h, xx, yy);
            let weight = (gx * gx + gy * gy).sqrt();
            data.xs[data.n] = dx;
            data.ys[data.n] = dy;
            data.vals[data.n] = val;
            data.grad_weights[data.n] = weight;
            data.grad_angles[data.n] = wrap_pi(gy.atan2(gx) + PI * 0.5);
            sum += val;
            data.n += 1;
        }
    }

    if data.n < MIN_SUPPORT {
        return None;
    }

    data.mean = sum / data.n as f32;
    let mut val_energy = 0.0f32;
    for i in 0..data.n {
        let centered = data.vals[i] - data.mean;
        data.vals_centered[i] = centered;
        val_energy += centered * centered;
    }
    data.val_energy = val_energy;
    Some(data)
}

/// Compute the 3×3 Sobel gradient at integer pixel `(x, y)`, clamping
/// at the image boundary.
pub(super) fn sobel_at(img: &[u8], w: usize, h: usize, x: i32, y: i32) -> (f32, f32) {
    let p = |xx: i32, yy: i32| -> f32 {
        let xc = xx.clamp(0, w.saturating_sub(1) as i32) as usize;
        let yc = yy.clamp(0, h.saturating_sub(1) as i32) as usize;
        img[yc * w + xc] as f32
    };

    let tl = p(x - 1, y - 1);
    let tc = p(x, y - 1);
    let tr = p(x + 1, y - 1);
    let ml = p(x - 1, y);
    let mr = p(x + 1, y);
    let bl = p(x - 1, y + 1);
    let bc = p(x, y + 1);
    let br = p(x + 1, y + 1);

    let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    (gx, gy)
}
