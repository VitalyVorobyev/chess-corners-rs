//! Corner descriptor that can be used for chessboard detection
//! #[cfg(feature = "tracing")]
use tracing::instrument;

/// A detected ChESS corner (subpixel).
#[derive(Clone, Debug)]
pub struct Corner {
    /// Subpixel location in image coordinates (x, y).
    pub xy: [f32; 2],
    /// Raw ChESS response at the integer peak (before COM refinement).
    pub strength: f32,
}

/// Describes a detected chessboard corner in full-resolution image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct CornerDescriptor {
    /// Subpixel position in full-resolution image pixels.
    pub x: f32,
    pub y: f32,

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

/// Convert raw corner candidates into full descriptors by sampling the source image.
///
/// Orientation, phase, and anisotropy follow the conventions documented on
/// [`CornerDescriptor`].
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "info",
        skip(img, corners),
        fields(corners = corners.len())
    )
)]
pub fn corners_to_descriptors(
    img: &[u8],
    w: usize,
    h: usize,
    corners: Vec<Corner>,
) -> Vec<CornerDescriptor> {
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let (orientation, anisotropy) =
            estimate_corner_orientation_anisotropy(img, w, h, c.xy[0], c.xy[1]);
        let phase = estimate_corner_phase(img, w, h, c.xy[0], c.xy[1], orientation);

        out.push(CornerDescriptor {
            x: c.xy[0],
            y: c.xy[1],
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
