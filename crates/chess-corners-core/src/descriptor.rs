//! Corner descriptors and helpers for chessboard detection.
//!
//! This module turns raw ChESS corner candidates into richer
//! [`CornerDescriptor`] values that carry subpixel position,
//! response, and orientation.
//!
//! The detector in [`crate::detect`] produces intermediate
//! [`Corner`] values; [`corners_to_descriptors`] then samples the
//! original image on a ChESS ring around each corner to estimate the
//! additional attributes using the conventions documented on
//! [`CornerDescriptor`].
use crate::ring::ring_offsets;
#[cfg(feature = "tracing")]
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
}

/// Convert raw corner candidates into full descriptors by sampling the source image.
///
/// Orientation follows the conventions documented on [`CornerDescriptor`].
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
    radius: u32,
    corners: Vec<Corner>,
) -> Vec<CornerDescriptor> {
    let ring = ring_offsets(radius);
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let samples = sample_ring(img, w, h, c.xy[0], c.xy[1], ring);
        let orientation = estimate_orientation_from_ring(&samples, ring);

        out.push(CornerDescriptor {
            x: c.xy[0],
            y: c.xy[1],
            response: c.strength,
            orientation,
        });
    }
    out
}

/// Sample the 16-point ChESS ring around (x, y) using bilinear interpolation.
fn sample_ring(
    img: &[u8],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    ring: &[(i32, i32); 16],
) -> [f32; 16] {
    let mut samples = [0.0f32; 16];
    for (i, &(dx, dy)) in ring.iter().enumerate() {
        let sx = x + dx as f32;
        let sy = y + dy as f32;
        samples[i] = sample_bilinear(img, w, h, sx, sy);
    }
    samples
}

#[inline]
fn estimate_orientation_from_ring(samples: &[f32; 16], ring: &[(i32, i32); 16]) -> f32 {
    // Same logic you use now: 2nd harmonic over sample index.
    let mean = samples.iter().copied().sum::<f32>() / 16.0;

    let mut c2 = 0.0f32;
    let mut s2 = 0.0f32;

    for (&v_raw, &(dx_i, dy_i)) in samples.iter().zip(ring.iter()) {
        let v = v_raw - mean;
        let angle = (dy_i as f32).atan2(dx_i as f32);
        let a2 = 2.0 * angle;
        c2 += v * a2.cos();
        s2 += v * a2.sin();
    }

    let mut theta = 0.5 * s2.atan2(c2);
    if theta < 0.0 {
        theta += core::f32::consts::PI;
    }
    if !theta.is_finite() {
        theta = 0.0;
    }

    theta
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
