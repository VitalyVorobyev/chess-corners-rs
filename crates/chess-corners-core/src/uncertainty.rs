//! Position-uncertainty stage: the tier selector, and the pieces every
//! localizer shares to report a covariance — the per-pixel gradient
//! accumulator for noise propagation, the edge-width observable, and the
//! systematic-error model.
//!
//! A localizer's position covariance is
//!
//! ```text
//! Σ = max(σₙ², 1/12)·C + Σ_sys
//! ```
//!
//! `C = Σₚ (∂x̂/∂Iₚ)(∂x̂/∂Iₚ)ᵀ` is the first-order noise propagation over
//! the input pixels the estimate depends on (per unit noise variance);
//! `Σ_sys` is the expected systematic error from the localizer's
//! [`SystematicModel`].

use crate::imageview::ImageView;
use crate::refine::PositionCovariance;
use serde::{Deserialize, Serialize};

/// Which position uncertainty the detector reports for each corner.
///
/// Detector-independent: it applies to every localizer (the ChESS
/// refiners and the Radon peak fit) the same way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Uncertainty {
    /// No position covariance (the default). Localizers skip the noise
    /// propagation and the systematic model, and the detector makes no
    /// noise estimate. Positions are the same as under every other tier.
    #[default]
    Off,
    /// Image noise propagated through the localizer to first order, plus
    /// the localizer's systematic-error model (from the corner's offset to
    /// the localizer's fit window and the width of its edges).
    ///
    /// The systematic part includes a per-localizer floor measured on real
    /// calibration images (about 0.05 px per axis for the saddle-point fit
    /// and the Radon peak fit, 0.07 center of mass). On
    /// real, well-exposed images this floor is usually the largest part of
    /// the covariance, so the reported σ varies little from corner to
    /// corner there. Weighting a calibration by it has not been shown to
    /// beat uniform weighting on real data. Errors shared by all corners of
    /// an image (defocus, lens-model error) are outside it. The floor was
    /// fitted on one camera pair and is likely low for other optics. Within
    /// that pair the two cameras' own floors differed by up to 2×.
    Propagated,
}

impl Uncertainty {
    /// Whether this tier reports a covariance.
    #[inline]
    pub(crate) fn reports(self) -> bool {
        !matches!(self, Uncertainty::Off)
    }
}

/// Variance of a quantity spread uniformly over an interval `width` long:
/// `width²/12`.
pub(crate) const fn uniform_variance(width: f32) -> f32 {
    width * width / 12.0
}

/// Variance of rounding to integer gray levels (8-bit quantization), gray
/// level². It is also the least pixel-noise variance a covariance uses.
pub(crate) const ROUNDING_VAR: f32 = uniform_variance(1.0);

/// A localizer's position-covariance model: everything needed to express it
/// in px² for a given image noise level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CovarianceModel {
    /// First-order noise covariance per unit noise variance (px² per gray
    /// level²).
    pub noise: PositionCovariance,
    /// Systematic covariance, px².
    pub systematic: PositionCovariance,
    /// Second-order growth of the noise term per axis, per gray level²:
    /// axis `a` of the noise term is scaled by `1 + var·k_a²` (0 for a
    /// localizer whose first-order term is accurate).
    pub growth: [f32; 2],
}

impl CovarianceModel {
    /// The model with an image noise of `noise_sigma` gray levels, in px²:
    /// `max(σₙ², 1/12)·C` (each axis grown by its second-order factor) plus
    /// `Σ_sys`.
    pub(crate) fn total(&self, noise_sigma: f32) -> PositionCovariance {
        self.noise_at(noise_sigma) + self.systematic
    }

    /// The noise term alone at `noise_sigma`, in px². `noise_sigma` is the
    /// total noise of the 8-bit image, rounding included; its variance is
    /// taken as at least [`ROUNDING_VAR`].
    pub(crate) fn noise_at(&self, noise_sigma: f32) -> PositionCovariance {
        let var = (noise_sigma * noise_sigma).max(ROUNDING_VAR);
        let f = self.growth.map(|k| (1.0 + var * k * k).sqrt());
        let c = self.noise;
        PositionCovariance::new(
            var * c.xx * f[0] * f[0],
            var * c.xy * f[0] * f[1],
            var * c.yy * f[1] * f[1],
        )
    }
}

// ---------------------------------------------------------------------------
// Noise propagation
// ---------------------------------------------------------------------------

/// `∂x̂/∂Iₚ` of one estimate, accumulated over a square of input pixels.
///
/// A localizer adds every path from a pixel to its estimate (a pixel read
/// by several response samples, ray sums or gradients contributes once per
/// path), so pixels shared between paths correlate exactly; the noise
/// covariance per unit variance is then `Σₚ gₚ gₚᵀ`. The buffer is reused
/// across corners: it grows to the largest support seen and never shrinks.
#[derive(Debug, Default)]
pub(crate) struct PixelGradients {
    origin: [i32; 2],
    side: usize,
    grad: Vec<[f32; 2]>,
}

impl PixelGradients {
    /// Clear the accumulator for a `side × side` support whose top-left
    /// pixel is `origin` (in the caller's pixel frame).
    pub(crate) fn reset(&mut self, origin: [i32; 2], side: usize) {
        self.origin = origin;
        self.side = side;
        let n = side * side;
        if self.grad.len() < n {
            self.grad.resize(n, [0.0; 2]);
        }
        self.grad[..n].fill([0.0; 2]);
    }

    /// Add `v` to the gradient of pixel `(x, y)`. Pixels outside the
    /// support are a caller bug; they are ignored in release builds.
    #[inline]
    pub(crate) fn add(&mut self, x: i32, y: i32, v: [f32; 2]) {
        let (i, j) = (x - self.origin[0], y - self.origin[1]);
        debug_assert!(
            i >= 0 && j >= 0 && (i as usize) < self.side && (j as usize) < self.side,
            "pixel ({x}, {y}) outside the {}-px support at {:?}",
            self.side,
            self.origin
        );
        if i < 0 || j < 0 || i as usize >= self.side || j as usize >= self.side {
            return;
        }
        self.add_at(j as usize * self.side + i as usize, v);
    }

    /// Row stride of the support, for [`Self::add_at`] offsets.
    #[inline]
    pub(crate) fn side(&self) -> usize {
        self.side
    }

    /// Index of pixel `(x, y)` in the support, which must hold it; pixel
    /// `(x + dx, y + dy)` is at `index(x, y) + dy·side + dx`.
    #[inline]
    pub(crate) fn index(&self, x: i32, y: i32) -> usize {
        let (i, j) = ((x - self.origin[0]) as usize, (y - self.origin[1]) as usize);
        debug_assert!(i < self.side && j < self.side);
        j * self.side + i
    }

    /// Add `v` to the gradient of the pixel at support index `k` (see
    /// [`Self::index`]).
    #[inline]
    pub(crate) fn add_at(&mut self, k: usize, v: [f32; 2]) {
        let g = &mut self.grad[k];
        g[0] += v[0];
        g[1] += v[1];
    }

    /// `Σₚ gₚ gₚᵀ`, or `None` when it is not finite.
    pub(crate) fn covariance(&self) -> Option<PositionCovariance> {
        gram(&self.grad[..self.side * self.side])
    }
}

/// `Σ g gᵀ` over `grads`, or `None` when it is not finite.
pub(crate) fn gram(grads: &[[f32; 2]]) -> Option<PositionCovariance> {
    // Four partial sums, so the additions do not wait on one chain. Each
    // product of two `f32` is exact in `f64`, and the sums carry 29 more
    // bits than the `f32` result: the order moves the result by far less
    // than its last bit.
    let mut acc = [[0.0f64; 3]; 4];
    let (chunks, rest) = grads.as_chunks::<4>();
    let mut add = |l: usize, g: &[f32; 2]| {
        let (gx, gy) = (f64::from(g[0]), f64::from(g[1]));
        acc[l][0] += gx * gx;
        acc[l][1] += gx * gy;
        acc[l][2] += gy * gy;
    };
    for c in chunks {
        for (l, g) in c.iter().enumerate() {
            add(l, g);
        }
    }
    for g in rest {
        add(0, g);
    }
    let total = |k: usize| ((acc[0][k] + acc[1][k]) + (acc[2][k] + acc[3][k])) as f32;
    PositionCovariance::new(total(0), total(1), total(2)).finite()
}

// ---------------------------------------------------------------------------
// Pixel tile and edge width
// ---------------------------------------------------------------------------

/// Radius of the disk [`PixelTile::edge_width`] reads, px.
const EDGE_DISK_RADIUS: i32 = 5;

/// Half-size of the square [`PixelTile::edge_width`] needs around its
/// centre: the disk, one pixel for the gradient and two for the smoothing.
pub(crate) const EDGE_REACH: i32 = EDGE_DISK_RADIUS + 3;

/// The pixels of the radius-[`EDGE_DISK_RADIUS`] disk in scan order (rows
/// top to bottom, each left to right), as flat indices into the
/// `(2·radius + 3)²` smoothed square [`PixelTile::edge_width`] works on.
const EDGE_DISK: [u8; edge_disk_len()] = {
    const R: i32 = EDGE_DISK_RADIUS;
    const N: i32 = 2 * (R + 1) + 1;
    let mut disk = [0u8; edge_disk_len()];
    let (mut m, mut dy) = (0, -R);
    while dy <= R {
        let mut dx = -R;
        while dx <= R {
            if dx * dx + dy * dy <= R * R {
                disk[m] = ((dy + R + 1) * N + dx + R + 1) as u8;
                m += 1;
            }
            dx += 1;
        }
        dy += 1;
    }
    disk
};

/// Number of pixels in the [`EDGE_DISK`].
const fn edge_disk_len() -> usize {
    const R: i32 = EDGE_DISK_RADIUS;
    let (mut n, mut dy) = (0, -R);
    while dy <= R {
        let mut dx = -R;
        while dx <= R {
            if dx * dx + dy * dy <= R * R {
                n += 1;
            }
            dx += 1;
        }
        dy += 1;
    }
    n
}

/// A square of image pixels around a corner, copied once (clamped at the
/// image border like [`ImageView::sample`]) so a localizer's covariance
/// and the edge width read it without per-pixel bounds handling. The
/// buffers are reused across corners.
#[derive(Debug, Default)]
pub(crate) struct PixelTile {
    origin: [i32; 2],
    side: usize,
    data: Vec<f32>,
}

impl PixelTile {
    /// Copy the `(2·reach + 1)²` pixels of `img` centred on `centre` (in
    /// the view's external frame).
    pub(crate) fn load(&mut self, img: &ImageView<'_>, centre: [i32; 2], reach: i32) {
        let side = (2 * reach + 1) as usize;
        self.origin = [centre[0] - reach, centre[1] - reach];
        self.side = side;
        self.data.clear();
        let (w, h) = (img.width as i32, img.height as i32);
        let (gx0, gy0) = (
            self.origin[0] + img.origin[0],
            self.origin[1] + img.origin[1],
        );
        let inside = gx0 >= 0 && gy0 >= 0 && gx0 + side as i32 <= w && gy0 + side as i32 <= h;
        if inside {
            for row in 0..side {
                let start = (gy0 as usize + row) * img.width + gx0 as usize;
                self.data
                    .extend(img.data[start..start + side].iter().map(|&v| f32::from(v)));
            }
        } else {
            for j in 0..side as i32 {
                for i in 0..side as i32 {
                    self.data
                        .push(img.sample(self.origin[0] + i, self.origin[1] + j));
                }
            }
        }
    }

    /// Pixel `(x, y)` (external frame); must lie in the tile.
    #[inline]
    pub(crate) fn at(&self, x: i32, y: i32) -> f32 {
        self.data[self.index(x, y)]
    }

    /// Index of pixel `(x, y)` (external frame) in [`Self::data`], which
    /// must hold it; pixel `(x + dx, y + dy)` is at
    /// `index(x, y) + dy·side + dx`.
    #[inline]
    pub(crate) fn index(&self, x: i32, y: i32) -> usize {
        let (i, j) = ((x - self.origin[0]) as usize, (y - self.origin[1]) as usize);
        debug_assert!(i < self.side && j < self.side);
        j * self.side + i
    }

    /// Bilinear interpolation at `(x, y)` (external frame), with the
    /// arithmetic of [`ImageView::sample_bilinear`]; the four pixels around
    /// the point must lie in the tile.
    #[inline]
    pub(crate) fn bilinear(&self, x: f32, y: f32) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (k, s) = (self.index(x0 as i32, y0 as i32), self.side);
        let d = &self.data;
        let a = d[k] + (d[k + 1] - d[k]) * tx;
        let b = d[k + s] + (d[k + s + 1] - d[k + s]) * tx;
        a + (b - a) * ty
    }

    /// The tile's pixels, row-major, [`Self::side`] per row.
    #[inline]
    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }

    /// Row stride of [`Self::data`].
    #[inline]
    pub(crate) fn side(&self) -> usize {
        self.side
    }

    /// Width of the edges around pixel `centre`, in px, robust to pixel
    /// noise: on the image smoothed by a 5×5 binomial kernel, the contrast
    /// across the radius-5 disk (mean of the samples within a fifth of the
    /// range of its maximum, minus the same at its minimum) divided by
    /// `√(2π)` times the mean of the eight largest gradient magnitudes on
    /// the disk. It grows with the lens blur
    /// (on camera-formed boards about 1.15 px at σ = 0.5, 1.4 at 1.5, 1.5 at
    /// 2.5; the smoothing, pixel integration and difference stencil set the
    /// floor) and does not depend on contrast; at a contrast of 60 gray
    /// levels, noise of σ ≤ 8 moves it by at most 6% (up to 15% at σ = 8 with a
    /// lens blur of 2.5). Needs
    /// [`EDGE_REACH`] of tile around `centre`. `None` on a flat patch.
    pub(crate) fn edge_width(&self, centre: [i32; 2]) -> Option<f32> {
        const R: i32 = EDGE_DISK_RADIUS;
        // The smoothed square covers the disk plus one pixel for the
        // gradient; the raw square two more for the kernel.
        const N: usize = (2 * (R + 1) + 1) as usize;
        const RAW: usize = N + 4;
        let reach = R + 3;
        let x0 = (centre[0] - reach - self.origin[0]) as usize;
        let y0 = (centre[1] - reach - self.origin[1]) as usize;
        debug_assert!(x0 + RAW <= self.side && y0 + RAW <= self.side);
        // Separable [1 4 6 4 1] smoothing, unnormalised: the width is a
        // ratio of a contrast to a gradient, so the scale cancels. Pixels
        // are integers and the weights sum to 16 per pass, so every sum is an
        // integer below 2¹⁶, exact in `f32` in any order: down the columns,
        // then across.
        let raw = |j: usize| -> &[f32; RAW] {
            let start = (y0 + j) * self.side + x0;
            self.data[start..start + RAW]
                .try_into()
                .expect("the tile holds the edge-width square")
        };
        let binomial = |a: f32, b: f32, c: f32, d: f32, e: f32| (a + e) + 4.0 * (b + d) + 6.0 * c;
        let mut down = [[0.0f32; RAW]; N];
        for (j, out) in down.iter_mut().enumerate() {
            let (r0, r1, r2, r3, r4) = (raw(j), raw(j + 1), raw(j + 2), raw(j + 3), raw(j + 4));
            for i in 0..RAW {
                out[i] = binomial(r0[i], r1[i], r2[i], r3[i], r4[i]);
            }
        }
        let mut sm = [0.0f32; N * N];
        for (out, row) in sm.as_chunks_mut::<N>().0.iter_mut().zip(&down) {
            for i in 0..N {
                out[i] = binomial(row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 4]);
            }
        }
        // The disk's samples in scan order: value and squared gradient
        // (central difference without the ½).
        let mut values = [0.0f32; EDGE_DISK.len()];
        let mut grads = [0.0f32; EDGE_DISK.len()];
        for (m, &i) in EDGE_DISK.iter().enumerate() {
            let i = usize::from(i);
            values[m] = sm[i];
            let gx = sm[i + 1] - sm[i - 1];
            let gy = sm[i + N] - sm[i - N];
            grads[m] = gx * gx + gy * gy;
        }
        // Plateau levels: means of the samples within a fifth of the range
        // of the extremes; ridge gradient: mean of the eight largest
        // magnitudes. Averages, so noise moves them little. The values are
        // integers and their sum stays below 2²⁴, so the extremes and sums
        // are exact in any order: four lanes at a time.
        let (mut vmin, mut vmax) = ([f32::INFINITY; 4], [f32::NEG_INFINITY; 4]);
        let (chunks, rest) = values.as_chunks::<4>();
        for c in chunks {
            for l in 0..4 {
                vmin[l] = vmin[l].min(c[l]);
                vmax[l] = vmax[l].max(c[l]);
            }
        }
        let vmin = vmin
            .iter()
            .chain(rest)
            .fold(f32::INFINITY, |a, &v| a.min(v));
        let vmax = vmax
            .iter()
            .chain(rest)
            .fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let band = 0.2 * (vmax - vmin);
        let (top_cut, low_cut) = (vmax - band, vmin + band);
        let (mut hi, mut nh, mut lo, mut nl) = ([0.0f32; 4], [0u32; 4], [0.0f32; 4], [0u32; 4]);
        let mut tally = |l: usize, v: f32| {
            let (h, w) = (v >= top_cut, v <= low_cut);
            hi[l] += if h { v } else { 0.0 };
            nh[l] += u32::from(h);
            lo[l] += if w { v } else { 0.0 };
            nl[l] += u32::from(w);
        };
        for c in chunks {
            for (l, &v) in c.iter().enumerate() {
                tally(l, v);
            }
        }
        for &v in rest {
            tally(0, v);
        }
        let sum4 = |v: [f32; 4]| (v[0] + v[1]) + (v[2] + v[3]);
        let (hi, lo) = (sum4(hi), sum4(lo));
        let (nh, nl): (u32, u32) = (nh.iter().sum(), nl.iter().sum());
        // The eight largest squared gradients; roots of those only.
        let m = grads.len();
        let k = 8;
        grads.select_nth_unstable_by(m - k - 1, |a, b| a.total_cmp(b));
        let gs: f32 = grads[m - k..].iter().map(|g2| g2.sqrt()).sum();
        let ng = k as u32;
        let (high, low) = (hi / nh.max(1) as f32, lo / nl.max(1) as f32);
        // Central difference without the ½: halve.
        let g = 0.5 * gs / ng.max(1) as f32;
        (g > 0.0).then(|| (high - low) / ((2.0 * core::f32::consts::PI).sqrt() * g))
    }
}

/// [`PixelTile::edge_width`] of `img` around `centre`, through `tile`.
pub(crate) fn edge_width(
    img: &ImageView<'_>,
    centre: [i32; 2],
    tile: &mut PixelTile,
) -> Option<f32> {
    tile.load(img, centre, EDGE_REACH);
    tile.edge_width(centre)
}

// ---------------------------------------------------------------------------
// Systematic model
// ---------------------------------------------------------------------------

/// One knot of a [`SystematicModel`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Knot {
    /// [`PixelTile::edge_width`] at the knot, px.
    pub width: f32,
    /// Window-offset gain `g`: a noise-free estimate lands at `g·u` from the
    /// centre of its fit window for a corner at `u`.
    pub gain: f32,
    /// RMS per axis of the systematic error the linear model leaves, px.
    pub residual: f32,
}

/// A [`Knot`] at edge width `width` with window-offset gain `gain` and
/// residual RMS `residual` px.
pub(crate) const fn knot(width: f32, gain: f32, residual: f32) -> Knot {
    Knot {
        width,
        gain,
        residual,
    }
}

/// Systematic error of one localizer as a function of the corner's offset
/// from the localizer's fit window and the [`edge_width`].
///
/// A localizer whose fit window is centred on a grid point (the rounded
/// seed, or a working-resolution pixel) is pulled toward that point: for a
/// corner at offset `u` from it the noise-free estimate lands near `g·u`,
/// with `g` set by the edge width, plus a residual of RMS `ρ` per axis. The
/// observed offset `û` and the prior that `u` is uniform over one grid cell
/// (variance `h²/12` per axis, grid spacing `h`) give, to second order, the
/// expected error `û − ū` and its spread `V` about it:
///
/// ```text
/// ū = g·P·û / (g²·P + ρ²),   V = P·ρ² / (g²·P + ρ²),   P = h²/12,
/// Σ_sys = (û − ū)(û − ū)ᵀ + V·I.
/// ```
///
/// With `g` near 1 and small `ρ` the term is small; as `g` approaches 0 the
/// estimate no longer tells where in the cell the corner is, and `Σ_sys`
/// approaches the cell's own variance `P·I`. A localizer that re-centres its
/// window on its estimate has `û = 0` and `g = 1`, so `Σ_sys = ρ²·I`.
///
/// The knots come from synthetic boards, whose corners are straight-edged
/// and point-symmetric. Real optics and targets add error those boards do
/// not have (asymmetric blur, shading, print); `floor` covers it, as
/// `floor²·I` added to `Σ_sys`. It is the value that whitens the
/// localizer's error on a real stereo calibration set to unit variance after
/// removing what is not the localizer's: the lens-model and target terms,
/// and a smooth per-view pattern common to every localizer (defocus,
/// depth), which no single-corner covariance can carry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SystematicModel {
    /// Knots by increasing edge width; linear in between, clamped outside.
    pub knots: &'static [Knot],
    /// Isotropic floor per axis, px, for the error of real images beyond
    /// the knots' synthetic model.
    pub floor: f32,
}

impl SystematicModel {
    /// `(g, ρ)` at edge width `width` (the first knot when it is unknown).
    fn at(&self, width: Option<f32>) -> (f32, f32) {
        let k = self.knots;
        let Some(w) = width.filter(|w| w.is_finite()) else {
            return (k[0].gain, k[0].residual);
        };
        if w <= k[0].width {
            return (k[0].gain, k[0].residual);
        }
        for pair in k.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if w <= b.width {
                let t = (w - a.width) / (b.width - a.width);
                return (
                    a.gain + t * (b.gain - a.gain),
                    a.residual + t * (b.residual - a.residual),
                );
            }
        }
        let last = k[k.len() - 1];
        (last.gain, last.residual)
    }

    /// The covariance model of an estimate with noise term `noise`: `Σ_sys`
    /// for an estimate at `offset` px from the centre of its fit window on a
    /// grid of spacing `grid` px, with edges of width `width`.
    pub(crate) fn model(
        &self,
        noise: PositionCovariance,
        offset: [f32; 2],
        width: Option<f32>,
        grid: f32,
    ) -> CovarianceModel {
        let (g, rho) = self.at(width);
        let prior = uniform_variance(grid);
        let rho2 = rho * rho;
        let denom = g * g * prior + rho2;
        // g = ρ = 0: the estimate says nothing about where in the cell the
        // corner is; the limit of the general expressions.
        let (error, spread) = if denom <= 0.0 {
            (offset, prior)
        } else {
            let shrink = g * prior / denom;
            (offset.map(|u| u - shrink * u), prior * rho2 / denom)
        };
        let [ex, ey] = error;
        let spread = spread + self.floor * self.floor;
        let systematic = PositionCovariance::new(ex * ex + spread, ex * ey, ey * ey + spread);
        CovarianceModel {
            noise,
            systematic,
            growth: [0.0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: SystematicModel = SystematicModel {
        knots: &[
            Knot {
                width: 1.0,
                gain: 0.7,
                residual: 0.01,
            },
            Knot {
                width: 3.0,
                gain: 0.0,
                residual: 0.02,
            },
        ],
        floor: 0.0,
    };

    #[test]
    fn systematic_model_limits() {
        let sys = |offset, width, grid| {
            MODEL
                .model(
                    PositionCovariance::new(0.0, 0.0, 0.0),
                    offset,
                    Some(width),
                    grid,
                )
                .systematic
        };
        // g = 0.7: the error is the linear pull (1 − 1/g)·û, the spread ≈ ρ/g.
        let c = sys([0.21, 0.0], 1.0, 1.0);
        let pull = 0.21 * (1.0 - 1.0 / 0.7);
        assert!(
            (c.xx - (pull * pull + (0.01f32 / 0.7).powi(2))).abs() < 2e-4,
            "{c:?}"
        );
        assert!((c.yy - (0.01f32 / 0.7).powi(2)).abs() < 1e-5, "{c:?}");
        // g = 0: the estimate carries no information about the offset; the
        // cell's own variance plus the observed offset.
        let c = sys([0.05, 0.0], 10.0, 0.5);
        let p = 0.25f32 / 12.0;
        assert!(
            (c.xx - (0.0025 + p)).abs() < 1e-6 && (c.yy - p).abs() < 1e-6,
            "{c:?}"
        );
        // Halfway between the knots.
        assert_eq!(MODEL.at(Some(2.0)), (0.35, 0.015));
        assert_eq!(MODEL.at(None), (0.7, 0.01));
    }

    /// Width at the centre corner of a camera-formed 9 px board, mean over
    /// `draws` noise draws (σ `noise`), with `dark`/`bright` levels.
    fn board_width(blur: f32, dark: u8, bright: u8, noise: f32, draws: u64) -> f32 {
        use chess_corners_testutil::{add_gaussian_noise, optical_chessboard, Lattice};
        let side = 41usize;
        let lattice = Lattice::new(9.0, (20.3, 19.8)).rotated(0.2);
        let clean = optical_chessboard(side, &lattice, blur, dark, bright);
        let mut tile = PixelTile::default();
        let mut sum = 0.0;
        for d in 0..draws {
            let mut img = clean.clone();
            add_gaussian_noise(&mut img, noise, 100 + d);
            let view = ImageView::from_u8_slice(side, side, &img).unwrap();
            sum += edge_width(&view, [20, 20], &mut tile).unwrap();
        }
        sum / draws as f32
    }

    #[test]
    fn edge_width_tracks_blur_not_contrast() {
        let (w05, w15, w25) = (
            board_width(0.5, 40, 200, 0.0, 1),
            board_width(1.5, 40, 200, 0.0, 1),
            board_width(2.5, 40, 200, 0.0, 1),
        );
        assert!(w15 > 1.15 * w05 && w25 > 1.05 * w15, "{w05} {w15} {w25}");
        let low = board_width(1.5, 100, 160, 0.0, 1);
        assert!((low - w15).abs() < 0.05 * w15, "{low} vs {w15}");
    }

    /// Pixel noise barely moves the width, even at low contrast (60 gray
    /// levels): within 6% of the noise-free value up to σ = 8 at lens blur
    /// ≤ 1.5 and up to σ = 4 at 2.5; 15% at σ = 8 and blur 2.5.
    #[test]
    fn edge_width_is_stable_under_noise() {
        for blur in [0.5f32, 1.5, 2.5] {
            let clean = board_width(blur, 100, 160, 0.0, 1);
            for noise in [1.0f32, 2.0, 4.0, 8.0] {
                let noisy = board_width(blur, 100, 160, noise, 16);
                let tol = if blur > 2.0 && noise > 4.0 {
                    0.15
                } else {
                    0.06
                };
                assert!(
                    (noisy - clean).abs() <= tol * clean,
                    "blur {blur} σ {noise}: {noisy} vs noise-free {clean}"
                );
            }
        }
    }
}
