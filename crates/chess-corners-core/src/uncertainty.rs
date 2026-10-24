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
    /// No position covariance. Localizers skip the noise propagation and
    /// the systematic model, and the detector makes no noise estimate.
    /// Positions are the same as under every other tier.
    Off,
    /// Image noise propagated through the localizer to first order, plus
    /// the localizer's systematic-error model (from the corner's offset to
    /// the localizer's fit window and the width of its edges). The
    /// default.
    #[default]
    Propagated,
}

impl Uncertainty {
    /// Whether this tier reports a covariance.
    #[inline]
    pub(crate) fn reports(self) -> bool {
        !matches!(self, Uncertainty::Off)
    }
}

/// Lower bound on the pixel-noise variance used for a covariance, in gray
/// level²: the variance of rounding to 8 bits.
pub(crate) const MIN_NOISE_VAR: f32 = 1.0 / 12.0;

/// `max(σₙ², 1/12)·C + Σ_sys`.
#[inline]
pub(crate) fn total_covariance(
    per_noise_variance: PositionCovariance,
    systematic: PositionCovariance,
    noise_sigma: f32,
) -> PositionCovariance {
    let var = (noise_sigma * noise_sigma).max(MIN_NOISE_VAR);
    PositionCovariance::new(
        var * per_noise_variance.xx + systematic.xx,
        var * per_noise_variance.xy + systematic.xy,
        var * per_noise_variance.yy + systematic.yy,
    )
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
        let g = &mut self.grad[j as usize * self.side + i as usize];
        g[0] += v[0];
        g[1] += v[1];
    }

    /// `Σₚ gₚ gₚᵀ`, or `None` when it is not finite.
    pub(crate) fn covariance(&self) -> Option<PositionCovariance> {
        let (mut xx, mut xy, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for g in &self.grad[..self.side * self.side] {
            let (gx, gy) = (f64::from(g[0]), f64::from(g[1]));
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
        let cov = PositionCovariance::new(xx as f32, xy as f32, yy as f32);
        [cov.xx, cov.xy, cov.yy]
            .iter()
            .all(|v| v.is_finite())
            .then_some(cov)
    }
}

// ---------------------------------------------------------------------------
// Systematic model
// ---------------------------------------------------------------------------

/// Radius of the disk [`edge_width`] reads, px.
const EDGE_DISK_RADIUS: i32 = 5;

/// Width of the edges around pixel `centre`, in px: the intensity range
/// over the disk of radius 5 px divided by `√(2π)` times the largest
/// central-difference gradient magnitude on it. For a straight edge blurred
/// by a Gaussian of σ that is about σ (pixel integration and the difference
/// stencil widen it by about 0.6 px at σ = 0); it grows with lens blur up
/// to σ ≈ 2.5 px, where the disk stops reaching the flat parts of the
/// cells, and does not depend on contrast. `None` on a flat patch.
pub(crate) fn edge_width(img: &ImageView<'_>, centre: [i32; 2]) -> Option<f32> {
    let r = EDGE_DISK_RADIUS;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut g2_max = 0.0f32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            let (x, y) = (centre[0] + dx, centre[1] + dy);
            let v = img.sample(x, y);
            lo = lo.min(v);
            hi = hi.max(v);
            let gx = 0.5 * (img.sample(x + 1, y) - img.sample(x - 1, y));
            let gy = 0.5 * (img.sample(x, y + 1) - img.sample(x, y - 1));
            g2_max = g2_max.max(gx * gx + gy * gy);
        }
    }
    (g2_max > 0.0).then(|| (hi - lo) / ((2.0 * core::f32::consts::PI).sqrt() * g2_max.sqrt()))
}

/// One knot of a [`SystematicModel`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Knot {
    /// [`edge_width`] at the knot, px.
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

    /// `Σ_sys` for an estimate at `offset` px from the centre of its fit
    /// window on a grid of spacing `grid` px, with edges of width `width`.
    pub(crate) fn covariance(
        &self,
        offset: [f32; 2],
        width: Option<f32>,
        grid: f32,
    ) -> PositionCovariance {
        let (g, rho) = self.at(width);
        let prior = grid * grid / 12.0;
        let rho2 = rho * rho;
        let denom = g * g * prior + rho2;
        if denom <= 0.0 {
            let f2 = self.floor * self.floor;
            return PositionCovariance::new(f2, 0.0, f2);
        }
        let shrink = g * prior / denom;
        let spread = prior * rho2 / denom;
        let spread = spread + self.floor * self.floor;
        let e = offset.map(|u| u - shrink * u);
        PositionCovariance::new(e[0] * e[0] + spread, e[0] * e[1], e[1] * e[1] + spread)
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
        // g = 0.7: the error is the linear pull (1 − 1/g)·û, the spread ≈ ρ/g.
        let c = MODEL.covariance([0.21, 0.0], Some(1.0), 1.0);
        let pull = 0.21 * (1.0 - 1.0 / 0.7);
        assert!(
            (c.xx - (pull * pull + (0.01f32 / 0.7).powi(2))).abs() < 2e-4,
            "{c:?}"
        );
        assert!((c.yy - (0.01f32 / 0.7).powi(2)).abs() < 1e-5, "{c:?}");
        // g = 0: the estimate carries no information about the offset; the
        // cell's own variance plus the observed offset.
        let c = MODEL.covariance([0.05, 0.0], Some(10.0), 0.5);
        let p = 0.25f32 / 12.0;
        assert!(
            (c.xx - (0.0025 + p)).abs() < 1e-6 && (c.yy - p).abs() < 1e-6,
            "{c:?}"
        );
        // Halfway between the knots.
        assert_eq!(MODEL.at(Some(2.0)), (0.35, 0.015));
        assert_eq!(MODEL.at(None), (0.7, 0.01));
    }

    #[test]
    fn edge_width_tracks_blur_not_contrast() {
        // A vertical step edge through the patch, blurred analytically.
        let side = 15usize;
        let edge = |sigma: f32, contrast: f32| {
            let data: Vec<u8> = (0..side * side)
                .map(|k| {
                    let x = (k % side) as f32 - 7.3;
                    let t = x / (sigma * core::f32::consts::SQRT_2);
                    // erf via tanh approximation is enough for a fixture.
                    let erf =
                        (core::f32::consts::FRAC_2_SQRT_PI * t * (1.0 + 0.089_431 * t * t)).tanh();
                    (100.0 + 0.5 * contrast * erf).round() as u8
                })
                .collect();
            let img = ImageView::from_u8_slice(side, side, &data).unwrap();
            edge_width(&img, [7, 7]).unwrap()
        };
        let (w1, w2, w3) = (edge(1.0, 160.0), edge(2.0, 160.0), edge(1.0, 60.0));
        assert!(w1 > 0.9 && w1 < 1.4, "{w1}");
        assert!(w2 > 1.4 * w1, "{w2} vs {w1}");
        assert!((w3 - w1).abs() < 0.1, "{w3} vs {w1}");
    }
}
