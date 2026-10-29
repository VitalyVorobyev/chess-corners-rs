//! Pluggable subpixel-refinement backends.
//!
//! All refiners implement the [`CornerRefiner`] trait and are
//! addressed through the user-facing [`RefinerKind`] enum. Default
//! settings select the center-of-mass refiner on the response map.
//!
//! - [`center_of_mass`] — 5×5 weighted centroid on the response
//!   map. Cheap; the library default.
//! - [`forstner`] — gradient structure-tensor refinement on the image
//!   intensity patch.
//! - [`saddle_point`] — quadratic surface fit on the image patch.

use crate::imageview::ImageView;
use crate::uncertainty::{CovarianceModel, Uncertainty};
use crate::ResponseMap;
use serde::{Deserialize, Serialize};

pub mod center_of_mass;
pub mod forstner;
pub mod saddle_point;

pub use center_of_mass::{CenterOfMassConfig, CenterOfMassRefiner};
pub use forstner::{ForstnerConfig, ForstnerRefiner};
pub use saddle_point::{SaddlePointConfig, SaddlePointRefiner};

/// Symmetric 2×2 covariance of a refined corner position, in px² (px² per
/// gray-level² in [`RefineResult::noise_covariance_per_unit_variance`]).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PositionCovariance {
    /// Variance of the x coordinate.
    pub xx: f32,
    /// Covariance of the x and y coordinates.
    pub xy: f32,
    /// Variance of the y coordinate.
    pub yy: f32,
}

impl PositionCovariance {
    /// Construct a [`PositionCovariance`].
    #[inline]
    pub fn new(xx: f32, xy: f32, yy: f32) -> Self {
        Self { xx, xy, yy }
    }

    pub(crate) const ZERO: Self = Self {
        xx: 0.0,
        xy: 0.0,
        yy: 0.0,
    };

    /// Every entry multiplied by `k`.
    #[inline]
    pub(crate) fn scaled(self, k: f32) -> Self {
        Self::new(self.xx * k, self.xy * k, self.yy * k)
    }

    /// `self` when every entry is finite.
    #[inline]
    pub(crate) fn finite(self) -> Option<Self> {
        (self.xx.is_finite() && self.xy.is_finite() && self.yy.is_finite()).then_some(self)
    }
}

/// The covariance of the sum of two independent errors.
impl core::ops::Add for PositionCovariance {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.xx + rhs.xx, self.xy + rhs.xy, self.yy + rhs.yy)
    }
}

/// Sealing supertrait for [`CornerRefiner`]. Implemented only for the
/// in-crate refiners, so downstream crates cannot add their own
/// [`CornerRefiner`] implementations.
mod private {
    pub trait Sealed {}
}

/// Status of a refinement attempt.
///
/// A refiner returns one of these variants to indicate whether it
/// produced a usable subpixel location and, if not, why it did not.
///
/// - [`Accepted`](RefineStatus::Accepted) — refinement converged; use
///   `RefineResult::x` and `y`.
/// - [`Rejected`](RefineStatus::Rejected) — the refiner ran but the
///   result did not pass an acceptance criterion (e.g. the computed
///   displacement exceeded [`ForstnerConfig::max_offset`]). Fall back to
///   the original seed or skip this candidate.
/// - [`OutOfBounds`](RefineStatus::OutOfBounds) — the seed is too close
///   to the image border for the refiner's patch window. The seed
///   coordinates in `RefineResult` are unchanged from the input.
/// - [`IllConditioned`](RefineStatus::IllConditioned) — the patch does
///   not contain enough structure for the refiner to produce a reliable
///   estimate (e.g. no positive response in the center-of-mass window,
///   a too-flat gradient for Förstner, or a degenerate Hessian for
///   saddle-point). The seed coordinates are unchanged.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefineStatus {
    /// Refinement converged; the result's `x`/`y` are usable.
    Accepted,
    /// The refiner ran but the result failed its acceptance criterion.
    Rejected,
    /// The seed is too close to the image border for the refiner's patch window.
    OutOfBounds,
    /// The patch lacks enough structure for a reliable estimate.
    IllConditioned,
}

/// Result of refining a single corner candidate.
///
/// Coordinates are in the **input image pixel frame**, with the origin
/// at the top-left corner of the top-left pixel (i.e. pixel (0, 0) has
/// its center at `(0.0, 0.0)`). Both `x` and `y` are subpixel values
/// in pixels — no normalization or scaling is applied by the refiner
/// itself.
///
/// Check [`status`](RefineResult::status) before using `x` and `y`:
/// only [`RefineStatus::Accepted`] guarantees that the refiner moved
/// the position. For all other statuses the coordinates are copied
/// unchanged from the input seed.
///
/// The [`score`](RefineResult::score) field measures how well the local
/// image structure supports the refined position. Its meaning depends on
/// the active refiner:
///
/// - **CenterOfMass**: `score` is the sum of positive response weights
///   in the patch (`Σ w` where `w = response.clamp(0, ∞)`). Higher is
///   stronger; units match the ChESS response scale. A window with no
///   positive response yields [`RefineStatus::IllConditioned`] and a
///   score of `0.0`.
/// - **Förstner**: `score` is `det(T) / (trace(T)² + ε)` where `T` is
///   the structure tensor. Ranges roughly in `(0, 0.25]`; higher means
///   the patch has balanced gradient energy in both directions (closer to
///   a true corner). The value `0.0` is returned on all failure paths.
/// - **SaddlePoint**: `score` is `sqrt(|det(H)|)` where `H` is the
///   fitted quadratic Hessian. Larger magnitude indicates a sharper
///   saddle (steeper curvature); no absolute scale is defined.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct RefineResult {
    /// Refined subpixel x coordinate in input-image pixels. Valid only
    /// when [`status`](RefineResult::status) is
    /// [`RefineStatus::Accepted`]; otherwise equals the input seed x.
    pub x: f32,
    /// Refined subpixel y coordinate in input-image pixels. Valid only
    /// when [`status`](RefineResult::status) is
    /// [`RefineStatus::Accepted`]; otherwise equals the input seed y.
    pub y: f32,
    /// Refiner-specific quality score. Higher values indicate stronger
    /// evidence for the refined position. See the [`RefineResult`] docs
    /// for per-refiner definitions. Always `0.0` on `OutOfBounds` and
    /// may be `0.0` on other non-`Accepted` statuses.
    pub score: f32,
    /// Whether the refiner accepted, rejected, or could not process this
    /// candidate. See [`RefineStatus`].
    pub status: RefineStatus,
    /// First-order covariance of the refined position per unit image-noise
    /// variance, in px² per gray-level²: the noise part of the position
    /// covariance. [`position_covariance`](Self::position_covariance) scales
    /// it by the image's noise variance and adds the refiner's systematic
    /// term. A first-order approximation that assumes independent,
    /// identically distributed per-pixel noise; spatially correlated noise
    /// (upscaling, demosaicing, JPEG, blur applied after the noise) makes it
    /// underestimate the scatter.
    ///
    /// Set on accepted corners by every refiner under
    /// [`Uncertainty::Propagated`] (see [`Refiner::with_uncertainty`]): the saddle-point refiner when its
    /// re-centred refinement converged, the center-of-mass refiner when the
    /// image and a ChESS response map are both in the [`RefineContext`],
    /// and the Förstner refiner. `None` otherwise, and always under
    /// [`Uncertainty::Off`] (the default). Read it with
    /// [`RefineResult::noise_covariance_per_unit_variance`]. Changing this
    /// field drops the systematic term.
    #[deprecated(
        since = "1.7.0",
        note = "read `noise_covariance_per_unit_variance()`, or `position_covariance(noise_sigma)` for the full covariance in px²; writing this field drops the systematic term"
    )]
    pub covariance_per_noise_variance: Option<PositionCovariance>,
    /// The full covariance model; `covariance_per_noise_variance` mirrors
    /// its noise term.
    pub(crate) model: Option<CovarianceModel>,
    /// The refiner reached its own fixed point: the saddle-point refiner's
    /// re-centring converged; always `true` for the single-shot refiners'
    /// accepted results. A Radon peak keeps its peak-fit position unless the
    /// refinement converged.
    pub(crate) converged: bool,
}

impl RefineResult {
    /// Build a [`RefineStatus::Accepted`] result at `xy` with the given `score`.
    #[inline]
    #[allow(deprecated)]
    pub fn accepted(xy: [f32; 2], score: f32) -> Self {
        Self {
            x: xy[0],
            y: xy[1],
            score,
            status: RefineStatus::Accepted,
            covariance_per_noise_variance: None,
            model: None,
            converged: true,
        }
    }

    /// A result that leaves the seed where it is, with `status`.
    #[inline]
    pub(crate) fn unmoved(seed_xy: [f32; 2], score: f32, status: RefineStatus) -> Self {
        Self {
            status,
            converged: false,
            ..Self::accepted(seed_xy, score)
        }
    }

    /// Position covariance in px² for an image whose pixels carry white
    /// noise of standard deviation `noise_sigma` gray levels:
    /// `max(σₙ², 1/12)·C + Σ_sys`, with `C` =
    /// [`noise_covariance_per_unit_variance`](Self::noise_covariance_per_unit_variance)
    /// and `Σ_sys` the refiner's systematic term. The model and its
    /// accuracy are described on
    /// [`Corner::position_covariance`](crate::Corner::position_covariance).
    /// `None` when the result carries no covariance. A result whose
    /// `covariance_per_noise_variance` field was set or changed by the
    /// caller carries that noise term only, without a systematic part.
    #[inline]
    pub fn position_covariance(&self, noise_sigma: f32) -> Option<PositionCovariance> {
        self.covariance_model().map(|m| m.total(noise_sigma))
    }

    /// The noise term of the position covariance per unit image-noise
    /// variance, px² per gray-level²: `C` in
    /// [`position_covariance`](Self::position_covariance). `None` when the
    /// result carries no covariance.
    #[inline]
    pub fn noise_covariance_per_unit_variance(&self) -> Option<PositionCovariance> {
        self.covariance_model().map(|m| m.noise)
    }

    /// Attach `model`, mirroring its noise term in the public field.
    #[inline]
    #[allow(deprecated)]
    pub(crate) fn set_model(&mut self, model: Option<CovarianceModel>) {
        self.covariance_per_noise_variance = model.map(|m| m.noise);
        self.model = model;
    }

    /// The covariance model this result carries (see
    /// [`CovarianceModel::effective`]).
    #[inline]
    #[allow(deprecated)]
    pub(crate) fn covariance_model(&self) -> Option<CovarianceModel> {
        CovarianceModel::effective(self.model, self.covariance_per_noise_variance)
    }
}

/// Inputs shared by refinement methods.
///
/// Callers pass whichever sources the active refiner requires:
/// - [`CenterOfMassRefiner`] reads `response`; its position covariance
///   also needs `image`, the image that ChESS response was computed from
///   (see [`RefineResult::noise_covariance_per_unit_variance`]).
/// - [`ForstnerRefiner`] and [`SaddlePointRefiner`] read `image`;
///   they ignore `response`.
///
/// Passing `None` for a required source causes the refiner to return
/// [`RefineStatus::Rejected`] without moving the seed.
#[derive(Copy, Clone, Debug, Default)]
#[non_exhaustive]
pub struct RefineContext<'a> {
    /// Grayscale image view. Required by image-patch refiners
    /// (Förstner, SaddlePoint).
    pub image: Option<ImageView<'a>>,
    /// Dense ChESS response map. Required by the CenterOfMass refiner.
    pub response: Option<&'a ResponseMap>,
}

impl<'a> RefineContext<'a> {
    /// Construct a [`RefineContext`] with the given image and response.
    #[inline]
    pub fn new(image: Option<ImageView<'a>>, response: Option<&'a ResponseMap>) -> Self {
        Self { image, response }
    }
}

/// Trait implemented by the built-in subpixel refinement backends.
///
/// # Stability
///
/// This trait is **sealed** via a private supertrait bound and cannot
/// be implemented outside this crate. The built-in implementors are
/// [`CenterOfMassRefiner`], [`ForstnerRefiner`], [`SaddlePointRefiner`],
/// and the [`Refiner`] dispatcher. It is not a public extension point:
/// select a backend through [`RefinerKind`] rather than implementing
/// this trait.
pub trait CornerRefiner: private::Sealed {
    /// Half-width of the patch the refiner needs around the seed,
    /// in input-image pixels. The caller must ensure the seed is at
    /// least this many pixels away from every image border before
    /// calling [`refine`](CornerRefiner::refine); violating this
    /// contract yields [`RefineStatus::OutOfBounds`].
    fn radius(&self) -> i32;
    /// Attempt to refine the subpixel position of a corner candidate.
    ///
    /// `seed_xy` is the initial `[x, y]` position in input-image pixels
    /// (origin at the top-left corner of the top-left pixel). Returns a
    /// [`RefineResult`] whose `status` indicates whether the position was
    /// updated.
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}

/// User-facing enum selecting a refinement backend.
///
/// Each variant carries the configuration struct for that backend.
/// Construct via [`RefinerKind::default`] for center-of-mass with
/// library defaults, or use one of the variant constructors. The active
/// backend is instantiated into a [`Refiner`] via [`Refiner::from_kind`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RefinerKind {
    /// 5×5 weighted centroid on the response map.
    CenterOfMass(CenterOfMassConfig),
    /// Gradient structure-tensor refinement on the image patch.
    Forstner(ForstnerConfig),
    /// Quadratic surface fit on the image patch.
    SaddlePoint(SaddlePointConfig),
}

impl Default for RefinerKind {
    fn default() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
}

/// Runtime refiner with reusable scratch buffers.
///
/// Constructed from a [`RefinerKind`] via [`Refiner::from_kind`].
/// Implements [`CornerRefiner`] by dispatching to the selected backend.
/// Reuse across many candidates on the same frame to amortize
/// allocation; scratch buffers are sized at construction and are not
/// reallocated on each call.
#[derive(Debug)]
#[non_exhaustive]
pub enum Refiner {
    /// 5×5 weighted centroid on the response map.
    CenterOfMass(CenterOfMassRefiner),
    /// Gradient structure-tensor refinement on the image patch.
    Forstner(ForstnerRefiner),
    /// Quadratic surface fit on the image patch.
    SaddlePoint(SaddlePointRefiner),
}

impl Refiner {
    /// Construct a [`Refiner`] from the given kind, allocating scratch
    /// buffers at the configured size. It reports no position covariance
    /// ([`Uncertainty::Off`]); see [`Self::with_uncertainty`].
    pub fn from_kind(kind: RefinerKind) -> Self {
        match kind {
            RefinerKind::CenterOfMass(cfg) => Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg)),
            RefinerKind::Forstner(cfg) => Refiner::Forstner(ForstnerRefiner::new(cfg)),
            RefinerKind::SaddlePoint(cfg) => Refiner::SaddlePoint(SaddlePointRefiner::new(cfg)),
        }
    }

    /// Select which position uncertainty the refiner reports. A refiner
    /// from [`Self::from_kind`] reports none; pass
    /// [`Uncertainty::Propagated`] for a covariance on every accepted
    /// corner. Refined positions do not depend on the tier.
    pub fn with_uncertainty(mut self, uncertainty: Uncertainty) -> Self {
        match &mut self {
            Refiner::CenterOfMass(r) => r.uncertainty = uncertainty,
            Refiner::Forstner(r) => r.uncertainty = uncertainty,
            Refiner::SaddlePoint(r) => r.uncertainty = uncertainty,
        }
        self
    }
}

impl CornerRefiner for Refiner {
    #[inline]
    fn radius(&self) -> i32 {
        match self {
            Refiner::CenterOfMass(r) => r.radius(),
            Refiner::Forstner(r) => r.radius(),
            Refiner::SaddlePoint(r) => r.radius(),
        }
    }

    #[inline]
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        match self {
            Refiner::CenterOfMass(r) => r.refine(seed_xy, ctx),
            Refiner::Forstner(r) => r.refine(seed_xy, ctx),
            Refiner::SaddlePoint(r) => r.refine(seed_xy, ctx),
        }
    }
}

impl private::Sealed for CenterOfMassRefiner {}
impl private::Sealed for ForstnerRefiner {}
impl private::Sealed for SaddlePointRefiner {}
impl private::Sealed for Refiner {}

#[cfg(test)]
pub(crate) mod test_fixtures {
    /// `side × side` smooth saddle `mid + amp·tanh(n₁·p/width)·tanh(n₂·p/width)`
    /// with edge normals at `angles` (radians) and its centre at pixel
    /// `centre + offset`, plus a deterministic ±`jitter` gray-level texture
    /// so no two pixels tie by symmetry. Unrounded.
    pub(crate) fn tanh_saddle(
        side: usize,
        centre: [i32; 2],
        offset: [f32; 2],
        angles: [f32; 2],
        width: f32,
        jitter: f32,
    ) -> Vec<f32> {
        let normals = angles.map(|a| [a.cos(), a.sin()]);
        let mut state = 0x2545_f491u32;
        (0..side * side)
            .map(|k| {
                let x = (k % side) as f32 - centre[0] as f32 - offset[0];
                let y = (k / side) as f32 - centre[1] as f32 - offset[1];
                let edge = |n: [f32; 2]| ((n[0] * x + n[1] * y) / width).tanh();
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let u = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                120.0 + 80.0 * edge(normals[0]) * edge(normals[1]) + jitter * u
            })
            .collect()
    }

    /// Mildly-blurred synthetic chessboard centred at `offset`. Used
    /// across the per-refiner test modules.
    pub(crate) fn synthetic_checkerboard(
        size: usize,
        offset: (f32, f32),
        dark: u8,
        bright: u8,
    ) -> Vec<u8> {
        let mut img = vec![0u8; size * size];
        let ox = offset.0;
        let oy = offset.1;
        for y in 0..size {
            for x in 0..size {
                let xf = x as f32 - ox;
                let yf = y as f32 - oy;
                let dark_quad = (xf >= 0.0 && yf >= 0.0) || (xf < 0.0 && yf < 0.0);
                img[y * size + x] = if dark_quad { dark } else { bright };
            }
        }
        let mut blurred = img.clone();
        for y in 1..(size - 1) {
            for x in 1..(size - 1) {
                let mut acc = 0u32;
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        acc +=
                            img[(y as i32 + ky) as usize * size + (x as i32 + kx) as usize] as u32;
                    }
                }
                blurred[y * size + x] = (acc / 9) as u8;
            }
        }
        blurred
    }
}
