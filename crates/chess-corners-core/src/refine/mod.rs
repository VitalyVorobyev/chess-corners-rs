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
use crate::ResponseMap;
use serde::{Deserialize, Serialize};

pub mod center_of_mass;
pub mod forstner;
pub mod saddle_point;

pub use center_of_mass::{CenterOfMassConfig, CenterOfMassRefiner};
pub use forstner::{ForstnerConfig, ForstnerRefiner};
pub use saddle_point::{SaddlePointConfig, SaddlePointRefiner};

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
///   estimate (e.g. too-flat gradient for Förstner, or a degenerate
///   Hessian for saddle-point). The seed coordinates are unchanged.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefineStatus {
    Accepted,
    Rejected,
    OutOfBounds,
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
///   stronger; units match the ChESS response scale. A score of `0.0`
///   means no positive response was found.
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
}

impl RefineResult {
    #[inline]
    pub fn accepted(xy: [f32; 2], score: f32) -> Self {
        Self {
            x: xy[0],
            y: xy[1],
            score,
            status: RefineStatus::Accepted,
        }
    }
}

/// Inputs shared by refinement methods.
///
/// Callers pass whichever sources the active refiner requires:
/// - [`CenterOfMassRefiner`] reads `response`; it ignores `image`.
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
    CenterOfMass(CenterOfMassRefiner),
    Forstner(ForstnerRefiner),
    SaddlePoint(SaddlePointRefiner),
}

impl Refiner {
    /// Construct a [`Refiner`] from the given kind, allocating scratch
    /// buffers at the configured size.
    pub fn from_kind(kind: RefinerKind) -> Self {
        match kind {
            RefinerKind::CenterOfMass(cfg) => Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg)),
            RefinerKind::Forstner(cfg) => Refiner::Forstner(ForstnerRefiner::new(cfg)),
            RefinerKind::SaddlePoint(cfg) => Refiner::SaddlePoint(SaddlePointRefiner::new(cfg)),
        }
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
