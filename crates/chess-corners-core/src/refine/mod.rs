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
//! - [`radon_peak`] — Radon-projection refiner along candidate axes.

use crate::imageview::ImageView;
use crate::ResponseMap;
use serde::{Deserialize, Serialize};

pub mod center_of_mass;
pub mod forstner;
pub mod radon_peak;
pub mod saddle_point;

pub use center_of_mass::{CenterOfMassConfig, CenterOfMassRefiner};
pub use forstner::{ForstnerConfig, ForstnerRefiner};
pub use radon_peak::{RadonPeakConfig, RadonPeakRefiner};
pub use saddle_point::{SaddlePointConfig, SaddlePointRefiner};

/// Status of a refinement attempt.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefineStatus {
    Accepted,
    Rejected,
    OutOfBounds,
    IllConditioned,
}

/// Result of refining a single corner candidate.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct RefineResult {
    /// Refined subpixel x coordinate.
    pub x: f32,
    /// Refined subpixel y coordinate.
    pub y: f32,
    pub score: f32,
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
#[derive(Copy, Clone, Debug, Default)]
#[non_exhaustive]
pub struct RefineContext<'a> {
    pub image: Option<ImageView<'a>>,
    pub response: Option<&'a ResponseMap>,
}

impl<'a> RefineContext<'a> {
    /// Construct a [`RefineContext`] with the given image and response.
    #[inline]
    pub fn new(image: Option<ImageView<'a>>, response: Option<&'a ResponseMap>) -> Self {
        Self { image, response }
    }
}

/// Trait implemented by pluggable refinement backends.
pub trait CornerRefiner {
    /// Half-width of the patch the refiner needs around the seed.
    fn radius(&self) -> i32;
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}

/// User-facing enum selecting a refinement backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RefinerKind {
    CenterOfMass(CenterOfMassConfig),
    Forstner(ForstnerConfig),
    SaddlePoint(SaddlePointConfig),
    RadonPeak(RadonPeakConfig),
}

impl Default for RefinerKind {
    fn default() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
}

/// Runtime refiner with reusable scratch buffers.
#[derive(Debug)]
#[non_exhaustive]
pub enum Refiner {
    CenterOfMass(CenterOfMassRefiner),
    Forstner(ForstnerRefiner),
    SaddlePoint(SaddlePointRefiner),
    RadonPeak(RadonPeakRefiner),
}

impl Refiner {
    pub fn from_kind(kind: RefinerKind) -> Self {
        match kind {
            RefinerKind::CenterOfMass(cfg) => Refiner::CenterOfMass(CenterOfMassRefiner::new(cfg)),
            RefinerKind::Forstner(cfg) => Refiner::Forstner(ForstnerRefiner::new(cfg)),
            RefinerKind::SaddlePoint(cfg) => Refiner::SaddlePoint(SaddlePointRefiner::new(cfg)),
            RefinerKind::RadonPeak(cfg) => Refiner::RadonPeak(RadonPeakRefiner::new(cfg)),
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
            Refiner::RadonPeak(r) => r.radius(),
        }
    }

    #[inline]
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult {
        match self {
            Refiner::CenterOfMass(r) => r.refine(seed_xy, ctx),
            Refiner::Forstner(r) => r.refine(seed_xy, ctx),
            Refiner::SaddlePoint(r) => r.refine(seed_xy, ctx),
            Refiner::RadonPeak(r) => r.refine(seed_xy, ctx),
        }
    }
}

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
