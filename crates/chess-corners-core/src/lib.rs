#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Core primitives for computing ChESS responses and extracting subpixel corners.
//!
//! The crate is organized along the three orthogonal axes the
//! detector pipeline composes:
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`detect`] | Feature-detection pipelines. Two independent families share a common output type: [`detect::chess`] (ChESS response + NMS) and [`detect::radon`] (Radon SAT + peak detection). |
//! | [`refine`] | Pluggable subpixel-refinement backends. The [`refine::CornerRefiner`] trait dispatches across center-of-mass, Förstner, saddle-point, and Radon-peak refiners. |
//! | [`orientation`] | Two-axis orientation fit (`OrientationMethod::RingFit` ring fit, `OrientationMethod::DiskFit` full-disk crossing-line) shared between detectors. The [`orientation::describe_corners`] entry point produces [`CornerDescriptor`] values with subpixel position, two-axis orientation, per-axis 1σ uncertainty, contrast, and fit residual. |
//! | [`imageview`] | Zero-copy [`ImageView`] into a borrowed grayscale buffer, with optional `origin` offset for pyramid/ROI support. |
//!
//! Most users should work through the `chess-corners` facade crate rather than
//! depending on `chess-corners-core` directly. Depend on this crate only when
//! you need raw response maps, custom refiners, or the Radon detector primitives.
//!
//! # Features
//!
//! - `std` *(default)* – enables use of the Rust standard library. When
//!   disabled, the crate is `no_std` + `alloc`.
//! - `rayon` – parallelizes the dense response computation and Radon accumulation
//!   over image rows using the `rayon` crate. Does not change numerical results.
//! - `simd` – enables a SIMD‑accelerated inner loop for the ChESS response
//!   kernel, based on `portable_simd`. Requires a nightly compiler; the
//!   scalar path remains the reference implementation.
//! - `tracing` – emits structured spans around response and detector functions
//!   using the [`tracing`](https://docs.rs/tracing) ecosystem, useful for
//!   profiling and diagnostics.
//!
//! Feature combinations:
//!
//! - no features / `std` only – single‑threaded scalar implementation.
//! - `rayon` – same scalar math, but rows are processed in parallel.
//! - `simd` – single‑threaded, but the inner ring computation is vectorized.
//! - `rayon + simd` – rows are processed in parallel *and* each row uses the
//!   SIMD‑accelerated inner loop.
//!
//! The detector in [`detect`] is independent of `rayon`/`simd`, and `tracing`
//! only adds observability; none of these features change the numerical
//! results, only performance and instrumentation.
//!
//! The ChESS idea is proposed in Bennett, Lasenby, *ChESS: A Fast and
//! Accurate Chessboard Corner Detector*, CVIU 2014.

pub mod detect;
pub mod imageview;
pub mod orientation;
pub mod refine;

use crate::detect::chess::ring::RingOffsets;
use serde::{Deserialize, Serialize};

pub use crate::detect::radon::primitives::{fit_peak_frac, PeakFitMode};
pub use crate::detect::radon::{
    detect_corners_from_radon, radon_response_u8, RadonBuffers, RadonDetectorParams,
    RadonResponseView, SatElem,
};
pub use crate::detect::{AxisEstimate, Corner, CornerDescriptor};
pub use crate::orientation::{
    fit_axes_at_point, fit_axes_from_samples, AxisFitResult, OrientationMethod,
};
pub use crate::refine::{
    CenterOfMassConfig, CenterOfMassRefiner, CornerRefiner, ForstnerConfig, ForstnerRefiner,
    RadonPeakConfig, RadonPeakRefiner, RefineContext, RefineResult, RefineStatus, Refiner,
    RefinerKind, SaddlePointConfig, SaddlePointRefiner,
};
pub use imageview::ImageView;
/// Tunable parameters for the ChESS response computation and corner detection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessParams {
    /// Use the larger r=10 ring instead of the canonical r=5.
    pub use_radius10: bool,
    /// Optional override for descriptor sampling ring (r=5 vs r=10). Falls back
    /// to `use_radius10` when `None`.
    pub descriptor_use_radius10: Option<bool>,
    /// Relative threshold as a fraction of max response (e.g. 0.2 = 20%).
    pub threshold_rel: f32,
    /// Absolute threshold override; if `Some`, this is used instead of `threshold_rel`.
    pub threshold_abs: Option<f32>,
    /// Non-maximum suppression radius (in pixels).
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbors in NMS window
    /// to accept a corner (rejects isolated noise).
    pub min_cluster_size: u32,
    /// Subpixel refinement backend and its configuration. Defaults to the legacy
    /// center-of-mass refiner on the response map.
    pub refiner: RefinerKind,
    /// Orientation-fit method used to estimate the two grid axes at
    /// each detected corner. Default [`OrientationMethod::RingFit`]
    /// fits the parametric two-axis model with robust seeding and
    /// calibrated per-axis uncertainties.
    #[serde(default)]
    pub orientation_method: OrientationMethod,
}

impl Default for ChessParams {
    fn default() -> Self {
        Self {
            use_radius10: false,
            descriptor_use_radius10: None,
            // Paper's contract: accept every strictly-positive ChESS
            // response. `threshold_abs = Some(0.0)` combined with the
            // strict comparison in `detect_corners_from_response` gives
            // "R > 0 ⇒ corner". `threshold_rel = 0.2` is kept as a
            // default-sized opt-in value for callers that explicitly
            // switch to `threshold_abs = None`.
            threshold_rel: 0.2,
            threshold_abs: Some(0.0),
            nms_radius: 2,
            min_cluster_size: 2,
            refiner: RefinerKind::default(),
            orientation_method: OrientationMethod::default(),
        }
    }
}

impl ChessParams {
    #[inline]
    pub fn ring_radius(&self) -> u32 {
        if self.use_radius10 {
            10
        } else {
            5
        }
    }

    #[inline]
    pub fn descriptor_ring_radius(&self) -> u32 {
        match self.descriptor_use_radius10 {
            Some(true) => 10,
            Some(false) => 5,
            None => self.ring_radius(),
        }
    }

    #[inline]
    pub fn ring(&self) -> RingOffsets {
        RingOffsets::from_radius(self.ring_radius())
    }

    #[inline]
    pub fn descriptor_ring(&self) -> RingOffsets {
        RingOffsets::from_radius(self.descriptor_ring_radius())
    }
}

/// Dense response map in row-major layout.
#[derive(Clone, Debug)]
pub struct ResponseMap {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) data: Vec<f32>,
}

impl ResponseMap {
    /// Create a new response map. `data` must have exactly `w * h` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != w * h`.
    pub fn new(w: usize, h: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), w * h, "ResponseMap data length mismatch");
        Self { w, h, data }
    }

    /// Width of the response map.
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    /// Height of the response map.
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    /// Raw response data in row-major order.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to the raw response data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    /// Response value at an integer coordinate.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
}
