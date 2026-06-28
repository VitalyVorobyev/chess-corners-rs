#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Core primitives for ChESS/Radon response computation, subpixel
//! refinement, and corner descriptors.
//!
//! The crate exposes a deliberate low-level contract through its
//! crate root: response computation ([`chess_response_u8`],
//! [`radon_response_u8`]), corner detection ([`find_corners_u8`],
//! [`detect_corners_from_response`]), pluggable subpixel refinement
//! (the [`CornerRefiner`] trait and built-in refiners), the two-axis
//! orientation fit ([`fit_axes_at_point`], [`describe_corners`]), and
//! the [`ImageView`] borrowed-buffer type. The detector pipeline
//! composes three orthogonal stages — detection, refinement, and
//! orientation fit — all reachable from the crate root.
//!
//! Most users should work through the `chess-corners` facade crate rather than
//! depending on `chess-corners-core` directly. Depend on this crate only when
//! you need raw response maps, custom refiners, or the Radon detector primitives.
//!
//! Implementation-specific primitives that benches, experiments, and
//! advanced callers occasionally need — ring offset tables and the
//! scalar reference response path — live under [`unstable`], which
//! carries no semver guarantee.
//!
//! # Features
//!
//! - `std` *(default)* – compatibility feature reserved for future use.
//!   The current detector implementation requires the Rust standard library.
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
//! The detector is independent of `rayon`/`simd`, and `tracing`
//! only adds observability; none of these features change the numerical
//! results, only performance and instrumentation.
//!
//! # Minimum supported Rust version
//!
//! The default (stable) build requires Rust **1.88** or newer, as
//! declared by `rust-version` in `Cargo.toml`. The optional `simd`
//! feature uses `portable_simd` and therefore requires a **nightly**
//! toolchain; every other feature builds on stable.
//!
//! The ChESS idea is proposed in Bennett, Lasenby, *ChESS: A Fast and
//! Accurate Chessboard Corner Detector*, CVIU 2014.

mod detect;
mod imageview;
mod orientation;
mod params;
mod refine;

pub mod unstable;

// Internal name resolution only. `ChessParams` is an implementation-level
// parameter type; its public path is [`unstable::ChessParams`], not the
// crate root.
pub(crate) use crate::params::ChessParams;

pub use crate::detect::chess::response::{chess_response_u8, Roi};
pub use crate::detect::dense::{ChessBuffers, ChessDetector, DenseDetector, RadonDetector};
pub use crate::detect::radon::primitives::PeakFitMode;
pub use crate::detect::radon::{
    detect_peaks_from_radon, radon_response_u8, RadonBuffers, RadonDetectorParams,
    RadonResponseView,
};
pub use crate::detect::{
    detect_corners_from_response, detect_corners_from_response_with_refiner, find_corners_u8,
    merge_corners_simple, AxisEstimate, Corner, CornerDescriptor,
};
pub use crate::orientation::{
    describe_corners, fit_axes_at_point, fit_axes_from_samples, AxisFitResult, OrientationMethod,
};
pub use crate::refine::{
    CenterOfMassConfig, CenterOfMassRefiner, CornerRefiner, ForstnerConfig, ForstnerRefiner,
    RefineContext, RefineResult, RefineStatus, Refiner, SaddlePointConfig, SaddlePointRefiner,
};
pub use imageview::ImageView;

/// Dense response map in row-major layout.
#[derive(Clone, Debug, Default)]
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
