#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Core primitives for computing ChESS responses and extracting subpixel corners.
//!
//! # Overview
//!
//! This crate exposes two main building blocks:
//!
//! - [`response`] – dense ChESS response computation on 8‑bit grayscale images.
//! - [`detect`] – thresholding, non‑maximum suppression (NMS), and 5×5
//!   center‑of‑mass refinement on a response map.
//!
//! The response is based on a 16‑sample ring (see [`ring`]) and is intended for
//! chessboard‑like corner detection, as described in the ChESS paper
//! (“Chess‑board Extraction by Subtraction and Summation”).
//!
//! # Features
//!
//! - `std` *(default)* – enables use of the Rust standard library. When
//!   disabled, the crate is `no_std` + `alloc`.
//! - `rayon` – parallelizes the dense response computation over image rows
//!   using the `rayon` crate. This does not change numerical results, only
//!   performance on multi‑core machines.
//! - `simd` – enables a SIMD‑accelerated inner loop for the response
//!   computation, based on `portable_simd`. This feature currently requires a
//!   nightly compiler and is intended as a performance optimization; the
//!   scalar path remains the reference implementation.
//!
//! Feature combinations:
//!
//! - no features / `std` only – single‑threaded scalar implementation.
//! - `rayon` – same scalar math, but rows are processed in parallel.
//! - `simd` – single‑threaded, but the inner ring computation is vectorized.
//! - `rayon + simd` – rows are processed in parallel *and* each row uses the
//!   SIMD‑accelerated inner loop.
//!
//! The detector in [`detect`] is independent of `rayon`/`simd` and runs the
//! same logic regardless of these features; only the time to produce the dense
//! response map changes.

pub mod detect;
pub mod response;
pub mod ring;

/// Tunable parameters for the ChESS response computation and corner detection.
#[derive(Clone, Debug)]
pub struct ChessParams {
    /// Ring radius in pixels (canonical 5, or 10 for heavy blur).
    pub radius: u32,
    /// Relative threshold as a fraction of max response (e.g. 0.015 = 1.5%).
    pub threshold_rel: f32,
    /// Absolute threshold override; if `Some`, this is used instead of `threshold_rel`.
    pub threshold_abs: Option<f32>,
    /// Non-maximum suppression radius (in pixels).
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbors in NMS window
    /// to accept a corner (rejects isolated noise).
    pub min_cluster_size: u32,
}

impl Default for ChessParams {
    fn default() -> Self {
        Self {
            radius: 5,
            threshold_rel: 0.015,
            threshold_abs: None,
            nms_radius: 1,
            min_cluster_size: 2,
        }
    }
}

/// Dense response map in row-major layout.
#[derive(Clone, Debug)]
pub struct ResponseMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl ResponseMap {
    #[inline]
    /// Response value at an integer coordinate.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
}
