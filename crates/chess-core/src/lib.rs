#![cfg_attr(not(feature = "std"), no_std)]
//! Core primitives for computing ChESS responses and extracting subpixel corners.

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
