//! Low-level ChESS detection parameters.
//!
//! [`ChessParams`] is the parameter bundle consumed by the ChESS
//! response and detection stages. It is re-exported from the crate
//! root as part of the low-level contract: the `chess-corners` facade
//! lowers its `DetectorConfig` onto it, and callers driving the
//! response and detection stages directly construct it themselves.

use crate::detect::chess::ring::RingOffsets;
use crate::orientation::OrientationMethod;
use crate::refine::RefinerKind;
use serde::{Deserialize, Serialize};

/// Tunable parameters for the ChESS response computation and corner detection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessParams {
    /// Use the larger r=10 ring instead of the canonical r=5.
    pub use_radius10: bool,
    /// Absolute response floor: a corner is kept when its raw ChESS
    /// response exceeds this value (strict `>`). `0.0` accepts every
    /// strictly-positive response — the paper's contract.
    pub threshold: f32,
    /// Non-maximum suppression radius (in pixels).
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbors in NMS window
    /// to accept a corner (rejects isolated noise).
    pub min_cluster_size: u32,
    /// Subpixel refinement backend and its configuration. Defaults to
    /// center-of-mass on the response map.
    pub refiner: RefinerKind,
    /// Orientation-fit method used to estimate the two grid axes at
    /// each detected corner, or `None` to skip the fit entirely (every
    /// descriptor then carries `axes: None`). Default
    /// `Some(`[`OrientationMethod::RingFit`]`)` fits the parametric
    /// two-axis model with robust seeding and calibrated per-axis
    /// uncertainties.
    #[serde(default = "default_orientation_method")]
    pub orientation_method: Option<OrientationMethod>,
}

#[inline]
fn default_orientation_method() -> Option<OrientationMethod> {
    Some(OrientationMethod::default())
}

impl Default for ChessParams {
    fn default() -> Self {
        Self {
            use_radius10: false,
            // Paper's contract: accept every strictly-positive ChESS
            // response. `threshold = 0.0` combined with the strict
            // comparison in `detect_corners_from_response` gives
            // "R > 0 ⇒ corner". The facade raises this to a denoise
            // floor for real images.
            threshold: 0.0,
            nms_radius: 2,
            min_cluster_size: 2,
            refiner: RefinerKind::default(),
            orientation_method: Some(OrientationMethod::default()),
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
    pub(crate) fn ring(&self) -> RingOffsets {
        RingOffsets::from_radius(self.ring_radius())
    }
}
