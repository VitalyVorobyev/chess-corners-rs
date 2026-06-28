//! Low-level ChESS detection parameters.
//!
//! [`ChessParams`] is the parameter bundle consumed by the ChESS
//! response and detection stages. It is re-exported from
//! [`crate::unstable`] rather than the crate root: it is an
//! implementation-level translation target for the `chess-corners`
//! facade and the benchmarks, not part of the stable low-level
//! contract.

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
    /// Relative threshold as a fraction of max response (e.g. 0.2 = 20%).
    pub threshold_rel: f32,
    /// Absolute threshold override; if `Some`, this is used instead of `threshold_rel`.
    pub threshold_abs: Option<f32>,
    /// Non-maximum suppression radius (in pixels).
    pub nms_radius: u32,
    /// Minimum count of positive-response neighbors in NMS window
    /// to accept a corner (rejects isolated noise).
    pub min_cluster_size: u32,
    /// Subpixel refinement backend and its configuration. Defaults to
    /// center-of-mass on the response map.
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
    pub fn ring(&self) -> RingOffsets {
        RingOffsets::from_radius(self.ring_radius())
    }
}
