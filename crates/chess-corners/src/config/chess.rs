use chess_corners_core::{CenterOfMassConfig, ForstnerConfig, SaddlePointConfig};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Detector kernel / ring selection
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Selects the `r=5` (canonical) or `r=10`
/// (broad) ring used by the dense response kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChessRing {
    /// Paper-default radius-5 ring (16 samples).
    #[default]
    Canonical,
    /// Radius-10 ring. Larger support window for callers that want the
    /// detector to sample farther from the candidate center.
    Broad,
}

// ---------------------------------------------------------------------------
// Refiner enum
// ---------------------------------------------------------------------------

/// Subpixel refiner selection for the ChESS detector.
///
/// Each variant carries its own tuning struct as a payload: there is
/// no shared discriminator + parallel-tuning-struct shape, so
/// switching variants can never leave a stale config field behind.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChessRefiner {
    /// Center-of-mass (intensity centroid) refinement on the response
    /// map. Cheapest refiner in the shipped benchmark; the library default.
    CenterOfMass(CenterOfMassConfig),
    /// Förstner structure-tensor refinement on the image patch.
    Forstner(ForstnerConfig),
    /// Quadratic surface fit at the saddle point.
    SaddlePoint(SaddlePointConfig),
    /// ML-backed subpixel refinement. Runs a small ONNX model on a
    /// normalized intensity patch around each candidate. Requires the
    /// `ml-refiner` feature.
    #[cfg(feature = "ml-refiner")]
    Ml,
}

impl Default for ChessRefiner {
    fn default() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
}

impl ChessRefiner {
    /// Center-of-mass refinement with default tuning.
    pub fn center_of_mass() -> Self {
        Self::CenterOfMass(CenterOfMassConfig::default())
    }
    /// Förstner structure-tensor refinement with default tuning.
    pub fn forstner() -> Self {
        Self::Forstner(ForstnerConfig::default())
    }
    /// Saddle-point quadratic fit with default tuning.
    pub fn saddle_point() -> Self {
        Self::SaddlePoint(SaddlePointConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Per-strategy config
// ---------------------------------------------------------------------------

/// Configuration for the ChESS detector branch of [`crate::DetectionStrategy`].
///
/// Carries the detector ring choice and the subpixel refiner. The shared
/// NMS / clustering thresholds ([`crate::DetectionParams`]), multiscale, and
/// upscale live at the top level of [`crate::DetectorConfig`] and apply to both
/// strategies. Descriptors always sample at the detector ring radius.
///
/// # Common knobs
///
/// - [`ring`](ChessConfig::ring) — choose the detector kernel radius.
/// - [`refiner`](ChessConfig::refiner) — select and configure the
///   subpixel refinement backend.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ChessConfig {
    /// Detector ring radius. `Canonical` selects the paper's `r=5`,
    /// `Broad` selects `r=10`. Descriptors sample at this same radius.
    pub ring: ChessRing,
    /// Subpixel refiner. Each variant carries its tuning struct.
    pub refiner: ChessRefiner,
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self {
            ring: ChessRing::Canonical,
            refiner: ChessRefiner::default(),
        }
    }
}
