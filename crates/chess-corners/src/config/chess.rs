use chess_corners_core::CenterOfMassConfig;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Detector kernel / ring selection
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Selects the `r=5` (canonical) or `r=10`
/// (broad) ring used by the dense response kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
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
// Per-strategy config
// ---------------------------------------------------------------------------

/// Configuration for the ChESS detector branch of [`crate::DetectionStrategy`].
///
/// Carries the detector ring choice and the tuning of ChESS's own subpixel
/// localizer. The shared NMS / clustering thresholds
/// ([`crate::DetectionParams`]), multiscale, upscale and the optional image
/// refiner ([`crate::DetectorConfig::refiner`]) live at the top level of
/// [`crate::DetectorConfig`] and apply to both strategies. Descriptors
/// always sample at the detector ring radius.
///
/// # Common knobs
///
/// - [`ring`](ChessConfig::ring) — choose the detector kernel radius.
/// - [`center_of_mass`](ChessConfig::center_of_mass) — tune the subpixel
///   localizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ChessConfig {
    /// Detector ring radius. `Canonical` selects the paper's `r=5`,
    /// `Broad` selects `r=10`. Descriptors sample at this same radius.
    pub ring: ChessRing,
    /// ChESS's subpixel localizer: the response-weighted center of mass of
    /// the response map around each integer peak. It gives every corner its
    /// final position unless an image refiner
    /// ([`DetectorConfig::refiner`](crate::DetectorConfig::refiner)) is set;
    /// the image refiner then starts from the integer peak instead.
    pub center_of_mass: CenterOfMassConfig,
}
