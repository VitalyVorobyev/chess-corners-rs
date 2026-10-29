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
/// - [`on_refiner_reject`](ChessConfig::on_refiner_reject) — keep the
///   corners an image refiner rejects at their center-of-mass position.
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
    /// What happens to a corner the image refiner
    /// ([`DetectorConfig::refiner`](crate::DetectorConfig::refiner)) rejects.
    /// Ignored when no image refiner is set. Default
    /// [`OnRefinerReject::Drop`].
    pub on_refiner_reject: OnRefinerReject,
}

/// What a ChESS detection does with a corner the image refiner rejects (see
/// [`ChessConfig::on_refiner_reject`]).
///
/// The image refiner rejects a peak whose patch it cannot fit — not a clear
/// saddle, a fit that leaves its window, a seed too close to the border. On
/// a clean board those are rare; on a cluttered or blurred one they include
/// real corners the center of mass localizes. The Radon detector always
/// keeps such a corner at its peak-fit position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OnRefinerReject {
    /// Drop the corner: every reported corner was localized by the image
    /// refiner.
    #[default]
    Drop,
    /// Keep the corner at the ChESS center-of-mass position (with that
    /// localizer's position covariance). The corners the image refiner
    /// accepts are unchanged, so a single-scale detection returns the `Drop`
    /// output plus these corners. Under a pyramid the coarse seeds follow
    /// the same rule, so a seed kept at the coarse level can also yield a
    /// corner the image refiner localizes at the base level.
    KeepCenterOfMass,
}
