use serde::{Deserialize, Serialize};

/// Coarse-to-fine multiscale configuration.
///
/// JSON shape mirrors [`crate::UpscaleConfig`]:
///
/// - `{ "single_scale": null }` — run the detector once on the full image.
/// - `{ "pyramid": { "levels": 3, "min_size": 128, "refinement_radius": 3 } }`
///   — build an image pyramid, detect seeds on the coarsest level, and
///   refine each seed into the base image. Honoured by both ChESS and
///   Radon strategies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MultiscaleConfig {
    /// Single-scale detection (no pyramid).
    #[default]
    SingleScale,
    /// Coarse-to-fine pyramid detection.
    Pyramid {
        /// Number of pyramid levels (≥ 1). Level 0 is the base image;
        /// each subsequent level is a 2× box-filter downsample.
        levels: u8,
        /// Minimum short-edge length in pixels. The pyramid stops once
        /// the next level would fall below this size.
        min_size: usize,
        /// ROI half-radius at the coarse level used to refine each seed
        /// into the base image, in coarse-level pixels.
        refinement_radius: u32,
    },
}

impl MultiscaleConfig {
    /// Three-level pyramid with library defaults (`min_size = 128`, `refinement_radius = 3`).
    /// Equivalent to the multiscale preset used by [`crate::DetectorConfig::chess_multiscale`]
    /// and [`crate::DetectorConfig::radon_multiscale`].
    pub const fn pyramid_default() -> Self {
        Self::Pyramid {
            levels: 3,
            min_size: 128,
            refinement_radius: 3,
        }
    }
    /// Pyramid with caller-supplied parameters.
    pub const fn pyramid(levels: u8, min_size: usize, refinement_radius: u32) -> Self {
        Self::Pyramid {
            levels,
            min_size,
            refinement_radius,
        }
    }
}
