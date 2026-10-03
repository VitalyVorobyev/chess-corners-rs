use chess_corners_core::{PeakFitMode, RadonDetectorParams};
use serde::{Deserialize, Serialize};

/// Configuration for the whole-image Radon detector branch of
/// [`crate::DetectionStrategy`].
///
/// All radii and counts are in **working-resolution** pixels (i.e.
/// after `image_upsample`). The shared NMS / clustering thresholds
/// ([`crate::DetectionParams`]), multiscale, and upscale live at the top level
/// of [`crate::DetectorConfig`] and apply to both strategies.
///
/// # Common knobs
///
/// - [`image_upsample`](RadonConfig::image_upsample) — `2` (the default)
///   reproduces the paper's 2× supersampled detection; `1` is faster but
///   less accurate on low-resolution inputs.
///
/// # Advanced tuning
///
/// The remaining fields control low-level detection behaviour. The
/// defaults reproduce the paper's recommended settings and work well
/// for typical camera images. Adjust them only when you have a specific
/// reason (e.g. a non-standard image resolution or SNR budget).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct RadonConfig {
    /// Advanced tuning. Half-length of each Radon ray, in
    /// working-resolution pixels (integer `>= 1`; the ray has
    /// `2·ray_radius + 1` samples). Paper default at `image_upsample = 2`
    /// is `4`. Shorter rays are faster but integrate less signal; longer
    /// rays are more discriminating but may cross into neighbouring cells.
    #[cfg_attr(feature = "schemars", schemars(range(min = 1)))]
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub ray_radius: u32,
    /// Image-level supersampling factor applied before ray integration
    /// (`1` or `2`). `1` operates on the input grid; `2` (paper default)
    /// bilinearly upsamples the input first, giving sub-pixel ray
    /// positioning. The core detector clamps values outside `1..=2`.
    #[cfg_attr(feature = "schemars", schemars(range(min = 1, max = 2)))]
    pub image_upsample: u32,
    /// Advanced tuning. Half-size of the box blur applied to the Radon
    /// response map after integration, in working-resolution pixels
    /// (integer `>= 0`). `0` disables blurring; the default `1` yields a
    /// 3×3 box that smooths quantisation noise.
    #[cfg_attr(feature = "schemars", schemars(extend("x-unit" = "px")))]
    pub response_blur_radius: u32,
    /// Advanced tuning. Peak-fit mode for the 3-point subpixel
    /// refinement of the response-map argmax. `Gaussian` (default) fits
    /// on log-response (more accurate near the peak); `Parabolic` fits
    /// directly on the response values. See [`PeakFitMode`].
    pub peak_fit: PeakFitMode,
}

impl Default for RadonConfig {
    fn default() -> Self {
        // Single source of truth: mirror `RadonDetectorParams::default()`
        // (chess-corners-core) rather than restating the paper defaults here.
        let core_defaults = RadonDetectorParams::default();
        Self {
            ray_radius: core_defaults.ray_radius,
            image_upsample: core_defaults.image_upsample,
            response_blur_radius: core_defaults.response_blur_radius,
            peak_fit: core_defaults.peak_fit,
        }
    }
}
