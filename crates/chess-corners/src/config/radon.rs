use chess_corners_core::PeakFitMode;
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
#[serde(default)]
#[non_exhaustive]
pub struct RadonConfig {
    /// Advanced tuning. Half-length of each Radon ray in
    /// working-resolution pixels. The ray has `2·ray_radius + 1`
    /// samples. Paper default at `image_upsample = 2` is `ray_radius = 4`.
    /// Shorter rays are faster but integrate less signal; longer rays are
    /// more discriminating but may cross into neighbouring cells.
    pub ray_radius: u32,
    /// Image-level supersampling factor applied before ray integration.
    /// `1` operates on the input grid; `2` (paper default) is equivalent
    /// to bilinearly upsampling the input first, giving sub-pixel ray
    /// positioning. Values ≥ 3 are clamped to 2 by the core detector.
    pub image_upsample: u32,
    /// Advanced tuning. Half-size of the box blur applied to the Radon
    /// response map after integration. `0` disables blurring; `1`
    /// (default) yields a 3×3 box, smoothing quantisation noise in the
    /// response. Increase only on very high-SNR images where extra
    /// smoothing is unwanted.
    pub response_blur_radius: u32,
    /// Advanced tuning. Peak-fit mode for the 3-point subpixel
    /// refinement of the response-map argmax. `Gaussian` (default) fits
    /// on log-response (more accurate near the peak); `Parabolic` fits
    /// directly on the response values. See [`PeakFitMode`].
    pub peak_fit: PeakFitMode,
}

impl Default for RadonConfig {
    fn default() -> Self {
        Self {
            ray_radius: 4,
            image_upsample: 2,
            response_blur_radius: 1,
            peak_fit: PeakFitMode::Gaussian,
        }
    }
}
