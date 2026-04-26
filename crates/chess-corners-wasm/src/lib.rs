//! WebAssembly bindings for the ChESS / Radon corner detector.
//!
//! Two ways to drive the detector from JS:
//!
//! 1. **Setter shortcuts** — `new ChessDetector()` then
//!    `det.set_threshold(...)` / `det.set_pyramid_levels(...)` /
//!    `det.set_detector_mode("radon")` etc. Existing demo code uses
//!    this path. Backwards-compatible.
//! 2. **Typed `ChessConfig`** — construct the typed
//!    [`ChessConfig`] (with nested `RefinerConfig`,
//!    `RadonDetectorParams`, `UpscaleConfig`, …), then pass it to
//!    [`ChessDetector::with_config`]. Exposes every public Rust
//!    facade field with type-safe getters/setters.
//!
//! Both paths edit the same underlying Rust struct, so they can be
//! mixed at will.
//!
//! # Editing nested typed configs
//!
//! Nested edits propagate naturally — getters hand back a wrapper
//! backed by the same shared cell as the parent, so chained mutation
//! works without a round-trip:
//!
//! ```ignore
//! cfg.refiner.kind = RefinementMethod.RadonPeak;   // propagates
//! cfg.refiner.forstner.maxOffset = 2.0;            // propagates
//! cfg.radonDetector.rayRadius = 5;                 // propagates
//! ```
//!
//! See [`config`] for the cell-sharing details.

pub mod config;

use chess_corners::{
    radon_heatmap_u8, ChessConfig as RsChessConfig, DetectorMode as RsDetectorMode, PyramidBuffers,
    RefinementMethod as RsRefinementMethod, ThresholdMode as RsThresholdMode,
    UpscaleConfig as RsUpscaleConfig,
};
use chess_corners_core::response::chess_response_u8;
use chess_corners_core::ResponseMap;
use wasm_bindgen::prelude::*;

pub use crate::config::{
    CenterOfMassConfig, ChessConfig, DescriptorMode, DetectorMode, ForstnerConfig, PeakFitMode,
    RadonDetectorParams, RadonPeakConfig, RefinementMethod, RefinerConfig, SaddlePointConfig,
    ThresholdMode, UpscaleConfig, UpscaleMode,
};

/// Convert RGBA pixels to grayscale using BT.601 luminance weights.
fn rgba_to_gray(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let npix = (width * height) as usize;
    let mut gray = Vec::with_capacity(npix);
    for i in 0..npix {
        let r = rgba[4 * i] as f32;
        let g = rgba[4 * i + 1] as f32;
        let b = rgba[4 * i + 2] as f32;
        gray.push((0.299 * r + 0.587 * g + 0.114 * b) as u8);
    }
    gray
}

/// Stateful ChESS corner detector for use from JavaScript.
///
/// Holds configuration and reusable pyramid buffers so that repeated
/// calls (e.g. from a webcam feed) avoid re-allocating intermediate
/// storage.
#[wasm_bindgen]
pub struct ChessDetector {
    config: RsChessConfig,
    buffers: PyramidBuffers,
    last_response: Option<ResponseMap>,
    last_radon_response: Option<ResponseMap>,
    /// Working-to-input scale factor cached at the moment
    /// `radon_heatmap` produced `last_radon_response`. Returning a
    /// cached value (instead of recomputing from the live config)
    /// keeps `radon_heatmap_width` / `_height` / `_scale` mutually
    /// consistent if the caller mutates the detector's upscale or
    /// `radon_detector.image_upsample` between the heatmap call and
    /// the accessor calls. `0` until the first heatmap is computed.
    last_radon_scale: u32,
}

impl Default for ChessDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ChessDetector {
    /// Create a detector with default single-scale configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: RsChessConfig::single_scale(),
            buffers: PyramidBuffers::with_capacity(1),
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        }
    }

    /// Create a detector with the recommended multiscale preset.
    pub fn multiscale() -> Self {
        let config = RsChessConfig::multiscale();
        let levels = config.pyramid_levels;
        Self {
            config,
            buffers: PyramidBuffers::with_capacity(levels),
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        }
    }

    /// Create a detector seeded from a typed
    /// [`ChessConfig`]. The full public
    /// config surface — refiner subconfigs, Radon params, descriptor
    /// mode, coarse-to-fine radii, upscale — is reachable through
    /// the typed object (see module docs for the alternative
    /// setter-shortcut path).
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: &ChessConfig) -> Self {
        let snapshot = config.snapshot();
        let levels = snapshot.pyramid_levels;
        Self {
            config: snapshot,
            buffers: PyramidBuffers::with_capacity(levels),
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        }
    }

    /// Snapshot the current configuration as a typed
    /// [`ChessConfig`]. The returned object is a snapshot — its
    /// cells are *not* shared with the detector's live state. Use
    /// [`Self::apply_config`] to commit edits made on the snapshot.
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> ChessConfig {
        ChessConfig::from_inner_for_js(self.config.clone())
    }

    /// Replace the detector's configuration with the given typed
    /// [`ChessConfig`]. Resizes pyramid scratch buffers if
    /// `pyramid_levels` changed.
    #[wasm_bindgen(js_name = applyConfig)]
    pub fn apply_config(&mut self, config: &ChessConfig) {
        let snapshot = config.snapshot();
        let levels = snapshot.pyramid_levels;
        if levels != self.config.pyramid_levels {
            self.buffers = PyramidBuffers::with_capacity(levels);
        }
        self.config = snapshot;
    }

    // ---- Config setters ----
    //
    // Threshold, NMS radius, and min-cluster-size setters mirror into
    // both the ChESS-side `ChessConfig` fields and the Radon-side
    // `ChessConfig::radon_detector` fields. The detector pipeline
    // reads only the fields that correspond to the active
    // `detector_mode`, but mirroring means JS callers that toggle
    // between modes at runtime get their tuning applied on both paths
    // without having to re-invoke each setter after every mode switch.

    /// Set the relative threshold (fraction of max response, default 0.2).
    ///
    /// Applied to both detectors. For ChESS this switches
    /// `threshold_mode` to `Relative`; for Radon this sets
    /// `radon_detector.threshold_rel` and clears any absolute override
    /// on `threshold_abs` so the relative value wins.
    pub fn set_threshold(&mut self, rel: f32) {
        self.config.threshold_mode = RsThresholdMode::Relative;
        self.config.threshold_value = rel;
        self.config.radon_detector.threshold_rel = rel;
        self.config.radon_detector.threshold_abs = None;
    }

    /// Set the non-maximum suppression radius (default 2).
    ///
    /// Mirrored into both detectors. **Convention note:** the ChESS
    /// `nms_radius` is measured in input-image pixels, while the Radon
    /// `nms_radius` is measured in **working-resolution** pixels
    /// (post-`image_upsample`). At the default `image_upsample = 2`
    /// the same numeric value therefore selects a 2× smaller physical
    /// neighbourhood on the Radon path. Most calibration fixtures
    /// don't notice this, but callers tuning both modes side-by-side
    /// should be aware.
    pub fn set_nms_radius(&mut self, r: u32) {
        self.config.nms_radius = r;
        self.config.radon_detector.nms_radius = r;
    }

    /// Toggle the large r=10 ring (default: r=5).
    pub fn set_broad_mode(&mut self, v: bool) {
        self.config.detector_mode = if v {
            RsDetectorMode::Broad
        } else {
            RsDetectorMode::Canonical
        };
    }

    /// Select the detector kernel: `"canonical"`, `"broad"`, or
    /// `"radon"`. `canonical` / `broad` are the two ChESS variants;
    /// `radon` picks the whole-image Duda-Frese detector, useful
    /// under heavy blur or low contrast. Threshold / NMS / cluster
    /// tuning from the other setters is preserved across mode
    /// switches because those setters mirror into both detectors.
    pub fn set_detector_mode(&mut self, name: &str) -> Result<(), JsValue> {
        self.config.detector_mode = match name {
            "canonical" => RsDetectorMode::Canonical,
            "broad" => RsDetectorMode::Broad,
            "radon" => RsDetectorMode::Radon,
            _ => {
                return Err(JsValue::from_str(
                    "unknown detector_mode: use canonical, broad, or radon",
                ))
            }
        };
        Ok(())
    }

    /// Set the minimum cluster size for accepting a corner (default 2).
    /// Mirrored into both detectors so runtime detector_mode switches
    /// preserve the tuning.
    pub fn set_min_cluster_size(&mut self, v: u32) {
        self.config.min_cluster_size = v;
        self.config.radon_detector.min_cluster_size = v;
    }

    /// Set the number of pyramid levels (1 = single-scale, >=2 = multiscale).
    ///
    /// Returns an error if `n` is 0.
    pub fn set_pyramid_levels(&mut self, n: u8) -> Result<(), JsValue> {
        if n == 0 {
            return Err(JsValue::from_str("pyramid_levels must be >= 1"));
        }
        self.config.pyramid_levels = n;
        self.buffers = PyramidBuffers::with_capacity(n);
        Ok(())
    }

    /// Set the minimum pyramid level size in pixels (default 128).
    pub fn set_pyramid_min_size(&mut self, v: u32) {
        self.config.pyramid_min_size = v as usize;
    }

    /// Set the optional pre-pipeline upscale factor.
    ///
    /// `factor == 0` or `factor == 1` disables upscaling. Accepts
    /// integer factors 2, 3, 4. Corner coordinates are always returned
    /// in input-image pixel space; callers do not need to rescale.
    pub fn set_upscale_factor(&mut self, factor: u32) -> Result<(), JsValue> {
        self.config.upscale = match factor {
            0 | 1 => RsUpscaleConfig::disabled(),
            2..=4 => RsUpscaleConfig::fixed(factor),
            other => {
                return Err(JsValue::from_str(&format!(
                    "unsupported upscale factor {other} (expected 0, 1, 2, 3, or 4)",
                )));
            }
        };
        Ok(())
    }

    /// Set the subpixel refiner: `"center_of_mass"`, `"forstner"`,
    /// `"saddle_point"`, or `"radon_peak"`.
    pub fn set_refiner(&mut self, name: &str) -> Result<(), JsValue> {
        self.config.refiner.kind =
            match name {
                "center_of_mass" => RsRefinementMethod::CenterOfMass,
                "forstner" => RsRefinementMethod::Forstner,
                "saddle_point" => RsRefinementMethod::SaddlePoint,
                "radon_peak" => RsRefinementMethod::RadonPeak,
                _ => return Err(JsValue::from_str(
                    "unknown refiner: use center_of_mass, forstner, saddle_point, or radon_peak",
                )),
            };
        Ok(())
    }

    // ---- Detection ----

    /// Detect corners from grayscale u8 pixels.
    ///
    /// Returns a `Float32Array` with stride 9 per corner:
    /// `[x, y, response, contrast, fit_rms,
    ///   axis0_angle, axis0_sigma, axis1_angle, axis1_sigma, ...]`.
    pub fn detect(&mut self, pixels: &[u8], width: u32, height: u32) -> js_sys::Float32Array {
        let corners = chess_corners::find_chess_corners_u8(pixels, width, height, &self.config);
        corners_to_f32_array(&corners)
    }

    /// Detect corners from RGBA pixels (e.g. from canvas `getImageData`).
    ///
    /// Converts to grayscale internally, then detects.
    pub fn detect_rgba(&mut self, pixels: &[u8], width: u32, height: u32) -> js_sys::Float32Array {
        let gray = rgba_to_gray(pixels, width, height);
        self.detect(&gray, width, height)
    }

    // ---- Response map ----

    /// Compute the raw ChESS response map from grayscale pixels.
    ///
    /// Returns a `Float32Array` in row-major order (width x height).
    pub fn response(&mut self, pixels: &[u8], width: u32, height: u32) -> js_sys::Float32Array {
        let params = self.config.to_chess_params();
        let resp = chess_response_u8(pixels, width as usize, height as usize, &params);
        let arr = js_sys::Float32Array::new_with_length(resp.data().len() as u32);
        arr.copy_from(resp.data());
        self.last_response = Some(resp);
        arr
    }

    /// Compute the response map from RGBA pixels.
    pub fn response_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> js_sys::Float32Array {
        let gray = rgba_to_gray(pixels, width, height);
        self.response(&gray, width, height)
    }

    /// Width of the last computed response map.
    pub fn response_width(&self) -> u32 {
        self.last_response.as_ref().map_or(0, |r| r.width() as u32)
    }

    /// Height of the last computed response map.
    pub fn response_height(&self) -> u32 {
        self.last_response.as_ref().map_or(0, |r| r.height() as u32)
    }

    // ---- Radon heatmap ----

    /// Compute the whole-image Radon detector heatmap from grayscale
    /// pixels.
    ///
    /// Returns the dense `(max_α S_α − min_α S_α)²` Radon response as a
    /// row-major `Float32Array` at *working resolution* — that is,
    /// `width * upscale * radon_image_upsample` by the same in `y`.
    /// Use [`Self::radon_heatmap_width`] / [`Self::radon_heatmap_height`]
    /// to get the actual dimensions, and [`Self::radon_heatmap_scale`]
    /// for the working-to-input scale factor.
    ///
    /// Honours `set_upscale_factor`, `set_threshold` (via
    /// `radon_detector.threshold_*`), and other Radon tuning state on
    /// the detector. The detector mode does not need to be set to
    /// `"radon"` to call this — the heatmap is always computable.
    pub fn radon_heatmap(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> js_sys::Float32Array {
        let resp = radon_heatmap_u8(pixels, width, height, &self.config);
        let arr = js_sys::Float32Array::new_with_length(resp.data().len() as u32);
        arr.copy_from(resp.data());
        // Cache the scale alongside the response so the three
        // accessors (width / height / scale) stay mutually consistent
        // even if the caller mutates `set_upscale_factor` /
        // `radon_detector.image_upsample` after this call.
        let upscale = self.config.upscale.effective_factor().max(1);
        let radon_up = self.config.radon_detector.image_upsample.clamp(1, 2);
        self.last_radon_scale = upscale * radon_up;
        self.last_radon_response = Some(resp);
        arr
    }

    /// Compute the Radon heatmap from RGBA pixels (e.g. from canvas
    /// `getImageData`). Converts to grayscale internally.
    pub fn radon_heatmap_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> js_sys::Float32Array {
        let gray = rgba_to_gray(pixels, width, height);
        self.radon_heatmap(&gray, width, height)
    }

    /// Width of the last computed Radon heatmap (working resolution).
    pub fn radon_heatmap_width(&self) -> u32 {
        self.last_radon_response
            .as_ref()
            .map_or(0, |r| r.width() as u32)
    }

    /// Height of the last computed Radon heatmap (working resolution).
    pub fn radon_heatmap_height(&self) -> u32 {
        self.last_radon_response
            .as_ref()
            .map_or(0, |r| r.height() as u32)
    }

    /// Working-to-input scale factor for the last computed Radon heatmap.
    ///
    /// Multiply input-pixel coordinates by this factor to land on the
    /// corresponding heatmap pixel; divide heatmap-pixel coordinates by
    /// it to recover input pixels. Equals
    /// `upscale_factor * radon_detector.image_upsample` (clamped to
    /// the supported range) **as it was at the time of the last
    /// `radon_heatmap` call** — mutating the detector's upscale or
    /// `radon_detector.image_upsample` afterwards does not change
    /// this value, so the trio of width / height / scale stays
    /// consistent for overlay alignment. Returns `0` if no heatmap
    /// has been computed yet.
    pub fn radon_heatmap_scale(&self) -> u32 {
        self.last_radon_scale
    }
}

/// Flat array stride per corner: `[x, y, response, contrast, fit_rms,
/// axis0_angle, axis0_sigma, axis1_angle, axis1_sigma]`.
const CORNER_STRIDE: usize = 9;

fn corners_to_f32_array(corners: &[chess_corners::CornerDescriptor]) -> js_sys::Float32Array {
    let mut flat = Vec::with_capacity(corners.len() * CORNER_STRIDE);
    for c in corners {
        flat.push(c.x);
        flat.push(c.y);
        flat.push(c.response);
        flat.push(c.contrast);
        flat.push(c.fit_rms);
        flat.push(c.axes[0].angle);
        flat.push(c.axes[0].sigma);
        flat.push(c.axes[1].angle);
        flat.push(c.axes[1].sigma);
    }
    let arr = js_sys::Float32Array::new_with_length(flat.len() as u32);
    arr.copy_from(&flat);
    arr
}

// Helper used by `lib.rs` to wrap a Rust facade `ChessConfig` value
// in a fresh JS-facing wrapper. The returned wrapper owns brand-new
// cells (not shared with the source value) — this is intentional for
// `getConfig()` snapshot semantics.
impl ChessConfig {
    pub(crate) fn from_inner_for_js(inner: chess_corners::ChessConfig) -> Self {
        Self::from_value_pub(inner)
    }
}
