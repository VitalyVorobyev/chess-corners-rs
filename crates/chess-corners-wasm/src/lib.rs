//! WebAssembly bindings for the ChESS / Radon corner detector.
//!
//! Two ways to drive the detector from JS:
//!
//! 1. **Setter shortcuts** — `new ChessDetector()` then
//!    `det.set_threshold(...)` / `det.set_pyramid_levels(...)` /
//!    `det.set_detector_mode("radon")` etc. Useful for quick demos.
//! 2. **Typed [`DetectorConfig`]** — construct the typed config (with
//!    nested [`RefinerConfig`], [`DetectionStrategy`],
//!    [`UpscaleConfig`], …), then pass it to
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
//! cfg.refiner.kind = RefinementMethod.RadonPeak;       // propagates
//! cfg.refiner.forstner.maxOffset = 2.0;                // propagates
//! cfg.strategy.chess.nmsRadius = 3;                    // propagates
//! cfg.multiscale.pyramidLevels = 4;                    // propagates
//! ```
//!
//! See [`config`] for the cell-sharing details.

pub mod config;

use chess_corners::{
    chess_response_u8, radon_heatmap_u8, ChessConfig as RsChessConfig,
    DetectionStrategy as RsDetectionStrategy, Detector as RsDetector,
    MultiscaleParams as RsMultiscaleParams, RefinementMethod as RsRefinementMethod, ResponseMap,
    Threshold as RsThreshold, UpscaleConfig as RsUpscaleConfig,
};
use wasm_bindgen::prelude::*;

pub use crate::config::{
    CenterOfMassConfig, ChessRing, ChessStrategy, DescriptorMode, DetectionStrategy,
    DetectorConfig, ForstnerConfig, MultiscaleParams, OrientationMethod, PeakFitMode,
    RadonPeakConfig, RadonStrategy, RefinementMethod, RefinerConfig, SaddlePointConfig, Threshold,
    UpscaleConfig, UpscaleMode,
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
#[non_exhaustive]
#[wasm_bindgen]
pub struct ChessDetector {
    inner: RsDetector,
    last_response: Option<ResponseMap>,
    last_radon_response: Option<ResponseMap>,
    /// Working-to-input scale factor cached at the moment
    /// `radon_heatmap` produced `last_radon_response`. Returning a
    /// cached value (instead of recomputing from the live config)
    /// keeps `radon_heatmap_width` / `_height` / `_scale` mutually
    /// consistent if the caller mutates the detector's upscale or
    /// the Radon `image_upsample` between the heatmap call and the
    /// accessor calls. `0` until the first heatmap is computed.
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
            inner: RsDetector::new(RsChessConfig::single_scale())
                .expect("single_scale config is always valid"),
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        }
    }

    /// Create a detector with the recommended multiscale preset.
    pub fn multiscale() -> Self {
        Self {
            inner: RsDetector::new(RsChessConfig::multiscale())
                .expect("multiscale preset is always valid"),
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        }
    }

    /// Create a detector seeded from a typed [`DetectorConfig`]. The
    /// full public config surface — refiner subconfigs, strategy
    /// branches, descriptor mode, upscale — is reachable through the
    /// typed object (see module docs for the alternative setter-
    /// shortcut path).
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: &DetectorConfig) -> Result<ChessDetector, JsValue> {
        let snapshot = config.snapshot();
        let inner = RsDetector::new(snapshot).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            inner,
            last_response: None,
            last_radon_response: None,
            last_radon_scale: 0,
        })
    }

    /// Snapshot the current configuration as a typed [`DetectorConfig`].
    /// The returned object is a snapshot — its cells are *not* shared
    /// with the detector's live state. Use [`Self::apply_config`] to
    /// commit edits made on the snapshot.
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> DetectorConfig {
        DetectorConfig::from_inner_for_js(self.inner.config().clone())
    }

    /// Replace the detector's configuration with the given typed
    /// [`DetectorConfig`]. Resizes pyramid scratch buffers if the
    /// multiscale settings changed.
    #[wasm_bindgen(js_name = applyConfig)]
    pub fn apply_config(&mut self, config: &DetectorConfig) -> Result<(), JsValue> {
        let snapshot = config.snapshot();
        self.inner
            .set_config(snapshot)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ---- Config setters ----
    //
    // These shortcuts route writes to the active strategy branch.
    // The typed [`DetectorConfig`] is the canonical API — prefer
    // `cfg.strategy.chess.nmsRadius = N` etc. for new code. Shortcuts
    // exist because they're convenient for one-liner demos.

    /// Switch the threshold to `Relative(rel)`.
    ///
    /// The same enum is honoured by both detectors, so this is a
    /// single source of truth — no per-detector mirroring is needed.
    pub fn set_threshold(&mut self, rel: f32) {
        self.inner.config_mut().threshold = RsThreshold::Relative(rel);
    }

    /// Select the detection strategy: `"chess"` or `"radon"`.
    ///
    /// Switching to `"chess"` installs a default [`ChessStrategy`]
    /// (single-scale, canonical ring, threshold left untouched).
    /// Switching to `"radon"` installs a default [`RadonStrategy`].
    /// Any prior per-branch tuning is replaced — for incremental
    /// edits use the typed config (`cfg.strategy = ...`).
    pub fn set_detector_mode(&mut self, name: &str) -> Result<(), JsValue> {
        let strategy = match name {
            "chess" => RsDetectionStrategy::Chess(Default::default()),
            "radon" => RsDetectionStrategy::Radon(Default::default()),
            _ => {
                return Err(JsValue::from_str(
                    "unknown detector mode: use \"chess\" or \"radon\"",
                ));
            }
        };
        self.inner.config_mut().strategy = strategy;
        Ok(())
    }

    /// Toggle the broad (r=10) ChESS sampling ring.
    ///
    /// Only meaningful when the active strategy is ChESS. Returns an
    /// error if the strategy is Radon — switch via
    /// `set_detector_mode("chess")` first.
    pub fn set_broad_mode(&mut self, v: bool) -> Result<(), JsValue> {
        match &mut self.inner.config_mut().strategy {
            RsDetectionStrategy::Chess(chess) => {
                chess.ring = if v {
                    chess_corners::ChessRing::Broad
                } else {
                    chess_corners::ChessRing::Canonical
                };
                Ok(())
            }
            _ => Err(JsValue::from_str(
                "set_broad_mode is only valid when the active strategy is chess",
            )),
        }
    }

    /// Set the non-maximum suppression radius.
    ///
    /// Writes to whichever strategy is active. The ChESS `nms_radius`
    /// is measured in input-image pixels; the Radon `nms_radius` is
    /// measured in working-resolution pixels (post-`imageUpsample`).
    pub fn set_nms_radius(&mut self, r: u32) {
        match &mut self.inner.config_mut().strategy {
            RsDetectionStrategy::Chess(chess) => chess.nms_radius = r,
            RsDetectionStrategy::Radon(radon) => radon.nms_radius = r,
            _ => {}
        }
    }

    /// Set the minimum cluster size required to accept a corner.
    /// Writes to whichever strategy is active.
    pub fn set_min_cluster_size(&mut self, v: u32) {
        match &mut self.inner.config_mut().strategy {
            RsDetectionStrategy::Chess(chess) => chess.min_cluster_size = v,
            RsDetectionStrategy::Radon(radon) => radon.min_cluster_size = v,
            _ => {}
        }
    }

    /// Set the number of pyramid levels (1 = single-scale, ≥2 = multiscale).
    ///
    /// Auto-promotes from single-scale to multiscale by attaching a
    /// defaulted [`MultiscaleParams`] if none is currently attached.
    /// Honoured by both ChESS and Radon strategies.
    /// Returns an error if `n == 0`.
    pub fn set_pyramid_levels(&mut self, n: u8) -> Result<(), JsValue> {
        if n == 0 {
            return Err(JsValue::from_str("pyramid_levels must be >= 1"));
        }
        let cfg = self.inner.config_mut();
        let ms = cfg
            .multiscale
            .get_or_insert_with(RsMultiscaleParams::default);
        ms.pyramid_levels = n;
        Ok(())
    }

    /// Set the minimum pyramid-level size in pixels (default 128).
    ///
    /// Auto-promotes single-scale → multiscale just like
    /// [`Self::set_pyramid_levels`]. Honoured by both ChESS and Radon.
    pub fn set_pyramid_min_size(&mut self, v: u32) -> Result<(), JsValue> {
        let cfg = self.inner.config_mut();
        let ms = cfg
            .multiscale
            .get_or_insert_with(RsMultiscaleParams::default);
        ms.pyramid_min_size = v as usize;
        Ok(())
    }

    /// Set the optional pre-pipeline upscale factor.
    ///
    /// `factor == 0` or `factor == 1` disables upscaling. Accepts
    /// integer factors 2, 3, 4. Corner coordinates are always returned
    /// in input-image pixel space; callers do not need to rescale.
    pub fn set_upscale_factor(&mut self, factor: u32) -> Result<(), JsValue> {
        self.inner.config_mut().upscale = match factor {
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
        self.inner.config_mut().refiner.kind =
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
    ///
    /// Throws a JS error string if the pixel buffer length does not match
    /// `width * height` or if the upscale configuration is invalid.
    pub fn detect(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        let corners = self
            .inner
            .detect_u8(pixels, width, height)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(corners_to_f32_array(&corners))
    }

    /// Detect corners from RGBA pixels (e.g. from canvas `getImageData`).
    ///
    /// Converts to grayscale internally, then detects.
    pub fn detect_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        let gray = rgba_to_gray(pixels, width, height);
        self.detect(&gray, width, height)
    }

    // ---- Response map ----

    /// Compute the raw ChESS response map from grayscale pixels.
    ///
    /// Returns a `Float32Array` in row-major order (width x height).
    pub fn response(&mut self, pixels: &[u8], width: u32, height: u32) -> js_sys::Float32Array {
        let params = self.inner.config().to_chess_params();
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
    /// Honours `set_upscale_factor`, `set_threshold`, and the active
    /// Radon strategy tuning on the detector. The detector mode does
    /// not need to be `"radon"` to call this — the heatmap is always
    /// computable from the current effective Radon params.
    pub fn radon_heatmap(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        let resp = radon_heatmap_u8(pixels, width, height, self.inner.config())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let arr = js_sys::Float32Array::new_with_length(resp.data().len() as u32);
        arr.copy_from(resp.data());
        // Cache the scale alongside the response so the three
        // accessors (width / height / scale) stay mutually consistent
        // even if the caller mutates upscale / Radon image_upsample
        // after this call.
        let upscale = self.inner.config().upscale.effective_factor().max(1);
        let radon_up = self
            .inner
            .config()
            .to_radon_detector_params()
            .image_upsample
            .clamp(1, 2);
        self.last_radon_scale = upscale * radon_up;
        self.last_radon_response = Some(resp);
        Ok(arr)
    }

    /// Compute the Radon heatmap from RGBA pixels (e.g. from canvas
    /// `getImageData`). Converts to grayscale internally.
    pub fn radon_heatmap_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
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
    /// `upscale_factor * radon_image_upsample` (clamped to the
    /// supported range) **as it was at the time of the last
    /// `radon_heatmap` call** — mutating the detector's upscale or
    /// the Radon image_upsample afterwards does not change this
    /// value, so the trio of width / height / scale stays consistent
    /// for overlay alignment. Returns `0` if no heatmap has been
    /// computed yet.
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

// Helper used by `lib.rs` to wrap a Rust facade `DetectorConfig` value
// in a fresh JS-facing wrapper. The returned wrapper owns brand-new
// cells (not shared with the source value) — this is intentional for
// `getConfig()` snapshot semantics.
impl DetectorConfig {
    pub(crate) fn from_inner_for_js(inner: chess_corners::ChessConfig) -> Self {
        Self::from_value_pub(inner)
    }
}
