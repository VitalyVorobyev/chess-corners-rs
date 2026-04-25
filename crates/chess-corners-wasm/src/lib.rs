use chess_corners::{
    radon_heatmap_u8, ChessConfig, DetectorMode, PyramidBuffers, RefinementMethod, ThresholdMode,
    UpscaleConfig,
};
use chess_corners_core::response::chess_response_u8;
use chess_corners_core::ResponseMap;
use wasm_bindgen::prelude::*;

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
    config: ChessConfig,
    buffers: PyramidBuffers,
    last_response: Option<ResponseMap>,
    last_radon_response: Option<ResponseMap>,
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
            config: ChessConfig::single_scale(),
            buffers: PyramidBuffers::with_capacity(1),
            last_response: None,
            last_radon_response: None,
        }
    }

    /// Create a detector with the recommended multiscale preset.
    pub fn multiscale() -> Self {
        let config = ChessConfig::multiscale();
        let levels = config.pyramid_levels;
        Self {
            config,
            buffers: PyramidBuffers::with_capacity(levels),
            last_response: None,
            last_radon_response: None,
        }
    }

    // ---- Config setters ----

    /// Set the relative threshold (fraction of max response, default 0.2).
    pub fn set_threshold(&mut self, rel: f32) {
        self.config.threshold_mode = ThresholdMode::Relative;
        self.config.threshold_value = rel;
    }

    /// Set the non-maximum suppression radius (default 2).
    pub fn set_nms_radius(&mut self, r: u32) {
        self.config.nms_radius = r;
    }

    /// Toggle the large r=10 ring (default: r=5).
    pub fn set_broad_mode(&mut self, v: bool) {
        self.config.detector_mode = if v {
            DetectorMode::Broad
        } else {
            DetectorMode::Canonical
        };
    }

    /// Select the detector kernel: `"canonical"`, `"broad"`, or
    /// `"radon"`. `canonical` / `broad` are the two ChESS variants;
    /// `radon` picks the whole-image Duda-Frese detector, useful
    /// under heavy blur or low contrast.
    pub fn set_detector_mode(&mut self, name: &str) -> Result<(), JsValue> {
        self.config.detector_mode = match name {
            "canonical" => DetectorMode::Canonical,
            "broad" => DetectorMode::Broad,
            "radon" => DetectorMode::Radon,
            _ => {
                return Err(JsValue::from_str(
                    "unknown detector_mode: use canonical, broad, or radon",
                ))
            }
        };
        Ok(())
    }

    /// Set the minimum cluster size for accepting a corner (default 2).
    pub fn set_min_cluster_size(&mut self, v: u32) {
        self.config.min_cluster_size = v;
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
            0 | 1 => UpscaleConfig::disabled(),
            2..=4 => UpscaleConfig::fixed(factor),
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
                "center_of_mass" => RefinementMethod::CenterOfMass,
                "forstner" => RefinementMethod::Forstner,
                "saddle_point" => RefinementMethod::SaddlePoint,
                "radon_peak" => RefinementMethod::RadonPeak,
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
    /// `upscale_factor * radon_detector.image_upsample` (clamped to the
    /// supported range).
    pub fn radon_heatmap_scale(&self) -> u32 {
        let upscale = self.config.upscale.effective_factor().max(1);
        let radon_up = self.config.radon_detector.image_upsample.clamp(1, 2);
        upscale * radon_up
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
