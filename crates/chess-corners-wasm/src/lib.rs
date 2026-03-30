use chess_corners::{ChessConfig, DetectorMode, PyramidBuffers, RefinementMethod, ThresholdMode};
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

    /// Set the subpixel refiner: "center_of_mass", "forstner", or "saddle_point".
    pub fn set_refiner(&mut self, name: &str) -> Result<(), JsValue> {
        self.config.refiner.kind = match name {
            "center_of_mass" => RefinementMethod::CenterOfMass,
            "forstner" => RefinementMethod::Forstner,
            "saddle_point" => RefinementMethod::SaddlePoint,
            _ => {
                return Err(JsValue::from_str(
                    "unknown refiner: use center_of_mass, forstner, or saddle_point",
                ))
            }
        };
        Ok(())
    }

    // ---- Detection ----

    /// Detect corners from grayscale u8 pixels.
    ///
    /// Returns a `Float32Array` with stride 4: `[x, y, response, orientation, ...]`.
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
}

fn corners_to_f32_array(corners: &[chess_corners::CornerDescriptor]) -> js_sys::Float32Array {
    let mut flat = Vec::with_capacity(corners.len() * 4);
    for c in corners {
        flat.push(c.x);
        flat.push(c.y);
        flat.push(c.response);
        flat.push(c.orientation);
    }
    let arr = js_sys::Float32Array::new_with_length(flat.len() as u32);
    arr.copy_from(&flat);
    arr
}
