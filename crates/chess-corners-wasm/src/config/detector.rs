//! `#[wasm_bindgen]` wrapper for the top-level `DetectorConfig`.

use chess_corners::{DetectorConfig as RsDetectorConfig, OrientationMethod as RsOrientationMethod};
use wasm_bindgen::prelude::*;

use super::multiscale::MultiscaleConfig;
use super::refiners::ChessRefiner;
use super::strategy::{DetectionParams, DetectionStrategy};
use super::upscale::UpscaleConfig;
use super::{cell, Cell, ChessRing, OrientationMethod, PeakFitMode};

// ---------------------------------------------------------------------------
// DetectorConfig
// ---------------------------------------------------------------------------

/// High-level detector configuration. Mirrors
/// [`chess_corners::DetectorConfig`].
///
/// Build one with [`DetectorConfig::chess`],
/// [`DetectorConfig::chess_multiscale`], [`DetectorConfig::radon`], or
/// [`DetectorConfig::radon_multiscale`] and tweak only the fields you
/// need. In JS the factory names are camel-cased
/// (`DetectorConfig.chess()`, `DetectorConfig.chessMultiscale()`,
/// `DetectorConfig.radonMultiscale()`).
#[non_exhaustive]
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectorConfig {
    strategy: DetectionStrategy,
    threshold: f32,
    detection: DetectionParams,
    multiscale: MultiscaleConfig,
    upscale: UpscaleConfig,
    orientation_method: Cell<Option<RsOrientationMethod>>,
    merge_radius: Cell<f32>,
}

impl DetectorConfig {
    pub(crate) fn from_value_pub(value: RsDetectorConfig) -> Self {
        Self::from_value(value)
    }

    fn from_value(value: RsDetectorConfig) -> Self {
        Self {
            strategy: DetectionStrategy::from_value(value.strategy),
            threshold: value.threshold,
            detection: DetectionParams::from_value(value.detection),
            multiscale: MultiscaleConfig::from_value(value.multiscale),
            upscale: UpscaleConfig::from_value(value.upscale),
            orientation_method: cell(value.orientation_method),
            merge_radius: cell(value.merge_radius),
        }
    }

    /// Create a deep-independent copy by round-tripping through the Rust
    /// snapshot. Used by builder methods so edits on the returned config
    /// do not alias the source's cells.
    fn deep_clone(&self) -> Self {
        Self::from_value(self.snapshot())
    }

    /// Snapshot the current state into the Rust facade
    /// [`RsDetectorConfig`] for hand-off to the detector.
    pub(crate) fn snapshot(&self) -> RsDetectorConfig {
        let mut cfg = RsDetectorConfig::default();
        cfg.strategy = self.strategy.snapshot();
        cfg.threshold = self.threshold;
        cfg.detection = self.detection.snapshot();
        cfg.multiscale = self.multiscale.snapshot();
        cfg.upscale = self.upscale.snapshot();
        cfg.orientation_method = *self.orientation_method.borrow();
        cfg.merge_radius = *self.merge_radius.borrow();
        cfg
    }
}

#[wasm_bindgen]
impl DetectorConfig {
    /// Construct a `DetectorConfig` with library defaults
    /// (single-scale ChESS, absolute threshold = 0.0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsDetectorConfig::default())
    }

    /// Single-scale ChESS preset. JS: `DetectorConfig.chess()`.
    pub fn chess() -> Self {
        Self::from_value(RsDetectorConfig::chess())
    }

    /// Three-level coarse-to-fine ChESS preset. JS: `DetectorConfig.chessMultiscale()`.
    #[wasm_bindgen(js_name = chessMultiscale)]
    pub fn chess_multiscale() -> Self {
        Self::from_value(RsDetectorConfig::chess_multiscale())
    }

    /// Whole-image Radon detector preset (relative threshold 0.01).
    /// JS: `DetectorConfig.radon()`.
    pub fn radon() -> Self {
        Self::from_value(RsDetectorConfig::radon())
    }

    /// Coarse-to-fine Radon preset. JS: `DetectorConfig.radonMultiscale()`.
    #[wasm_bindgen(js_name = radonMultiscale)]
    pub fn radon_multiscale() -> Self {
        Self::from_value(RsDetectorConfig::radon_multiscale())
    }

    // ---- Chainable builder methods ----

    /// Return a copy of this config with the threshold replaced.
    /// JS: `cfg.withThreshold(0.15)`.
    #[wasm_bindgen(js_name = withThreshold)]
    pub fn with_threshold(&self, threshold: f32) -> Self {
        let mut out = self.deep_clone();
        out.threshold = threshold;
        out
    }

    /// Return a copy of this config with the multiscale setting replaced.
    /// JS: `cfg.withMultiscale(MultiscaleConfig.pyramidDefault())`.
    #[wasm_bindgen(js_name = withMultiscale)]
    pub fn with_multiscale(&self, multiscale: &MultiscaleConfig) -> Self {
        let mut out = self.deep_clone();
        out.set_multiscale(multiscale);
        out
    }

    /// Return a copy of this config with the upscale setting replaced.
    /// JS: `cfg.withUpscale(UpscaleConfig.fixed(2))`.
    #[wasm_bindgen(js_name = withUpscale)]
    pub fn with_upscale(&self, upscale: &UpscaleConfig) -> Self {
        let mut out = self.deep_clone();
        out.set_upscale(upscale);
        out
    }

    /// Return a copy of this config with the orientation method replaced.
    /// JS: `cfg.withOrientationMethod(OrientationMethod.DiskFit)`.
    #[wasm_bindgen(js_name = withOrientationMethod)]
    pub fn with_orientation_method(&self, method: OrientationMethod) -> Self {
        let mut out = self.deep_clone();
        out.set_orientation_method(Some(method));
        out
    }

    /// Return a copy of this config with the per-corner orientation fit
    /// skipped. Detection still yields positions and responses, but the
    /// four axis values per corner are `NaN`. JS: `cfg.withoutOrientation()`.
    #[wasm_bindgen(js_name = withoutOrientation)]
    pub fn without_orientation(&self) -> Self {
        let mut out = self.deep_clone();
        out.set_orientation_method(None);
        out
    }

    /// Return a copy of this config with the merge radius replaced.
    /// JS: `cfg.withMergeRadius(5.0)`.
    #[wasm_bindgen(js_name = withMergeRadius)]
    pub fn with_merge_radius(&self, radius: f32) -> Self {
        let mut out = self.deep_clone();
        out.set_merge_radius(radius);
        out
    }

    /// Return a copy of this config with the ChESS refiner replaced.
    ///
    /// Use this instead of the `refiner` key in `withChess({})` — wasm-bindgen
    /// Rust structs cannot be passed through plain `js_sys::Object` iteration.
    /// JS: `cfg.withChessRefiner(ChessRefiner.withForstner(new ForstnerConfig()))`.
    #[wasm_bindgen(js_name = withChessRefiner)]
    pub fn with_chess_refiner(&self, refiner: &ChessRefiner) -> Self {
        let mut out = self.deep_clone();
        if out.strategy.kind() != "chess" {
            out.strategy.use_chess();
        }
        out.strategy.chess().set_refiner(refiner);
        out
    }

    /// Return a copy of this config with ChESS strategy fields patched
    /// from a plain JS options object.
    ///
    /// Accepted keys (all optional):
    /// - `ring`: `ChessRing`
    ///
    /// To set the refiner use the typed [`Self::with_chess_refiner`] builder
    /// instead — wasm-bindgen Rust structs cannot be passed via plain options
    /// objects. The shared NMS / clustering knobs moved to
    /// [`Self::with_detection`].
    ///
    /// Unknown keys throw `Error("unexpected option: '<key>'")`.
    /// JS: `cfg.withChess({ ring: ChessRing.Broad })`.
    #[wasm_bindgen(js_name = withChess)]
    pub fn with_chess(&self, opts: &js_sys::Object) -> Result<DetectorConfig, JsValue> {
        let mut out = self.deep_clone();
        // Ensure the strategy is Chess; switch if currently Radon.
        if out.strategy.kind() != "chess" {
            out.strategy.use_chess();
        }
        let keys = js_sys::Object::keys(opts);
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().unwrap_or_default();
            let val = js_sys::Reflect::get(opts, &key)?;
            match key_str.as_str() {
                "refiner" => {
                    apply_chess_refiner_from_js(&mut out, val)?;
                }
                "ring" => {
                    let disc = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("ring must be a ChessRing enum value"))?
                        as u8;
                    let ring = if disc == ChessRing::Broad as u8 {
                        ChessRing::Broad
                    } else {
                        ChessRing::Canonical
                    };
                    out.strategy.chess().set_ring(ring);
                }
                other => {
                    return Err(JsValue::from_str(&format!("unexpected option: '{other}'")));
                }
            }
        }
        Ok(out)
    }

    /// Return a copy of this config with Radon strategy fields patched
    /// from a plain JS options object.
    ///
    /// Accepted keys (all optional):
    /// - `rayRadius`: integer
    /// - `imageUpsample`: integer
    /// - `responseBlurRadius`: integer
    /// - `peakFit`: `PeakFitMode`
    ///
    /// Unknown keys throw `Error("unexpected option: '<key>'")`.
    /// JS: `cfg.withRadon({ rayRadius: 6, imageUpsample: 2, responseBlurRadius: 1, peakFit: PeakFitMode.Gaussian })`.
    #[wasm_bindgen(js_name = withRadon)]
    pub fn with_radon(&self, opts: &js_sys::Object) -> Result<DetectorConfig, JsValue> {
        let mut out = self.deep_clone();
        // Ensure the strategy is Radon; switch if currently Chess.
        if out.strategy.kind() != "radon" {
            out.strategy.use_radon();
        }
        let keys = js_sys::Object::keys(opts);
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().unwrap_or_default();
            let val = js_sys::Reflect::get(opts, &key)?;
            match key_str.as_str() {
                "rayRadius" => {
                    let r = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("rayRadius must be a number"))?
                        as u32;
                    out.strategy.radon().set_ray_radius(r);
                }
                "imageUpsample" => {
                    let r = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("imageUpsample must be a number"))?
                        as u32;
                    out.strategy.radon().set_image_upsample(r);
                }
                "responseBlurRadius" => {
                    let r = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("responseBlurRadius must be a number"))?
                        as u32;
                    out.strategy.radon().set_response_blur_radius(r);
                }
                "peakFit" => {
                    let disc = val.as_f64().ok_or_else(|| {
                        JsValue::from_str("peakFit must be a PeakFitMode enum value")
                    })? as u8;
                    let mode = if disc == PeakFitMode::Gaussian as u8 {
                        PeakFitMode::Gaussian
                    } else {
                        PeakFitMode::Parabolic
                    };
                    out.strategy.radon().set_peak_fit(mode);
                }
                other => {
                    return Err(JsValue::from_str(&format!("unexpected option: '{other}'")));
                }
            }
        }
        Ok(out)
    }

    /// Return a copy of this config with the shared detection params
    /// (NMS / clustering thresholds honoured by both strategies) patched
    /// from a plain JS options object.
    ///
    /// Accepted keys (all optional):
    /// - `nmsRadius`: integer
    /// - `minClusterSize`: integer
    ///
    /// Unknown keys throw `Error("unexpected option: '<key>'")`.
    /// JS: `cfg.withDetection({ nmsRadius: 4, minClusterSize: 2 })`.
    #[wasm_bindgen(js_name = withDetection)]
    pub fn with_detection(&self, opts: &js_sys::Object) -> Result<DetectorConfig, JsValue> {
        let out = self.deep_clone();
        let keys = js_sys::Object::keys(opts);
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().unwrap_or_default();
            let val = js_sys::Reflect::get(opts, &key)?;
            match key_str.as_str() {
                "nmsRadius" => {
                    let r = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("nmsRadius must be a number"))?
                        as u32;
                    out.detection().set_nms_radius(r);
                }
                "minClusterSize" => {
                    let r = val
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("minClusterSize must be a number"))?
                        as u32;
                    out.detection().set_min_cluster_size(r);
                }
                other => {
                    return Err(JsValue::from_str(&format!("unexpected option: '{other}'")));
                }
            }
        }
        Ok(out)
    }

    // ---- Top-level fields ----

    #[wasm_bindgen(getter)]
    pub fn strategy(&self) -> DetectionStrategy {
        self.strategy.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_strategy(&mut self, v: &DetectionStrategy) {
        self.strategy.copy_from(v);
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, v: f32) {
        self.threshold = v;
    }

    /// Shared NMS / clustering thresholds. Returns a wrapper backed by
    /// the same cells as the parent; edits propagate without a
    /// round-trip. Honoured by both ChESS and Radon strategies.
    #[wasm_bindgen(getter)]
    pub fn detection(&self) -> DetectionParams {
        self.detection.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_detection(&mut self, v: &DetectionParams) {
        self.detection.copy_from(v);
    }

    /// Coarse-to-fine multiscale configuration. Returns a wrapper
    /// backed by the same cells as the parent; edits propagate
    /// without a round-trip. Honoured by both ChESS and Radon
    /// strategies.
    #[wasm_bindgen(getter)]
    pub fn multiscale(&self) -> MultiscaleConfig {
        self.multiscale.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_multiscale(&mut self, v: &MultiscaleConfig) {
        self.multiscale.copy_from(v);
    }

    /// Pre-pipeline integer upscaling configuration.
    #[wasm_bindgen(getter)]
    pub fn upscale(&self) -> UpscaleConfig {
        self.upscale.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_upscale(&mut self, v: &UpscaleConfig) {
        self.upscale.copy_from(v);
    }

    #[wasm_bindgen(getter, js_name = orientationMethod)]
    pub fn orientation_method(&self) -> Option<OrientationMethod> {
        (*self.orientation_method.borrow()).map(Into::into)
    }
    #[wasm_bindgen(setter, js_name = orientationMethod)]
    pub fn set_orientation_method(&mut self, v: Option<OrientationMethod>) {
        *self.orientation_method.borrow_mut() = v.map(Into::into);
    }

    #[wasm_bindgen(getter, js_name = mergeRadius)]
    pub fn merge_radius(&self) -> f32 {
        *self.merge_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = mergeRadius)]
    pub fn set_merge_radius(&mut self, v: f32) {
        *self.merge_radius.borrow_mut() = v;
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers for with_chess / with_radon options objects
// ---------------------------------------------------------------------------

/// Reject an `refiner` key coming from the `with_chess` or `with_radon` options
/// object. wasm-bindgen Rust structs tagged with `#[wasm_bindgen]` are opaque
/// pointers to JS — they cannot be extracted from a plain `JsValue` via
/// `JsCast::dyn_ref`. Callers should use the dedicated typed builder methods
/// `withChessRefiner` instead.
fn apply_chess_refiner_from_js(_cfg: &mut DetectorConfig, _val: JsValue) -> Result<(), JsValue> {
    Err(JsValue::from_str(
        "refiner cannot be set via the options object; use .withChessRefiner(refiner) instead",
    ))
}
