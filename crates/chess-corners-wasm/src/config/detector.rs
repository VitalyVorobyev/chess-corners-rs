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

    /// Build from a plain JS value shaped like `chess_corners::DetectorConfig`
    /// (as produced by `JSON.parse`). Missing fields take their defaults;
    /// unknown keys are ignored (serde's default behaviour).
    fn from_js_value(value: JsValue) -> Result<Self, JsError> {
        serde_wasm_bindgen::from_value::<RsDetectorConfig>(value)
            .map(Self::from_value)
            .map_err(|e| JsError::new(&format!("invalid DetectorConfig JSON: {}", error_text(&e))))
    }

    /// Serialize to JSON text in the shape of `chess_corners::DetectorConfig`.
    pub(crate) fn to_json_text(&self) -> Result<String, JsError> {
        config_to_json(&self.snapshot())
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
    /// Construct a `DetectorConfig` with library defaults — equivalent to
    /// [`Self::chess`] (single-scale ChESS; see that preset's absolute
    /// threshold).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsDetectorConfig::default())
    }

    /// Single-scale ChESS preset (absolute threshold 30.0). JS: `DetectorConfig.chess()`.
    pub fn chess() -> Self {
        Self::from_value(RsDetectorConfig::chess())
    }

    /// Three-level coarse-to-fine ChESS preset. JS: `DetectorConfig.chessMultiscale()`.
    #[wasm_bindgen(js_name = chessMultiscale)]
    pub fn chess_multiscale() -> Self {
        Self::from_value(RsDetectorConfig::chess_multiscale())
    }

    /// Whole-image Radon detector preset (relative threshold 0.28, the
    /// `chess-corners-core` `RadonDetectorParams::DEFAULT_THRESHOLD_REL`).
    /// JS: `DetectorConfig.radon()`.
    pub fn radon() -> Self {
        Self::from_value(RsDetectorConfig::radon())
    }

    /// Coarse-to-fine Radon preset. JS: `DetectorConfig.radonMultiscale()`.
    #[wasm_bindgen(js_name = radonMultiscale)]
    pub fn radon_multiscale() -> Self {
        Self::from_value(RsDetectorConfig::radon_multiscale())
    }

    // ---- JSON interchange ----

    /// Parse a `DetectorConfig` from JSON in the same shape as
    /// `schemas/detector_config.json` shipped in the npm package (the
    /// serde shape of `chess_corners::DetectorConfig`: snake_case keys,
    /// externally tagged enums such as `{"chess": {...}}`).
    ///
    /// Every field is optional; omitted fields take their library
    /// defaults, so `"{}"` yields the default config, and unknown keys are
    /// ignored. Throws an `Error` when the text is not valid JSON (the
    /// engine's `SyntaxError`) or a value has the wrong type or an unknown
    /// enum variant (message such as `invalid type: string "high",
    /// expected f32`). Semantic validation (for example the
    /// allowed upscale factors) happens when the config is handed to
    /// `ChessDetector.withConfig` / `applyConfig`.
    /// JS: `DetectorConfig.fromJson('{"threshold": 40}')`.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<DetectorConfig, JsValue> {
        // Parsing is delegated to the host's `JSON.parse`, so a syntax error
        // surfaces as the engine's own `SyntaxError` with its position info.
        let value = js_sys::JSON::parse(json)?;
        Ok(Self::from_js_value(value)?)
    }

    /// Serialize this config to JSON (compact, same shape as
    /// `schemas/detector_config.json`). Round-trips through
    /// [`Self::from_json`]. JS: `JSON.parse(cfg.toJson())`.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        self.to_json_text()
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

/// `serde-wasm-bindgen` errors display as `Error: <message>`; drop the prefix.
fn error_text(e: &serde_wasm_bindgen::Error) -> String {
    let text = e.to_string();
    text.strip_prefix("Error: ")
        .map_or(text.clone(), str::to_owned)
}

/// Serialize a facade config to JSON text via `serde-wasm-bindgen` and the
/// host's `JSON.stringify`.
///
/// `serde-wasm-bindgen` widens `f32` to `f64`, so `0.28_f32` would print as
/// `0.2800000011920929`. [`shorten_f32`] restores the shortest decimal that
/// still names the same `f32`, matching the committed JSON Schema defaults.
pub(crate) fn config_to_json(cfg: &RsDetectorConfig) -> Result<String, JsError> {
    let value = serde_wasm_bindgen::to_value(cfg).map_err(|e| {
        JsError::new(&format!(
            "failed to serialize DetectorConfig: {}",
            error_text(&e)
        ))
    })?;
    shorten_f32(&value);
    let text = js_sys::JSON::stringify(&value)
        .map_err(|_| JsError::new("failed to serialize DetectorConfig: JSON.stringify threw"))?;
    Ok(String::from(text))
}

/// Recursively replace each non-integer number that is exactly an `f32`
/// with the shortest decimal string (<= 9 significant digits) that parses
/// back to the same `f32`. Config values are plain objects and scalars.
fn shorten_f32(value: &JsValue) {
    if !value.is_object() {
        return;
    }
    let keys = js_sys::Object::keys(value.unchecked_ref::<js_sys::Object>());
    for i in 0..keys.length() {
        let key = keys.get(i);
        let Ok(child) = js_sys::Reflect::get(value, &key) else {
            continue;
        };
        if let Some(n) = child.as_f64() {
            if let Some(short) = shortest_f32_decimal(n) {
                let _ = js_sys::Reflect::set(value, &key, &JsValue::from_f64(short));
            }
        } else {
            shorten_f32(&child);
        }
    }
}

fn shortest_f32_decimal(n: f64) -> Option<f64> {
    let as_f32 = n as f32;
    if !n.is_finite() || n.fract() == 0.0 || f64::from(as_f32) != n {
        return None;
    }
    let num = js_sys::Number::from(n);
    (1..=9u8).find_map(|digits| {
        let text = num.to_precision(digits).ok()?;
        let candidate = js_sys::parse_float(&String::from(text));
        (candidate as f32 == as_f32).then_some(candidate)
    })
}

#[cfg(test)]
mod json_tests {
    //! Off-wasm tests pin the serde contract that `fromJson` / `toJson`
    //! delegate to (the glue itself needs a JS host; it is exercised by
    //! `crates/chess-corners-wasm/tests/smoke.mjs` against the built package).
    use super::*;

    fn presets() -> [(&'static str, RsDetectorConfig); 4] {
        [
            ("chess", RsDetectorConfig::chess()),
            ("chess_multiscale", RsDetectorConfig::chess_multiscale()),
            ("radon", RsDetectorConfig::radon()),
            ("radon_multiscale", RsDetectorConfig::radon_multiscale()),
        ]
    }

    #[test]
    fn presets_round_trip_through_wrapper_and_json() {
        for (name, preset) in presets() {
            let snapshot = DetectorConfig::from_value(preset).snapshot();
            assert_eq!(snapshot, preset, "{name}: wrapper round trip");
            let json = serde_json::to_string(&snapshot).unwrap();
            let back: RsDetectorConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(back, preset, "{name}: {json}");
        }
    }

    #[test]
    fn json_is_snake_case_externally_tagged() {
        let json = serde_json::to_value(
            DetectorConfig::from_value(RsDetectorConfig::radon_multiscale()).snapshot(),
        )
        .unwrap();
        assert!(json.get("merge_radius").is_some(), "{json}");
        assert!(json["strategy"].get("radon").is_some(), "{json}");
        assert!(json["multiscale"].get("pyramid").is_some(), "{json}");
        assert_eq!(json["upscale"], "disabled");
    }

    #[test]
    fn empty_and_partial_json_fill_defaults() {
        let empty: RsDetectorConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, RsDetectorConfig::default());
        let partial: RsDetectorConfig = serde_json::from_str(
            r#"{"threshold": 55.0, "strategy": {"radon": {"ray_radius": 6}}}"#,
        )
        .unwrap();
        assert_eq!(partial.threshold, 55.0);
        match partial.strategy {
            chess_corners::DetectionStrategy::Radon(r) => {
                assert_eq!(r.ray_radius, 6);
                assert_eq!(r.image_upsample, 2, "unspecified field keeps default");
            }
            other => panic!("expected radon strategy, got {other:?}"),
        }
    }
}
