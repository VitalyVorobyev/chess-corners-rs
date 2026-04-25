//! Typed `#[wasm_bindgen]` wrappers around `chess-corners` config
//! structs.
//!
//! Each wrapper stores its inner Rust value by value and exposes
//! per-field getters/setters with TypeScript-friendly types. The
//! wrappers do not own any state beyond the inner value — passing a
//! wrapper into `ChessDetector::with_config` clones the inner Rust
//! struct.
//!
//! Why a wrapper layer at all? The Rust source-of-truth structs live
//! in `chess-corners` / `chess-corners-core` and must not depend on
//! `wasm-bindgen` (per the workspace dependency rule in `AGENTS.md`).
//! These wrappers add the JS-facing attribute layer in the WASM
//! crate only.

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessConfig as RsChessConfig,
    DescriptorMode as RsDescriptorMode, DetectorMode as RsDetectorMode,
    ForstnerConfig as RsForstnerConfig, PeakFitMode as RsPeakFitMode,
    RadonDetectorParams as RsRadonDetectorParams, RadonPeakConfig as RsRadonPeakConfig,
    RefinementMethod as RsRefinementMethod, RefinerConfig as RsRefinerConfig,
    SaddlePointConfig as RsSaddlePointConfig, ThresholdMode as RsThresholdMode,
    UpscaleConfig as RsUpscaleConfig, UpscaleMode as RsUpscaleMode,
};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Detector kernel selection. Mirrors [`chess_corners::DetectorMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorMode {
    Canonical = 0,
    Broad = 1,
    Radon = 2,
}

impl From<DetectorMode> for RsDetectorMode {
    fn from(v: DetectorMode) -> Self {
        match v {
            DetectorMode::Canonical => RsDetectorMode::Canonical,
            DetectorMode::Broad => RsDetectorMode::Broad,
            DetectorMode::Radon => RsDetectorMode::Radon,
        }
    }
}

impl From<RsDetectorMode> for DetectorMode {
    fn from(v: RsDetectorMode) -> Self {
        match v {
            RsDetectorMode::Canonical => DetectorMode::Canonical,
            RsDetectorMode::Broad => DetectorMode::Broad,
            RsDetectorMode::Radon => DetectorMode::Radon,
        }
    }
}

/// Descriptor sampling override. Mirrors [`chess_corners::DescriptorMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorMode {
    FollowDetector = 0,
    Canonical = 1,
    Broad = 2,
}

impl From<DescriptorMode> for RsDescriptorMode {
    fn from(v: DescriptorMode) -> Self {
        match v {
            DescriptorMode::FollowDetector => RsDescriptorMode::FollowDetector,
            DescriptorMode::Canonical => RsDescriptorMode::Canonical,
            DescriptorMode::Broad => RsDescriptorMode::Broad,
        }
    }
}

impl From<RsDescriptorMode> for DescriptorMode {
    fn from(v: RsDescriptorMode) -> Self {
        match v {
            RsDescriptorMode::FollowDetector => DescriptorMode::FollowDetector,
            RsDescriptorMode::Canonical => DescriptorMode::Canonical,
            RsDescriptorMode::Broad => DescriptorMode::Broad,
        }
    }
}

/// Threshold interpretation. Mirrors [`chess_corners::ThresholdMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdMode {
    Relative = 0,
    Absolute = 1,
}

impl From<ThresholdMode> for RsThresholdMode {
    fn from(v: ThresholdMode) -> Self {
        match v {
            ThresholdMode::Relative => RsThresholdMode::Relative,
            ThresholdMode::Absolute => RsThresholdMode::Absolute,
        }
    }
}

impl From<RsThresholdMode> for ThresholdMode {
    fn from(v: RsThresholdMode) -> Self {
        match v {
            RsThresholdMode::Relative => ThresholdMode::Relative,
            RsThresholdMode::Absolute => ThresholdMode::Absolute,
        }
    }
}

/// Subpixel refinement algorithm. Mirrors
/// [`chess_corners::RefinementMethod`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementMethod {
    CenterOfMass = 0,
    Forstner = 1,
    SaddlePoint = 2,
    RadonPeak = 3,
}

impl From<RefinementMethod> for RsRefinementMethod {
    fn from(v: RefinementMethod) -> Self {
        match v {
            RefinementMethod::CenterOfMass => RsRefinementMethod::CenterOfMass,
            RefinementMethod::Forstner => RsRefinementMethod::Forstner,
            RefinementMethod::SaddlePoint => RsRefinementMethod::SaddlePoint,
            RefinementMethod::RadonPeak => RsRefinementMethod::RadonPeak,
        }
    }
}

impl From<RsRefinementMethod> for RefinementMethod {
    fn from(v: RsRefinementMethod) -> Self {
        match v {
            RsRefinementMethod::CenterOfMass => RefinementMethod::CenterOfMass,
            RsRefinementMethod::Forstner => RefinementMethod::Forstner,
            RsRefinementMethod::SaddlePoint => RefinementMethod::SaddlePoint,
            RsRefinementMethod::RadonPeak => RefinementMethod::RadonPeak,
        }
    }
}

/// Subpixel peak-fit mode (Radon). Mirrors
/// [`chess_corners::PeakFitMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeakFitMode {
    Parabolic = 0,
    Gaussian = 1,
}

impl From<PeakFitMode> for RsPeakFitMode {
    fn from(v: PeakFitMode) -> Self {
        match v {
            PeakFitMode::Parabolic => RsPeakFitMode::Parabolic,
            PeakFitMode::Gaussian => RsPeakFitMode::Gaussian,
        }
    }
}

impl From<RsPeakFitMode> for PeakFitMode {
    fn from(v: RsPeakFitMode) -> Self {
        match v {
            RsPeakFitMode::Parabolic => PeakFitMode::Parabolic,
            RsPeakFitMode::Gaussian => PeakFitMode::Gaussian,
            // The core enum is `#[non_exhaustive]`. New variants
            // would land here; map them to `Gaussian` (the default)
            // until the wrapper learns the new variant.
            _ => PeakFitMode::Gaussian,
        }
    }
}

/// Pre-pipeline upscale mode. Mirrors [`chess_corners::UpscaleMode`].
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpscaleMode {
    Disabled = 0,
    Fixed = 1,
}

impl From<UpscaleMode> for RsUpscaleMode {
    fn from(v: UpscaleMode) -> Self {
        match v {
            UpscaleMode::Disabled => RsUpscaleMode::Disabled,
            UpscaleMode::Fixed => RsUpscaleMode::Fixed,
        }
    }
}

impl From<RsUpscaleMode> for UpscaleMode {
    fn from(v: RsUpscaleMode) -> Self {
        match v {
            RsUpscaleMode::Disabled => UpscaleMode::Disabled,
            RsUpscaleMode::Fixed => UpscaleMode::Fixed,
        }
    }
}

// ---------------------------------------------------------------------------
// CenterOfMassConfig
// ---------------------------------------------------------------------------

/// Center-of-mass refiner configuration.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct CenterOfMassConfig {
    inner: RsCenterOfMassConfig,
}

#[wasm_bindgen]
impl CenterOfMassConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsCenterOfMassConfig::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.inner.radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.inner.radius = v;
    }
}

impl Default for CenterOfMassConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CenterOfMassConfig {
    pub(crate) fn inner(&self) -> RsCenterOfMassConfig {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsCenterOfMassConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// ForstnerConfig
// ---------------------------------------------------------------------------

/// Förstner gradient-based refiner configuration.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct ForstnerConfig {
    inner: RsForstnerConfig,
}

#[wasm_bindgen]
impl ForstnerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsForstnerConfig::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.inner.radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.inner.radius = v;
    }

    #[wasm_bindgen(getter, js_name = minTrace)]
    pub fn min_trace(&self) -> f32 {
        self.inner.min_trace
    }
    #[wasm_bindgen(setter, js_name = minTrace)]
    pub fn set_min_trace(&mut self, v: f32) {
        self.inner.min_trace = v;
    }

    #[wasm_bindgen(getter, js_name = minDet)]
    pub fn min_det(&self) -> f32 {
        self.inner.min_det
    }
    #[wasm_bindgen(setter, js_name = minDet)]
    pub fn set_min_det(&mut self, v: f32) {
        self.inner.min_det = v;
    }

    #[wasm_bindgen(getter, js_name = maxConditionNumber)]
    pub fn max_condition_number(&self) -> f32 {
        self.inner.max_condition_number
    }
    #[wasm_bindgen(setter, js_name = maxConditionNumber)]
    pub fn set_max_condition_number(&mut self, v: f32) {
        self.inner.max_condition_number = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.inner.max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.inner.max_offset = v;
    }
}

impl Default for ForstnerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ForstnerConfig {
    pub(crate) fn inner(&self) -> RsForstnerConfig {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsForstnerConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// SaddlePointConfig
// ---------------------------------------------------------------------------

/// Quadratic saddle-point refiner configuration.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct SaddlePointConfig {
    inner: RsSaddlePointConfig,
}

#[wasm_bindgen]
impl SaddlePointConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsSaddlePointConfig::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.inner.radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.inner.radius = v;
    }

    #[wasm_bindgen(getter, js_name = detMargin)]
    pub fn det_margin(&self) -> f32 {
        self.inner.det_margin
    }
    #[wasm_bindgen(setter, js_name = detMargin)]
    pub fn set_det_margin(&mut self, v: f32) {
        self.inner.det_margin = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.inner.max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.inner.max_offset = v;
    }

    #[wasm_bindgen(getter, js_name = minAbsDet)]
    pub fn min_abs_det(&self) -> f32 {
        self.inner.min_abs_det
    }
    #[wasm_bindgen(setter, js_name = minAbsDet)]
    pub fn set_min_abs_det(&mut self, v: f32) {
        self.inner.min_abs_det = v;
    }
}

impl Default for SaddlePointConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SaddlePointConfig {
    pub(crate) fn inner(&self) -> RsSaddlePointConfig {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsSaddlePointConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// RadonPeakConfig
// ---------------------------------------------------------------------------

/// Radon-peak refiner configuration.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct RadonPeakConfig {
    inner: RsRadonPeakConfig,
}

#[wasm_bindgen]
impl RadonPeakConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsRadonPeakConfig::default(),
        }
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        self.inner.ray_radius
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        self.inner.ray_radius = v;
    }

    #[wasm_bindgen(getter, js_name = patchRadius)]
    pub fn patch_radius(&self) -> u32 {
        self.inner.patch_radius
    }
    #[wasm_bindgen(setter, js_name = patchRadius)]
    pub fn set_patch_radius(&mut self, v: u32) {
        self.inner.patch_radius = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        self.inner.image_upsample
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        self.inner.image_upsample = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        self.inner.response_blur_radius
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        self.inner.response_blur_radius = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        self.inner.peak_fit.into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.inner.peak_fit = v.into();
    }

    #[wasm_bindgen(getter, js_name = minResponse)]
    pub fn min_response(&self) -> f32 {
        self.inner.min_response
    }
    #[wasm_bindgen(setter, js_name = minResponse)]
    pub fn set_min_response(&mut self, v: f32) {
        self.inner.min_response = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.inner.max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.inner.max_offset = v;
    }
}

impl Default for RadonPeakConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonPeakConfig {
    pub(crate) fn inner(&self) -> RsRadonPeakConfig {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsRadonPeakConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// RefinerConfig
// ---------------------------------------------------------------------------

/// Subpixel refiner selection plus per-variant parameters.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RefinerConfig {
    inner: RsRefinerConfig,
}

#[wasm_bindgen]
impl RefinerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsRefinerConfig::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> RefinementMethod {
        self.inner.kind.into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_kind(&mut self, v: RefinementMethod) {
        self.inner.kind = v.into();
    }

    #[wasm_bindgen(getter, js_name = centerOfMass)]
    pub fn center_of_mass(&self) -> CenterOfMassConfig {
        CenterOfMassConfig::from_inner(self.inner.center_of_mass)
    }
    #[wasm_bindgen(setter, js_name = centerOfMass)]
    pub fn set_center_of_mass(&mut self, v: &CenterOfMassConfig) {
        self.inner.center_of_mass = v.inner();
    }

    #[wasm_bindgen(getter)]
    pub fn forstner(&self) -> ForstnerConfig {
        ForstnerConfig::from_inner(self.inner.forstner)
    }
    #[wasm_bindgen(setter)]
    pub fn set_forstner(&mut self, v: &ForstnerConfig) {
        self.inner.forstner = v.inner();
    }

    #[wasm_bindgen(getter, js_name = saddlePoint)]
    pub fn saddle_point(&self) -> SaddlePointConfig {
        SaddlePointConfig::from_inner(self.inner.saddle_point)
    }
    #[wasm_bindgen(setter, js_name = saddlePoint)]
    pub fn set_saddle_point(&mut self, v: &SaddlePointConfig) {
        self.inner.saddle_point = v.inner();
    }

    #[wasm_bindgen(getter, js_name = radonPeak)]
    pub fn radon_peak(&self) -> RadonPeakConfig {
        RadonPeakConfig::from_inner(self.inner.radon_peak)
    }
    #[wasm_bindgen(setter, js_name = radonPeak)]
    pub fn set_radon_peak(&mut self, v: &RadonPeakConfig) {
        self.inner.radon_peak = v.inner();
    }
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RefinerConfig {
    pub(crate) fn inner(&self) -> RsRefinerConfig {
        self.inner.clone()
    }
    pub(crate) fn from_inner(inner: RsRefinerConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// RadonDetectorParams
// ---------------------------------------------------------------------------

/// Whole-image Radon detector parameters.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct RadonDetectorParams {
    inner: RsRadonDetectorParams,
}

#[wasm_bindgen]
impl RadonDetectorParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsRadonDetectorParams::default(),
        }
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        self.inner.ray_radius
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        self.inner.ray_radius = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        self.inner.image_upsample
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        self.inner.image_upsample = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        self.inner.response_blur_radius
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        self.inner.response_blur_radius = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        self.inner.peak_fit.into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.inner.peak_fit = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdRel)]
    pub fn threshold_rel(&self) -> f32 {
        self.inner.threshold_rel
    }
    #[wasm_bindgen(setter, js_name = thresholdRel)]
    pub fn set_threshold_rel(&mut self, v: f32) {
        self.inner.threshold_rel = v;
    }

    /// Absolute response floor; `None` to clear and let the relative
    /// threshold decide. Pass a finite number to override.
    #[wasm_bindgen(getter, js_name = thresholdAbs)]
    pub fn threshold_abs(&self) -> Option<f32> {
        self.inner.threshold_abs
    }
    #[wasm_bindgen(setter, js_name = thresholdAbs)]
    pub fn set_threshold_abs(&mut self, v: Option<f32>) {
        self.inner.threshold_abs = v;
    }

    #[wasm_bindgen(getter, js_name = nmsRadius)]
    pub fn nms_radius(&self) -> u32 {
        self.inner.nms_radius
    }
    #[wasm_bindgen(setter, js_name = nmsRadius)]
    pub fn set_nms_radius(&mut self, v: u32) {
        self.inner.nms_radius = v;
    }

    #[wasm_bindgen(getter, js_name = minClusterSize)]
    pub fn min_cluster_size(&self) -> u32 {
        self.inner.min_cluster_size
    }
    #[wasm_bindgen(setter, js_name = minClusterSize)]
    pub fn set_min_cluster_size(&mut self, v: u32) {
        self.inner.min_cluster_size = v;
    }
}

impl Default for RadonDetectorParams {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonDetectorParams {
    pub(crate) fn inner(&self) -> RsRadonDetectorParams {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsRadonDetectorParams) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// UpscaleConfig
// ---------------------------------------------------------------------------

/// Pre-pipeline integer upscaling configuration.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct UpscaleConfig {
    inner: RsUpscaleConfig,
}

#[wasm_bindgen]
impl UpscaleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsUpscaleConfig::default(),
        }
    }

    /// Factory for a disabled upscale (default).
    pub fn disabled() -> Self {
        Self {
            inner: RsUpscaleConfig::disabled(),
        }
    }

    /// Factory for a fixed integer-factor upscale (2, 3, or 4).
    pub fn fixed(factor: u32) -> Self {
        Self {
            inner: RsUpscaleConfig::fixed(factor),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> UpscaleMode {
        self.inner.mode.into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_mode(&mut self, v: UpscaleMode) {
        self.inner.mode = v.into();
    }

    #[wasm_bindgen(getter)]
    pub fn factor(&self) -> u32 {
        self.inner.factor
    }
    #[wasm_bindgen(setter)]
    pub fn set_factor(&mut self, v: u32) {
        self.inner.factor = v;
    }
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl UpscaleConfig {
    pub(crate) fn inner(&self) -> RsUpscaleConfig {
        self.inner
    }
    pub(crate) fn from_inner(inner: RsUpscaleConfig) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// ChessConfig
// ---------------------------------------------------------------------------

/// High-level detector configuration. Mirrors
/// [`chess_corners::ChessConfig`].
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessConfig {
    inner: RsChessConfig,
}

#[wasm_bindgen]
impl ChessConfig {
    /// Construct a `ChessConfig` with library defaults
    /// (single-scale, absolute threshold = 0.0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsChessConfig::default(),
        }
    }

    /// Single-scale preset (alias for [`Self::new`]).
    #[wasm_bindgen(js_name = singleScale)]
    pub fn single_scale() -> Self {
        Self {
            inner: RsChessConfig::single_scale(),
        }
    }

    /// Recommended 3-level multiscale preset.
    pub fn multiscale() -> Self {
        Self {
            inner: RsChessConfig::multiscale(),
        }
    }

    /// Whole-image Radon detector preset.
    pub fn radon() -> Self {
        Self {
            inner: RsChessConfig::radon(),
        }
    }

    // ---- Top-level fields ----

    #[wasm_bindgen(getter, js_name = detectorMode)]
    pub fn detector_mode(&self) -> DetectorMode {
        self.inner.detector_mode.into()
    }
    #[wasm_bindgen(setter, js_name = detectorMode)]
    pub fn set_detector_mode(&mut self, v: DetectorMode) {
        self.inner.detector_mode = v.into();
    }

    #[wasm_bindgen(getter, js_name = descriptorMode)]
    pub fn descriptor_mode(&self) -> DescriptorMode {
        self.inner.descriptor_mode.into()
    }
    #[wasm_bindgen(setter, js_name = descriptorMode)]
    pub fn set_descriptor_mode(&mut self, v: DescriptorMode) {
        self.inner.descriptor_mode = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdMode)]
    pub fn threshold_mode(&self) -> ThresholdMode {
        self.inner.threshold_mode.into()
    }
    #[wasm_bindgen(setter, js_name = thresholdMode)]
    pub fn set_threshold_mode(&mut self, v: ThresholdMode) {
        self.inner.threshold_mode = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdValue)]
    pub fn threshold_value(&self) -> f32 {
        self.inner.threshold_value
    }
    #[wasm_bindgen(setter, js_name = thresholdValue)]
    pub fn set_threshold_value(&mut self, v: f32) {
        self.inner.threshold_value = v;
    }

    #[wasm_bindgen(getter, js_name = nmsRadius)]
    pub fn nms_radius(&self) -> u32 {
        self.inner.nms_radius
    }
    #[wasm_bindgen(setter, js_name = nmsRadius)]
    pub fn set_nms_radius(&mut self, v: u32) {
        self.inner.nms_radius = v;
    }

    #[wasm_bindgen(getter, js_name = minClusterSize)]
    pub fn min_cluster_size(&self) -> u32 {
        self.inner.min_cluster_size
    }
    #[wasm_bindgen(setter, js_name = minClusterSize)]
    pub fn set_min_cluster_size(&mut self, v: u32) {
        self.inner.min_cluster_size = v;
    }

    #[wasm_bindgen(getter)]
    pub fn refiner(&self) -> RefinerConfig {
        RefinerConfig::from_inner(self.inner.refiner.clone())
    }
    #[wasm_bindgen(setter)]
    pub fn set_refiner(&mut self, v: &RefinerConfig) {
        self.inner.refiner = v.inner();
    }

    #[wasm_bindgen(getter, js_name = pyramidLevels)]
    pub fn pyramid_levels(&self) -> u8 {
        self.inner.pyramid_levels
    }
    #[wasm_bindgen(setter, js_name = pyramidLevels)]
    pub fn set_pyramid_levels(&mut self, v: u8) {
        self.inner.pyramid_levels = v;
    }

    #[wasm_bindgen(getter, js_name = pyramidMinSize)]
    pub fn pyramid_min_size(&self) -> u32 {
        self.inner.pyramid_min_size as u32
    }
    #[wasm_bindgen(setter, js_name = pyramidMinSize)]
    pub fn set_pyramid_min_size(&mut self, v: u32) {
        self.inner.pyramid_min_size = v as usize;
    }

    #[wasm_bindgen(getter, js_name = refinementRadius)]
    pub fn refinement_radius(&self) -> u32 {
        self.inner.refinement_radius
    }
    #[wasm_bindgen(setter, js_name = refinementRadius)]
    pub fn set_refinement_radius(&mut self, v: u32) {
        self.inner.refinement_radius = v;
    }

    #[wasm_bindgen(getter, js_name = mergeRadius)]
    pub fn merge_radius(&self) -> f32 {
        self.inner.merge_radius
    }
    #[wasm_bindgen(setter, js_name = mergeRadius)]
    pub fn set_merge_radius(&mut self, v: f32) {
        self.inner.merge_radius = v;
    }

    #[wasm_bindgen(getter)]
    pub fn upscale(&self) -> UpscaleConfig {
        UpscaleConfig::from_inner(self.inner.upscale)
    }
    #[wasm_bindgen(setter)]
    pub fn set_upscale(&mut self, v: &UpscaleConfig) {
        self.inner.upscale = v.inner();
    }

    #[wasm_bindgen(getter, js_name = radonDetector)]
    pub fn radon_detector(&self) -> RadonDetectorParams {
        RadonDetectorParams::from_inner(self.inner.radon_detector)
    }
    #[wasm_bindgen(setter, js_name = radonDetector)]
    pub fn set_radon_detector(&mut self, v: &RadonDetectorParams) {
        self.inner.radon_detector = v.inner();
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ChessConfig {
    /// Borrow the underlying Rust config (used by the detector).
    pub(crate) fn inner(&self) -> &RsChessConfig {
        &self.inner
    }
    /// Mutable borrow used by the legacy setter shortcuts on
    /// `ChessDetector` (e.g. `set_threshold` mirroring into both
    /// ChESS and Radon paths).
    pub(crate) fn inner_mut(&mut self) -> &mut RsChessConfig {
        &mut self.inner
    }
}
