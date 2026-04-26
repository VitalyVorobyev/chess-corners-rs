//! Typed `#[wasm_bindgen]` wrappers around `chess-corners` config
//! structs.
//!
//! ## Live nested edits
//!
//! Each wrapper stores its inner Rust value in a shared
//! `Rc<RefCell<T>>` cell, and compound wrappers (`RefinerConfig`,
//! `ChessConfig`) hold `Rc` handles to their children's cells. A
//! getter returns a wrapper backed by the same cell as the parent,
//! so chained mutation propagates without a round-trip:
//!
//! ```js
//! const cfg = ChessConfig.multiscale();
//! cfg.refiner.kind = RefinementMethod.RadonPeak;        // works
//! cfg.refiner.forstner.maxOffset = 2.0;                  // works
//! cfg.radonDetector.rayRadius = 5;                       // works
//! ```
//!
//! Setters that take a nested wrapper (e.g. `cfg.refiner = newCfg`)
//! reseat the parent's `Rc` to point at the new value's cell, so
//! future getter calls return wrappers backed by the new cell. Any
//! JS reference held to the *previous* nested wrapper still
//! observes the previous cell — matching natural attribute-
//! reassignment semantics in JS.
//!
//! Single-threaded `Rc<RefCell<T>>` is sound on
//! `wasm32-unknown-unknown`; wasm-bindgen modules are not shared
//! across worker threads.
//!
//! Why a wrapper layer at all? The Rust source-of-truth structs live
//! in `chess-corners` / `chess-corners-core` and must not depend on
//! `wasm-bindgen` (per the workspace dependency rule in `AGENTS.md`).
//! These wrappers add the JS-facing attribute layer in the WASM
//! crate only.

use std::cell::RefCell;
use std::rc::Rc;

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

/// Shared mutable cell used by every wrapper. Single-threaded;
/// `wasm-bindgen` instances live entirely on one JS thread.
type Cell<T> = Rc<RefCell<T>>;

fn cell<T>(value: T) -> Cell<T> {
    Rc::new(RefCell::new(value))
}

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

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct CenterOfMassConfig {
    cell: Cell<RsCenterOfMassConfig>,
}

#[wasm_bindgen]
impl CenterOfMassConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsCenterOfMassConfig::default()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.cell.borrow().radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.cell.borrow_mut().radius = v;
    }
}

impl Default for CenterOfMassConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CenterOfMassConfig {
    pub(crate) fn share_cell(&self) -> Cell<RsCenterOfMassConfig> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsCenterOfMassConfig>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// ForstnerConfig
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ForstnerConfig {
    cell: Cell<RsForstnerConfig>,
}

#[wasm_bindgen]
impl ForstnerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsForstnerConfig::default()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.cell.borrow().radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.cell.borrow_mut().radius = v;
    }

    #[wasm_bindgen(getter, js_name = minTrace)]
    pub fn min_trace(&self) -> f32 {
        self.cell.borrow().min_trace
    }
    #[wasm_bindgen(setter, js_name = minTrace)]
    pub fn set_min_trace(&mut self, v: f32) {
        self.cell.borrow_mut().min_trace = v;
    }

    #[wasm_bindgen(getter, js_name = minDet)]
    pub fn min_det(&self) -> f32 {
        self.cell.borrow().min_det
    }
    #[wasm_bindgen(setter, js_name = minDet)]
    pub fn set_min_det(&mut self, v: f32) {
        self.cell.borrow_mut().min_det = v;
    }

    #[wasm_bindgen(getter, js_name = maxConditionNumber)]
    pub fn max_condition_number(&self) -> f32 {
        self.cell.borrow().max_condition_number
    }
    #[wasm_bindgen(setter, js_name = maxConditionNumber)]
    pub fn set_max_condition_number(&mut self, v: f32) {
        self.cell.borrow_mut().max_condition_number = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.cell.borrow().max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.cell.borrow_mut().max_offset = v;
    }
}

impl Default for ForstnerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ForstnerConfig {
    pub(crate) fn share_cell(&self) -> Cell<RsForstnerConfig> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsForstnerConfig>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// SaddlePointConfig
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct SaddlePointConfig {
    cell: Cell<RsSaddlePointConfig>,
}

#[wasm_bindgen]
impl SaddlePointConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsSaddlePointConfig::default()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> i32 {
        self.cell.borrow().radius
    }
    #[wasm_bindgen(setter)]
    pub fn set_radius(&mut self, v: i32) {
        self.cell.borrow_mut().radius = v;
    }

    #[wasm_bindgen(getter, js_name = detMargin)]
    pub fn det_margin(&self) -> f32 {
        self.cell.borrow().det_margin
    }
    #[wasm_bindgen(setter, js_name = detMargin)]
    pub fn set_det_margin(&mut self, v: f32) {
        self.cell.borrow_mut().det_margin = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.cell.borrow().max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.cell.borrow_mut().max_offset = v;
    }

    #[wasm_bindgen(getter, js_name = minAbsDet)]
    pub fn min_abs_det(&self) -> f32 {
        self.cell.borrow().min_abs_det
    }
    #[wasm_bindgen(setter, js_name = minAbsDet)]
    pub fn set_min_abs_det(&mut self, v: f32) {
        self.cell.borrow_mut().min_abs_det = v;
    }
}

impl Default for SaddlePointConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SaddlePointConfig {
    pub(crate) fn share_cell(&self) -> Cell<RsSaddlePointConfig> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsSaddlePointConfig>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// RadonPeakConfig
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonPeakConfig {
    cell: Cell<RsRadonPeakConfig>,
}

#[wasm_bindgen]
impl RadonPeakConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsRadonPeakConfig::default()),
        }
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        self.cell.borrow().ray_radius
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        self.cell.borrow_mut().ray_radius = v;
    }

    #[wasm_bindgen(getter, js_name = patchRadius)]
    pub fn patch_radius(&self) -> u32 {
        self.cell.borrow().patch_radius
    }
    #[wasm_bindgen(setter, js_name = patchRadius)]
    pub fn set_patch_radius(&mut self, v: u32) {
        self.cell.borrow_mut().patch_radius = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        self.cell.borrow().image_upsample
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        self.cell.borrow_mut().image_upsample = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        self.cell.borrow().response_blur_radius
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        self.cell.borrow_mut().response_blur_radius = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        self.cell.borrow().peak_fit.into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.cell.borrow_mut().peak_fit = v.into();
    }

    #[wasm_bindgen(getter, js_name = minResponse)]
    pub fn min_response(&self) -> f32 {
        self.cell.borrow().min_response
    }
    #[wasm_bindgen(setter, js_name = minResponse)]
    pub fn set_min_response(&mut self, v: f32) {
        self.cell.borrow_mut().min_response = v;
    }

    #[wasm_bindgen(getter, js_name = maxOffset)]
    pub fn max_offset(&self) -> f32 {
        self.cell.borrow().max_offset
    }
    #[wasm_bindgen(setter, js_name = maxOffset)]
    pub fn set_max_offset(&mut self, v: f32) {
        self.cell.borrow_mut().max_offset = v;
    }
}

impl Default for RadonPeakConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonPeakConfig {
    pub(crate) fn share_cell(&self) -> Cell<RsRadonPeakConfig> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsRadonPeakConfig>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// RefinerConfig
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RefinerConfig {
    /// Scalar `kind` field. Stored in its own cell so reseating
    /// `cfg.refiner = newOne` brings the new kind cell along with
    /// the per-variant cells.
    kind: Cell<RsRefinementMethod>,
    /// Per-variant cells. Each cell is shared with the corresponding
    /// per-variant wrapper returned from getters — mutating either
    /// side updates both.
    center_of_mass: Cell<RsCenterOfMassConfig>,
    forstner: Cell<RsForstnerConfig>,
    saddle_point: Cell<RsSaddlePointConfig>,
    radon_peak: Cell<RsRadonPeakConfig>,
}

#[wasm_bindgen]
impl RefinerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            kind: cell(RsRefinementMethod::default()),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
            forstner: cell(RsForstnerConfig::default()),
            saddle_point: cell(RsSaddlePointConfig::default()),
            radon_peak: cell(RsRadonPeakConfig::default()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> RefinementMethod {
        (*self.kind.borrow()).into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_kind(&mut self, v: RefinementMethod) {
        *self.kind.borrow_mut() = v.into();
    }

    #[wasm_bindgen(getter, js_name = centerOfMass)]
    pub fn center_of_mass(&self) -> CenterOfMassConfig {
        CenterOfMassConfig::from_cell(Rc::clone(&self.center_of_mass))
    }
    #[wasm_bindgen(setter, js_name = centerOfMass)]
    pub fn set_center_of_mass(&mut self, v: &CenterOfMassConfig) {
        self.center_of_mass = v.share_cell();
    }

    #[wasm_bindgen(getter)]
    pub fn forstner(&self) -> ForstnerConfig {
        ForstnerConfig::from_cell(Rc::clone(&self.forstner))
    }
    #[wasm_bindgen(setter)]
    pub fn set_forstner(&mut self, v: &ForstnerConfig) {
        self.forstner = v.share_cell();
    }

    #[wasm_bindgen(getter, js_name = saddlePoint)]
    pub fn saddle_point(&self) -> SaddlePointConfig {
        SaddlePointConfig::from_cell(Rc::clone(&self.saddle_point))
    }
    #[wasm_bindgen(setter, js_name = saddlePoint)]
    pub fn set_saddle_point(&mut self, v: &SaddlePointConfig) {
        self.saddle_point = v.share_cell();
    }

    #[wasm_bindgen(getter, js_name = radonPeak)]
    pub fn radon_peak(&self) -> RadonPeakConfig {
        RadonPeakConfig::from_cell(Rc::clone(&self.radon_peak))
    }
    #[wasm_bindgen(setter, js_name = radonPeak)]
    pub fn set_radon_peak(&mut self, v: &RadonPeakConfig) {
        self.radon_peak = v.share_cell();
    }
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RefinerConfig {
    pub(crate) fn snapshot(&self) -> RsRefinerConfig {
        RsRefinerConfig {
            kind: *self.kind.borrow(),
            center_of_mass: *self.center_of_mass.borrow(),
            forstner: *self.forstner.borrow(),
            saddle_point: *self.saddle_point.borrow(),
            radon_peak: *self.radon_peak.borrow(),
        }
    }

    pub(crate) fn from_value(value: RsRefinerConfig) -> Self {
        Self {
            kind: cell(value.kind),
            center_of_mass: cell(value.center_of_mass),
            forstner: cell(value.forstner),
            saddle_point: cell(value.saddle_point),
            radon_peak: cell(value.radon_peak),
        }
    }
}

// ---------------------------------------------------------------------------
// RadonDetectorParams
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonDetectorParams {
    cell: Cell<RsRadonDetectorParams>,
}

#[wasm_bindgen]
impl RadonDetectorParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsRadonDetectorParams::default()),
        }
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        self.cell.borrow().ray_radius
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        self.cell.borrow_mut().ray_radius = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        self.cell.borrow().image_upsample
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        self.cell.borrow_mut().image_upsample = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        self.cell.borrow().response_blur_radius
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        self.cell.borrow_mut().response_blur_radius = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        self.cell.borrow().peak_fit.into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.cell.borrow_mut().peak_fit = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdRel)]
    pub fn threshold_rel(&self) -> f32 {
        self.cell.borrow().threshold_rel
    }
    #[wasm_bindgen(setter, js_name = thresholdRel)]
    pub fn set_threshold_rel(&mut self, v: f32) {
        self.cell.borrow_mut().threshold_rel = v;
    }

    /// Absolute response floor; `None` to clear and let the relative
    /// threshold decide. Pass a finite number to override.
    #[wasm_bindgen(getter, js_name = thresholdAbs)]
    pub fn threshold_abs(&self) -> Option<f32> {
        self.cell.borrow().threshold_abs
    }
    #[wasm_bindgen(setter, js_name = thresholdAbs)]
    pub fn set_threshold_abs(&mut self, v: Option<f32>) {
        self.cell.borrow_mut().threshold_abs = v;
    }

    #[wasm_bindgen(getter, js_name = nmsRadius)]
    pub fn nms_radius(&self) -> u32 {
        self.cell.borrow().nms_radius
    }
    #[wasm_bindgen(setter, js_name = nmsRadius)]
    pub fn set_nms_radius(&mut self, v: u32) {
        self.cell.borrow_mut().nms_radius = v;
    }

    #[wasm_bindgen(getter, js_name = minClusterSize)]
    pub fn min_cluster_size(&self) -> u32 {
        self.cell.borrow().min_cluster_size
    }
    #[wasm_bindgen(setter, js_name = minClusterSize)]
    pub fn set_min_cluster_size(&mut self, v: u32) {
        self.cell.borrow_mut().min_cluster_size = v;
    }
}

impl Default for RadonDetectorParams {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonDetectorParams {
    pub(crate) fn share_cell(&self) -> Cell<RsRadonDetectorParams> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsRadonDetectorParams>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// UpscaleConfig
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct UpscaleConfig {
    cell: Cell<RsUpscaleConfig>,
}

#[wasm_bindgen]
impl UpscaleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsUpscaleConfig::default()),
        }
    }

    /// Factory for a disabled upscale (default).
    pub fn disabled() -> Self {
        Self {
            cell: cell(RsUpscaleConfig::disabled()),
        }
    }

    /// Factory for a fixed integer-factor upscale (2, 3, or 4).
    pub fn fixed(factor: u32) -> Self {
        Self {
            cell: cell(RsUpscaleConfig::fixed(factor)),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> UpscaleMode {
        self.cell.borrow().mode.into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_mode(&mut self, v: UpscaleMode) {
        self.cell.borrow_mut().mode = v.into();
    }

    #[wasm_bindgen(getter)]
    pub fn factor(&self) -> u32 {
        self.cell.borrow().factor
    }
    #[wasm_bindgen(setter)]
    pub fn set_factor(&mut self, v: u32) {
        self.cell.borrow_mut().factor = v;
    }
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl UpscaleConfig {
    pub(crate) fn share_cell(&self) -> Cell<RsUpscaleConfig> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsUpscaleConfig>) -> Self {
        Self { cell }
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
    // Scalar fields are stored in their own single-purpose cells.
    // The wrapper-level `Clone` returns a shallow copy that shares
    // these cells with the original — `cfg.refiner.kind = X` after
    // `let cfg2 = cfg.clone()` (Rust-side) would also affect cfg2,
    // intentionally, because the cells are shared.
    detector_mode: Cell<RsDetectorMode>,
    descriptor_mode: Cell<RsDescriptorMode>,
    threshold_mode: Cell<RsThresholdMode>,
    threshold_value: Cell<f32>,
    nms_radius: Cell<u32>,
    min_cluster_size: Cell<u32>,
    pyramid_levels: Cell<u8>,
    pyramid_min_size: Cell<usize>,
    refinement_radius: Cell<u32>,
    merge_radius: Cell<f32>,
    refiner: RefinerConfig,
    upscale: Cell<RsUpscaleConfig>,
    radon_detector: Cell<RsRadonDetectorParams>,
}

impl ChessConfig {
    pub(crate) fn from_value_pub(value: RsChessConfig) -> Self {
        Self::from_value(value)
    }

    fn from_value(value: RsChessConfig) -> Self {
        Self {
            detector_mode: cell(value.detector_mode),
            descriptor_mode: cell(value.descriptor_mode),
            threshold_mode: cell(value.threshold_mode),
            threshold_value: cell(value.threshold_value),
            nms_radius: cell(value.nms_radius),
            min_cluster_size: cell(value.min_cluster_size),
            pyramid_levels: cell(value.pyramid_levels),
            pyramid_min_size: cell(value.pyramid_min_size),
            refinement_radius: cell(value.refinement_radius),
            merge_radius: cell(value.merge_radius),
            refiner: RefinerConfig::from_value(value.refiner),
            upscale: cell(value.upscale),
            radon_detector: cell(value.radon_detector),
        }
    }

    /// Snapshot the current state into the Rust facade
    /// `ChessConfig` for hand-off to the detector.
    pub(crate) fn snapshot(&self) -> RsChessConfig {
        let mut cfg = RsChessConfig::default();
        cfg.detector_mode = *self.detector_mode.borrow();
        cfg.descriptor_mode = *self.descriptor_mode.borrow();
        cfg.threshold_mode = *self.threshold_mode.borrow();
        cfg.threshold_value = *self.threshold_value.borrow();
        cfg.nms_radius = *self.nms_radius.borrow();
        cfg.min_cluster_size = *self.min_cluster_size.borrow();
        cfg.pyramid_levels = *self.pyramid_levels.borrow();
        cfg.pyramid_min_size = *self.pyramid_min_size.borrow();
        cfg.refinement_radius = *self.refinement_radius.borrow();
        cfg.merge_radius = *self.merge_radius.borrow();
        cfg.refiner = self.refiner.snapshot();
        cfg.upscale = *self.upscale.borrow();
        cfg.radon_detector = *self.radon_detector.borrow();
        cfg
    }
}

#[wasm_bindgen]
impl ChessConfig {
    /// Construct a `ChessConfig` with library defaults
    /// (single-scale, absolute threshold = 0.0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsChessConfig::default())
    }

    /// Single-scale preset (alias for [`Self::new`]).
    #[wasm_bindgen(js_name = singleScale)]
    pub fn single_scale() -> Self {
        Self::from_value(RsChessConfig::single_scale())
    }

    /// Recommended 3-level multiscale preset.
    pub fn multiscale() -> Self {
        Self::from_value(RsChessConfig::multiscale())
    }

    /// Whole-image Radon detector preset.
    pub fn radon() -> Self {
        Self::from_value(RsChessConfig::radon())
    }

    // ---- Top-level scalar fields ----
    // Note: `Default for ChessConfig` is provided below outside the
    // `#[wasm_bindgen]` impl so that wasm-bindgen doesn't try to
    // expose it as a JS method.

    #[wasm_bindgen(getter, js_name = detectorMode)]
    pub fn detector_mode(&self) -> DetectorMode {
        (*self.detector_mode.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = detectorMode)]
    pub fn set_detector_mode(&mut self, v: DetectorMode) {
        *self.detector_mode.borrow_mut() = v.into();
    }

    #[wasm_bindgen(getter, js_name = descriptorMode)]
    pub fn descriptor_mode(&self) -> DescriptorMode {
        (*self.descriptor_mode.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = descriptorMode)]
    pub fn set_descriptor_mode(&mut self, v: DescriptorMode) {
        *self.descriptor_mode.borrow_mut() = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdMode)]
    pub fn threshold_mode(&self) -> ThresholdMode {
        (*self.threshold_mode.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = thresholdMode)]
    pub fn set_threshold_mode(&mut self, v: ThresholdMode) {
        *self.threshold_mode.borrow_mut() = v.into();
    }

    #[wasm_bindgen(getter, js_name = thresholdValue)]
    pub fn threshold_value(&self) -> f32 {
        *self.threshold_value.borrow()
    }
    #[wasm_bindgen(setter, js_name = thresholdValue)]
    pub fn set_threshold_value(&mut self, v: f32) {
        *self.threshold_value.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = nmsRadius)]
    pub fn nms_radius(&self) -> u32 {
        *self.nms_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = nmsRadius)]
    pub fn set_nms_radius(&mut self, v: u32) {
        *self.nms_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = minClusterSize)]
    pub fn min_cluster_size(&self) -> u32 {
        *self.min_cluster_size.borrow()
    }
    #[wasm_bindgen(setter, js_name = minClusterSize)]
    pub fn set_min_cluster_size(&mut self, v: u32) {
        *self.min_cluster_size.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = pyramidLevels)]
    pub fn pyramid_levels(&self) -> u8 {
        *self.pyramid_levels.borrow()
    }
    #[wasm_bindgen(setter, js_name = pyramidLevels)]
    pub fn set_pyramid_levels(&mut self, v: u8) {
        *self.pyramid_levels.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = pyramidMinSize)]
    pub fn pyramid_min_size(&self) -> u32 {
        *self.pyramid_min_size.borrow() as u32
    }
    #[wasm_bindgen(setter, js_name = pyramidMinSize)]
    pub fn set_pyramid_min_size(&mut self, v: u32) {
        *self.pyramid_min_size.borrow_mut() = v as usize;
    }

    #[wasm_bindgen(getter, js_name = refinementRadius)]
    pub fn refinement_radius(&self) -> u32 {
        *self.refinement_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = refinementRadius)]
    pub fn set_refinement_radius(&mut self, v: u32) {
        *self.refinement_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = mergeRadius)]
    pub fn merge_radius(&self) -> f32 {
        *self.merge_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = mergeRadius)]
    pub fn set_merge_radius(&mut self, v: f32) {
        *self.merge_radius.borrow_mut() = v;
    }

    // ---- Nested wrappers (live views via shared cells) ----

    #[wasm_bindgen(getter)]
    pub fn refiner(&self) -> RefinerConfig {
        self.refiner.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_refiner(&mut self, v: &RefinerConfig) {
        // Reseat all five cells so future `cfg.refiner.*` calls
        // observe `v`'s state. JS code that already held the
        // previous `cfg.refiner` keeps observing the previous cells
        // — matches natural JS attribute-replacement semantics.
        self.refiner = v.clone();
    }

    #[wasm_bindgen(getter)]
    pub fn upscale(&self) -> UpscaleConfig {
        UpscaleConfig::from_cell(Rc::clone(&self.upscale))
    }
    #[wasm_bindgen(setter)]
    pub fn set_upscale(&mut self, v: &UpscaleConfig) {
        self.upscale = v.share_cell();
    }

    #[wasm_bindgen(getter, js_name = radonDetector)]
    pub fn radon_detector(&self) -> RadonDetectorParams {
        RadonDetectorParams::from_cell(Rc::clone(&self.radon_detector))
    }
    #[wasm_bindgen(setter, js_name = radonDetector)]
    pub fn set_radon_detector(&mut self, v: &RadonDetectorParams) {
        self.radon_detector = v.share_cell();
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    //! Native Rust tests of the shared-cell semantics. These run on
    //! `cargo test -p chess-corners-wasm` (host target) and don't
    //! need a JS runner. wasm-bindgen-test would be needed to
    //! exercise the JS-facing getter/setter mangling, but the cell
    //! plumbing is what matters for the live-edit guarantee and is
    //! identical on host and on wasm32.

    use super::*;
    use chess_corners::{DetectorMode as RsDetectorMode, RefinementMethod as RsRefinementMethod};

    #[test]
    fn nested_edits_propagate_through_chess_config() {
        let cfg = ChessConfig::new();
        // `cfg.refiner` returns a wrapper sharing cells with `cfg`.
        let mut r = cfg.refiner();
        r.set_kind(RefinementMethod::Forstner);

        // The mutation must be visible on a fresh getter call.
        assert_eq!(
            cfg.snapshot().refiner.kind,
            RsRefinementMethod::Forstner,
            "cfg.refiner.kind = X must propagate without round-trip"
        );

        // Per-variant edits also propagate.
        let mut f = cfg.refiner().forstner();
        f.set_max_offset(2.5);
        assert_eq!(cfg.snapshot().refiner.forstner.max_offset, 2.5);
    }

    #[test]
    fn radon_detector_edits_propagate() {
        let cfg = ChessConfig::new();
        let mut radon = cfg.radon_detector();
        radon.set_ray_radius(7);
        radon.set_image_upsample(2);
        let snap = cfg.snapshot();
        assert_eq!(snap.radon_detector.ray_radius, 7);
        assert_eq!(snap.radon_detector.image_upsample, 2);
    }

    #[test]
    fn upscale_edits_propagate() {
        let cfg = ChessConfig::new();
        let mut up = cfg.upscale();
        up.set_factor(3);
        up.set_mode(UpscaleMode::Fixed);
        let snap = cfg.snapshot();
        assert_eq!(snap.upscale.factor, 3);
        assert_eq!(snap.upscale.mode, RsUpscaleMode::Fixed);
    }

    #[test]
    fn assigning_nested_wrapper_reseats_cells() {
        let mut cfg = ChessConfig::new();
        // Build a freestanding refiner with a custom kind.
        let mut new_refiner = RefinerConfig::new();
        new_refiner.set_kind(RefinementMethod::SaddlePoint);
        new_refiner.forstner().set_max_offset(3.5);

        cfg.set_refiner(&new_refiner);

        // Reseating must take effect for future getter calls.
        let snap = cfg.snapshot();
        assert_eq!(snap.refiner.kind, RsRefinementMethod::SaddlePoint);
        assert_eq!(snap.refiner.forstner.max_offset, 3.5);

        // And `cfg.refiner.*` is now backed by `new_refiner`'s cells:
        // mutating `new_refiner` afterwards must propagate.
        new_refiner.forstner().set_max_offset(4.5);
        assert_eq!(cfg.snapshot().refiner.forstner.max_offset, 4.5);
    }

    #[test]
    fn top_level_scalar_setters_match_facade_defaults() {
        let mut cfg = ChessConfig::new();
        cfg.set_detector_mode(DetectorMode::Radon);
        cfg.set_threshold_value(0.42);
        let snap = cfg.snapshot();
        assert_eq!(snap.detector_mode, RsDetectorMode::Radon);
        assert!((snap.threshold_value - 0.42).abs() < 1e-6);
    }

    #[test]
    fn snapshot_returns_independent_state() {
        // After capturing a snapshot, further edits to the wrapper
        // must NOT leak into the snapshot — the snapshot's
        // `RsChessConfig` value is plain Rust (no Rc).
        let mut cfg = ChessConfig::new();
        cfg.set_threshold_value(0.1);
        let snap = cfg.snapshot();
        cfg.set_threshold_value(0.9);
        assert!((snap.threshold_value - 0.1).abs() < 1e-6);
    }
}
