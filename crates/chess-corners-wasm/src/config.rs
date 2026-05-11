//! Typed `#[wasm_bindgen]` wrappers around `chess-corners` config
//! structs.
//!
//! ## Live nested edits
//!
//! Each wrapper stores its inner Rust value in a shared
//! `Rc<RefCell<T>>` cell, and compound wrappers (`RefinerConfig`,
//! `ChessConfig`, `DetectionStrategy`, `ChessStrategy`, `RadonStrategy`,
//! `MultiscaleParams`) hold `Rc` handles to their children's cells. A
//! getter returns a wrapper backed by the same cell as the parent,
//! so chained mutation propagates without a round-trip:
//!
//! ```js
//! const cfg = ChessConfig.multiscale();
//! cfg.refiner.kind = RefinementMethod.RadonPeak;        // works
//! cfg.refiner.forstner.maxOffset = 2.0;                  // works
//! cfg.strategy.chess.nmsRadius = 3;                      // works
//! cfg.strategy.chess.multiscale.pyramidLevels = 4;       // works
//! ```
//!
//! Setters that take a nested wrapper (e.g. `cfg.refiner = newCfg`)
//! reseat the parent's `Rc` to point at the new value's cell, so
//! future getter calls return wrappers backed by the new cell. Any
//! JS reference held to the *previous* nested wrapper still
//! observes the previous cell — matching natural attribute-
//! reassignment semantics in JS.
//!
//! ## Strategy discriminants
//!
//! [`DetectionStrategy`] is a tagged wrapper carrying both a
//! [`ChessStrategy`] and a [`RadonStrategy`] cell plus an internal
//! `kind` discriminant. Only the cell matching the active `kind` is
//! snapshotted into the Rust `RsDetectionStrategy`. Field edits to
//! the *inactive* variant are kept in their cell and become visible
//! again as soon as the discriminant flips back — this mirrors what
//! a JS developer expects when they pre-populate both branches
//! before choosing one.
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
    ChessRing as RsChessRing, ChessStrategy as RsChessStrategy, DescriptorMode as RsDescriptorMode,
    DetectionStrategy as RsDetectionStrategy, ForstnerConfig as RsForstnerConfig,
    MultiscaleParams as RsMultiscaleParams, OrientationMethod as RsOrientationMethod,
    PeakFitMode as RsPeakFitMode, RadonPeakConfig as RsRadonPeakConfig,
    RadonStrategy as RsRadonStrategy, RefinementMethod as RsRefinementMethod,
    RefinerConfig as RsRefinerConfig, SaddlePointConfig as RsSaddlePointConfig,
    Threshold as RsThreshold, UpscaleConfig as RsUpscaleConfig, UpscaleMode as RsUpscaleMode,
};
use wasm_bindgen::prelude::*;

/// Shared mutable cell used by every wrapper. Single-threaded;
/// `wasm-bindgen` instances live entirely on one JS thread.
type Cell<T> = Rc<RefCell<T>>;

fn cell<T>(value: T) -> Cell<T> {
    Rc::new(RefCell::new(value))
}

// ---------------------------------------------------------------------------
// Plain enums
// ---------------------------------------------------------------------------

/// ChESS sampling ring radius. Mirrors [`chess_corners::ChessRing`].
///
/// `Canonical` (= 0): paper-default radius-5 ring (16 samples).
/// `Broad` (= 1): radius-10 ring; useful for low-resolution / heavily
/// blurred imagery where the canonical ring under-samples.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChessRing {
    Canonical = 0,
    Broad = 1,
}

impl From<ChessRing> for RsChessRing {
    fn from(v: ChessRing) -> Self {
        match v {
            ChessRing::Canonical => RsChessRing::Canonical,
            ChessRing::Broad => RsChessRing::Broad,
        }
    }
}

impl From<RsChessRing> for ChessRing {
    fn from(v: RsChessRing) -> Self {
        match v {
            RsChessRing::Canonical => ChessRing::Canonical,
            RsChessRing::Broad => ChessRing::Broad,
            _ => ChessRing::Canonical,
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
            _ => DescriptorMode::FollowDetector,
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
            _ => RefinementMethod::CenterOfMass,
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
            _ => UpscaleMode::Disabled,
        }
    }
}

/// Two-axis orientation-fit method. Mirrors [`chess_corners::OrientationMethod`].
///
/// `RingFit` *(default, = 0)* — fits the parametric two-axis chessboard
/// intensity model to the 16-sample ring via Gauss-Newton, with calibrated
/// per-axis 1σ uncertainties. Suitable for the full range of standard
/// chessboard images.
///
/// `DiskFit` *(= 1)* — full-disk crossing-line estimator. Samples all image
/// pixels in a disk around the corner center and fits two possibly
/// non-orthogonal axes. Use when corners are imaged under strong projective
/// warp. Falls back to `RingFit` on clean orthogonal corners.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrientationMethod {
    RingFit = 0,
    DiskFit = 1,
}

impl From<OrientationMethod> for RsOrientationMethod {
    fn from(v: OrientationMethod) -> Self {
        match v {
            OrientationMethod::RingFit => RsOrientationMethod::RingFit,
            OrientationMethod::DiskFit => RsOrientationMethod::DiskFit,
        }
    }
}

impl From<RsOrientationMethod> for OrientationMethod {
    fn from(v: RsOrientationMethod) -> Self {
        match v {
            RsOrientationMethod::RingFit => OrientationMethod::RingFit,
            RsOrientationMethod::DiskFit => OrientationMethod::DiskFit,
            // Any future variants map to the default.
            _ => OrientationMethod::RingFit,
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrategyKind {
    Chess,
    Radon,
}

impl StrategyKind {
    fn as_str(self) -> &'static str {
        match self {
            StrategyKind::Chess => "chess",
            StrategyKind::Radon => "radon",
        }
    }
}

// ---------------------------------------------------------------------------
// Threshold (tagged-class pattern, enum-with-payload)
// ---------------------------------------------------------------------------

/// Acceptance threshold for the corner detector.
///
/// Constructed via [`Threshold::absolute`] or [`Threshold::relative`].
/// Mirrors the [`chess_corners::Threshold`] payload-carrying enum.
///
/// - `Threshold.absolute(value)` accepts responses `≥ value` in the
///   detector's native score units.
/// - `Threshold.relative(frac)` accepts responses `≥ frac · max(response)`
///   in the current frame, with `frac ∈ [0.0, 1.0]`.
///
/// The `kind` getter returns `"absolute"` or `"relative"`; the `value`
/// getter returns the carried payload.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Threshold {
    cell: Cell<RsThreshold>,
}

#[wasm_bindgen]
impl Threshold {
    /// Construct an `Absolute(value)` threshold (the library default).
    pub fn absolute(value: f32) -> Self {
        Self {
            cell: cell(RsThreshold::Absolute(value)),
        }
    }

    /// Construct a `Relative(frac)` threshold. `frac` should lie in
    /// `[0.0, 1.0]`.
    pub fn relative(frac: f32) -> Self {
        Self {
            cell: cell(RsThreshold::Relative(frac)),
        }
    }

    /// Discriminant tag: `"absolute"` or `"relative"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        match *self.cell.borrow() {
            RsThreshold::Absolute(_) => "absolute".into(),
            RsThreshold::Relative(_) => "relative".into(),
            // Any future variant maps to the safe default.
            _ => "absolute".into(),
        }
    }

    /// Numeric payload (absolute value or relative fraction).
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> f32 {
        match *self.cell.borrow() {
            RsThreshold::Absolute(v) => v,
            RsThreshold::Relative(f) => f,
            _ => 0.0,
        }
    }

    /// In-place setter for the carried payload, preserving the kind.
    #[wasm_bindgen(setter)]
    pub fn set_value(&mut self, v: f32) {
        let mut slot = self.cell.borrow_mut();
        *slot = match *slot {
            RsThreshold::Absolute(_) => RsThreshold::Absolute(v),
            RsThreshold::Relative(_) => RsThreshold::Relative(v),
            _ => RsThreshold::Absolute(v),
        };
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self {
            cell: cell(RsThreshold::default()),
        }
    }
}

impl Threshold {
    pub(crate) fn share_cell(&self) -> Cell<RsThreshold> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsThreshold>) -> Self {
        Self { cell }
    }
    pub(crate) fn from_value(value: RsThreshold) -> Self {
        Self { cell: cell(value) }
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
        RsRefinerConfig::build(
            *self.kind.borrow(),
            *self.center_of_mass.borrow(),
            *self.forstner.borrow(),
            *self.saddle_point.borrow(),
            *self.radon_peak.borrow(),
        )
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
// MultiscaleParams
// ---------------------------------------------------------------------------

/// Multiscale pipeline parameters, attached to a [`ChessStrategy`] when
/// the user opts into pyramid detection. Mirrors
/// [`chess_corners::MultiscaleParams`].
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct MultiscaleParams {
    cell: Cell<RsMultiscaleParams>,
}

#[wasm_bindgen]
impl MultiscaleParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cell: cell(RsMultiscaleParams::default()),
        }
    }

    #[wasm_bindgen(getter, js_name = pyramidLevels)]
    pub fn pyramid_levels(&self) -> u8 {
        self.cell.borrow().pyramid_levels
    }
    #[wasm_bindgen(setter, js_name = pyramidLevels)]
    pub fn set_pyramid_levels(&mut self, v: u8) {
        self.cell.borrow_mut().pyramid_levels = v;
    }

    #[wasm_bindgen(getter, js_name = pyramidMinSize)]
    pub fn pyramid_min_size(&self) -> u32 {
        self.cell.borrow().pyramid_min_size as u32
    }
    #[wasm_bindgen(setter, js_name = pyramidMinSize)]
    pub fn set_pyramid_min_size(&mut self, v: u32) {
        self.cell.borrow_mut().pyramid_min_size = v as usize;
    }

    #[wasm_bindgen(getter, js_name = refinementRadius)]
    pub fn refinement_radius(&self) -> u32 {
        self.cell.borrow().refinement_radius
    }
    #[wasm_bindgen(setter, js_name = refinementRadius)]
    pub fn set_refinement_radius(&mut self, v: u32) {
        self.cell.borrow_mut().refinement_radius = v;
    }
}

impl Default for MultiscaleParams {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiscaleParams {
    pub(crate) fn share_cell(&self) -> Cell<RsMultiscaleParams> {
        Rc::clone(&self.cell)
    }
    pub(crate) fn from_cell(cell: Cell<RsMultiscaleParams>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// ChessStrategy
// ---------------------------------------------------------------------------

/// ChESS-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::ChessStrategy`].
///
/// `multiscale` is an optional handle: `null` means single-scale.
/// Use `enableMultiscale()` to attach a defaulted [`MultiscaleParams`]
/// or assign a custom one via the setter; `clearMultiscale()` returns
/// the strategy to single-scale.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessStrategy {
    ring: Cell<RsChessRing>,
    nms_radius: Cell<u32>,
    min_cluster_size: Cell<u32>,
    /// `None` cell means single-scale. Switching to multiscale stores
    /// `Some(cell)` where `cell` is the shared inner cell of a
    /// [`MultiscaleParams`] wrapper; the wrapper returned from
    /// `multiscale` getters reuses that cell so edits propagate.
    multiscale: Cell<Option<Cell<RsMultiscaleParams>>>,
}

#[wasm_bindgen]
impl ChessStrategy {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let defaults = RsChessStrategy::default();
        Self {
            ring: cell(defaults.ring),
            nms_radius: cell(defaults.nms_radius),
            min_cluster_size: cell(defaults.min_cluster_size),
            multiscale: cell(defaults.multiscale.map(cell)),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn ring(&self) -> ChessRing {
        (*self.ring.borrow()).into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_ring(&mut self, v: ChessRing) {
        *self.ring.borrow_mut() = v.into();
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

    /// Returns the attached multiscale params, or `null` for single-scale.
    /// Edits through the returned wrapper propagate into this strategy.
    #[wasm_bindgen(getter)]
    pub fn multiscale(&self) -> Option<MultiscaleParams> {
        self.multiscale
            .borrow()
            .as_ref()
            .map(|c| MultiscaleParams::from_cell(Rc::clone(c)))
    }
    #[wasm_bindgen(setter)]
    pub fn set_multiscale(&mut self, v: Option<MultiscaleParams>) {
        *self.multiscale.borrow_mut() = v.map(|w| w.share_cell());
    }

    /// Detach multiscale settings; the strategy reverts to single-scale.
    #[wasm_bindgen(js_name = clearMultiscale)]
    pub fn clear_multiscale(&mut self) {
        *self.multiscale.borrow_mut() = None;
    }

    /// Attach a defaulted [`MultiscaleParams`] and return a handle to
    /// it so the caller can immediately tweak fields. Idempotent: if
    /// multiscale is already attached, returns the existing handle.
    #[wasm_bindgen(js_name = enableMultiscale)]
    pub fn enable_multiscale(&mut self) -> MultiscaleParams {
        {
            let mut slot = self.multiscale.borrow_mut();
            if slot.is_none() {
                *slot = Some(cell(RsMultiscaleParams::default()));
            }
        }
        // Cannot hold the borrow_mut() across the second borrow().
        let c = self
            .multiscale
            .borrow()
            .as_ref()
            .map(Rc::clone)
            .expect("multiscale just initialized");
        MultiscaleParams::from_cell(c)
    }
}

impl Default for ChessStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ChessStrategy {
    fn from_value(value: RsChessStrategy) -> Self {
        Self {
            ring: cell(value.ring),
            nms_radius: cell(value.nms_radius),
            min_cluster_size: cell(value.min_cluster_size),
            multiscale: cell(value.multiscale.map(cell)),
        }
    }

    fn snapshot(&self) -> RsChessStrategy {
        let mut s = RsChessStrategy::default();
        s.ring = *self.ring.borrow();
        s.nms_radius = *self.nms_radius.borrow();
        s.min_cluster_size = *self.min_cluster_size.borrow();
        s.multiscale = self.multiscale.borrow().as_ref().map(|c| *c.borrow());
        s
    }
}

// ---------------------------------------------------------------------------
// RadonStrategy
// ---------------------------------------------------------------------------

/// Radon-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::RadonStrategy`]. All radii / counts are in
/// **working-resolution** pixels (i.e. after `imageUpsample`).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonStrategy {
    ray_radius: Cell<u32>,
    image_upsample: Cell<u32>,
    response_blur_radius: Cell<u32>,
    peak_fit: Cell<RsPeakFitMode>,
    nms_radius: Cell<u32>,
    min_cluster_size: Cell<u32>,
}

#[wasm_bindgen]
impl RadonStrategy {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let d = RsRadonStrategy::default();
        Self {
            ray_radius: cell(d.ray_radius),
            image_upsample: cell(d.image_upsample),
            response_blur_radius: cell(d.response_blur_radius),
            peak_fit: cell(d.peak_fit),
            nms_radius: cell(d.nms_radius),
            min_cluster_size: cell(d.min_cluster_size),
        }
    }

    #[wasm_bindgen(getter, js_name = rayRadius)]
    pub fn ray_radius(&self) -> u32 {
        *self.ray_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = rayRadius)]
    pub fn set_ray_radius(&mut self, v: u32) {
        *self.ray_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = imageUpsample)]
    pub fn image_upsample(&self) -> u32 {
        *self.image_upsample.borrow()
    }
    #[wasm_bindgen(setter, js_name = imageUpsample)]
    pub fn set_image_upsample(&mut self, v: u32) {
        *self.image_upsample.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = responseBlurRadius)]
    pub fn response_blur_radius(&self) -> u32 {
        *self.response_blur_radius.borrow()
    }
    #[wasm_bindgen(setter, js_name = responseBlurRadius)]
    pub fn set_response_blur_radius(&mut self, v: u32) {
        *self.response_blur_radius.borrow_mut() = v;
    }

    #[wasm_bindgen(getter, js_name = peakFit)]
    pub fn peak_fit(&self) -> PeakFitMode {
        (*self.peak_fit.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = peakFit)]
    pub fn set_peak_fit(&mut self, v: PeakFitMode) {
        *self.peak_fit.borrow_mut() = v.into();
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
}

impl Default for RadonStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonStrategy {
    fn from_value(value: RsRadonStrategy) -> Self {
        Self {
            ray_radius: cell(value.ray_radius),
            image_upsample: cell(value.image_upsample),
            response_blur_radius: cell(value.response_blur_radius),
            peak_fit: cell(value.peak_fit),
            nms_radius: cell(value.nms_radius),
            min_cluster_size: cell(value.min_cluster_size),
        }
    }

    fn snapshot(&self) -> RsRadonStrategy {
        let mut s = RsRadonStrategy::default();
        s.ray_radius = *self.ray_radius.borrow();
        s.image_upsample = *self.image_upsample.borrow();
        s.response_blur_radius = *self.response_blur_radius.borrow();
        s.peak_fit = *self.peak_fit.borrow();
        s.nms_radius = *self.nms_radius.borrow();
        s.min_cluster_size = *self.min_cluster_size.borrow();
        s
    }
}

// ---------------------------------------------------------------------------
// DetectionStrategy (tagged wrapper, mirrors RsDetectionStrategy)
// ---------------------------------------------------------------------------

/// Top-level detector dispatch. Mirrors
/// [`chess_corners::DetectionStrategy`].
///
/// The wrapper carries both a [`ChessStrategy`] and a [`RadonStrategy`]
/// cell internally. The `kind` discriminant selects which one is
/// snapshotted into the Rust value. Field edits on the inactive
/// variant are preserved in its cell — they become live again when
/// the discriminant flips.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectionStrategy {
    kind: Cell<StrategyKind>,
    chess: ChessStrategy,
    radon: RadonStrategy,
}

#[wasm_bindgen]
impl DetectionStrategy {
    /// Construct a [`DetectionStrategy`] in the ChESS branch from a
    /// given [`ChessStrategy`]. JS: `DetectionStrategy.fromChess(s)`.
    #[wasm_bindgen(js_name = fromChess)]
    pub fn from_chess(strategy: &ChessStrategy) -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: strategy.clone(),
            radon: RadonStrategy::default(),
        }
    }

    /// Construct a [`DetectionStrategy`] in the Radon branch from a
    /// given [`RadonStrategy`]. JS: `DetectionStrategy.fromRadon(s)`.
    #[wasm_bindgen(js_name = fromRadon)]
    pub fn from_radon(strategy: &RadonStrategy) -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessStrategy::default(),
            radon: strategy.clone(),
        }
    }

    /// Default ChESS-branch strategy. Equivalent to
    /// `DetectionStrategy.fromChess(new ChessStrategy())`.
    #[wasm_bindgen(js_name = defaultChess)]
    pub fn default_chess() -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: ChessStrategy::default(),
            radon: RadonStrategy::default(),
        }
    }

    /// Default Radon-branch strategy. Equivalent to
    /// `DetectionStrategy.fromRadon(new RadonStrategy())`.
    #[wasm_bindgen(js_name = defaultRadon)]
    pub fn default_radon() -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessStrategy::default(),
            radon: RadonStrategy::default(),
        }
    }

    /// Discriminant tag: `"chess"` or `"radon"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// ChESS branch wrapper. Edits propagate even when the active kind
    /// is Radon (the cell is preserved for when the user flips back).
    #[wasm_bindgen(getter)]
    pub fn chess(&self) -> ChessStrategy {
        self.chess.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_chess(&mut self, v: &ChessStrategy) {
        self.chess = v.clone();
        // Setting a chess strategy implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Radon branch wrapper. Edits propagate even when the active kind
    /// is ChESS.
    #[wasm_bindgen(getter)]
    pub fn radon(&self) -> RadonStrategy {
        self.radon.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_radon(&mut self, v: &RadonStrategy) {
        self.radon = v.clone();
        // Setting a radon strategy implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Radon;
    }

    /// Switch the discriminant to ChESS without otherwise mutating
    /// the wrapper. The previously stored [`ChessStrategy`] cell
    /// becomes the live source again.
    #[wasm_bindgen(js_name = useChess)]
    pub fn use_chess(&mut self) {
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Switch the discriminant to Radon without otherwise mutating
    /// the wrapper. The previously stored [`RadonStrategy`] cell
    /// becomes the live source again.
    #[wasm_bindgen(js_name = useRadon)]
    pub fn use_radon(&mut self) {
        *self.kind.borrow_mut() = StrategyKind::Radon;
    }
}

impl Default for DetectionStrategy {
    fn default() -> Self {
        Self::default_chess()
    }
}

impl DetectionStrategy {
    pub(crate) fn from_value(value: RsDetectionStrategy) -> Self {
        match value {
            RsDetectionStrategy::Chess(c) => Self {
                kind: cell(StrategyKind::Chess),
                chess: ChessStrategy::from_value(c),
                radon: RadonStrategy::default(),
            },
            RsDetectionStrategy::Radon(r) => Self {
                kind: cell(StrategyKind::Radon),
                chess: ChessStrategy::default(),
                radon: RadonStrategy::from_value(r),
            },
            _ => Self::default_chess(),
        }
    }

    pub(crate) fn snapshot(&self) -> RsDetectionStrategy {
        match *self.kind.borrow() {
            StrategyKind::Chess => RsDetectionStrategy::Chess(self.chess.snapshot()),
            StrategyKind::Radon => RsDetectionStrategy::Radon(self.radon.snapshot()),
        }
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
///
/// Build one with [`ChessConfig::single_scale`],
/// [`ChessConfig::multiscale`], or [`ChessConfig::radon`] and tweak
/// only the fields you need.
#[non_exhaustive]
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessConfig {
    strategy: DetectionStrategy,
    threshold: Threshold,
    refiner: RefinerConfig,
    orientation_method: Cell<RsOrientationMethod>,
    descriptor_mode: Cell<RsDescriptorMode>,
    upscale: Cell<RsUpscaleConfig>,
    merge_radius: Cell<f32>,
}

impl ChessConfig {
    pub(crate) fn from_value_pub(value: RsChessConfig) -> Self {
        Self::from_value(value)
    }

    fn from_value(value: RsChessConfig) -> Self {
        Self {
            strategy: DetectionStrategy::from_value(value.strategy),
            threshold: Threshold::from_value(value.threshold),
            refiner: RefinerConfig::from_value(value.refiner),
            orientation_method: cell(value.orientation_method),
            descriptor_mode: cell(value.descriptor_mode),
            upscale: cell(value.upscale),
            merge_radius: cell(value.merge_radius),
        }
    }

    /// Snapshot the current state into the Rust facade
    /// [`RsChessConfig`] for hand-off to the detector.
    pub(crate) fn snapshot(&self) -> RsChessConfig {
        let mut cfg = RsChessConfig::default();
        cfg.strategy = self.strategy.snapshot();
        cfg.threshold = *self.threshold.share_cell().borrow();
        cfg.refiner = self.refiner.snapshot();
        cfg.orientation_method = *self.orientation_method.borrow();
        cfg.descriptor_mode = *self.descriptor_mode.borrow();
        cfg.upscale = *self.upscale.borrow();
        cfg.merge_radius = *self.merge_radius.borrow();
        cfg
    }
}

#[wasm_bindgen]
impl ChessConfig {
    /// Construct a `ChessConfig` with library defaults
    /// (single-scale ChESS, absolute threshold = 0.0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsChessConfig::default())
    }

    /// Single-scale ChESS preset (alias for [`Self::new`]).
    #[wasm_bindgen(js_name = singleScale)]
    pub fn single_scale() -> Self {
        Self::from_value(RsChessConfig::single_scale())
    }

    /// Recommended 3-level multiscale ChESS preset.
    pub fn multiscale() -> Self {
        Self::from_value(RsChessConfig::multiscale())
    }

    /// Whole-image Radon detector preset (relative threshold 0.01).
    pub fn radon() -> Self {
        Self::from_value(RsChessConfig::radon())
    }

    // ---- Top-level fields ----

    #[wasm_bindgen(getter)]
    pub fn strategy(&self) -> DetectionStrategy {
        self.strategy.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_strategy(&mut self, v: &DetectionStrategy) {
        self.strategy = v.clone();
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> Threshold {
        self.threshold.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, v: &Threshold) {
        // Reseat the threshold cell so both the parent and any
        // downstream getter call observe `v`'s state.
        self.threshold = Threshold::from_cell(v.share_cell());
    }

    #[wasm_bindgen(getter, js_name = descriptorMode)]
    pub fn descriptor_mode(&self) -> DescriptorMode {
        (*self.descriptor_mode.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = descriptorMode)]
    pub fn set_descriptor_mode(&mut self, v: DescriptorMode) {
        *self.descriptor_mode.borrow_mut() = v.into();
    }

    #[wasm_bindgen(getter, js_name = orientationMethod)]
    pub fn orientation_method(&self) -> OrientationMethod {
        (*self.orientation_method.borrow()).into()
    }
    #[wasm_bindgen(setter, js_name = orientationMethod)]
    pub fn set_orientation_method(&mut self, v: OrientationMethod) {
        *self.orientation_method.borrow_mut() = v.into();
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
    use chess_corners::{
        DetectionStrategy as RsDetectionStrategy, RefinementMethod as RsRefinementMethod,
        Threshold as RsThreshold,
    };

    #[test]
    fn nested_edits_propagate_through_chess_config() {
        let cfg = ChessConfig::new();
        let mut r = cfg.refiner();
        r.set_kind(RefinementMethod::Forstner);

        assert_eq!(
            cfg.snapshot().refiner.kind,
            RsRefinementMethod::Forstner,
            "cfg.refiner.kind = X must propagate without round-trip"
        );

        let mut f = cfg.refiner().forstner();
        f.set_max_offset(2.5);
        assert_eq!(cfg.snapshot().refiner.forstner.max_offset, 2.5);
    }

    #[test]
    fn chess_strategy_field_edits_propagate() {
        let cfg = ChessConfig::single_scale();
        let mut chess = cfg.strategy().chess();
        chess.set_nms_radius(7);
        chess.set_ring(ChessRing::Broad);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(s) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert_eq!(s.nms_radius, 7);
        assert_eq!(s.ring, chess_corners::ChessRing::Broad);
    }

    #[test]
    fn enable_multiscale_attaches_and_edits_propagate() {
        let cfg = ChessConfig::single_scale();
        let mut chess = cfg.strategy().chess();
        let mut ms = chess.enable_multiscale();
        ms.set_pyramid_levels(5);
        ms.set_pyramid_min_size(64);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(s) = snap.strategy else {
            panic!("expected chess strategy")
        };
        let attached = s.multiscale.expect("multiscale should be attached");
        assert_eq!(attached.pyramid_levels, 5);
        assert_eq!(attached.pyramid_min_size, 64);
    }

    #[test]
    fn clear_multiscale_returns_to_single_scale() {
        let cfg = ChessConfig::multiscale();
        let mut chess = cfg.strategy().chess();
        assert!(chess.multiscale().is_some());
        chess.clear_multiscale();

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(s) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert!(s.multiscale.is_none());
    }

    #[test]
    fn radon_strategy_field_edits_propagate() {
        let cfg = ChessConfig::radon();
        let mut radon = cfg.strategy().radon();
        radon.set_ray_radius(7);
        radon.set_image_upsample(2);
        let snap = cfg.snapshot();
        let RsDetectionStrategy::Radon(s) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert_eq!(s.ray_radius, 7);
        assert_eq!(s.image_upsample, 2);
    }

    #[test]
    fn threshold_kind_and_value_round_trip() {
        let abs = Threshold::absolute(3.5);
        assert_eq!(abs.kind(), "absolute");
        assert!((abs.value() - 3.5).abs() < f32::EPSILON);

        let rel = Threshold::relative(0.42);
        assert_eq!(rel.kind(), "relative");
        assert!((rel.value() - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn cfg_threshold_propagates() {
        let mut cfg = ChessConfig::new();
        let t = Threshold::relative(0.15);
        cfg.set_threshold(&t);
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.threshold, RsThreshold::Relative(f) if (f - 0.15).abs() < f32::EPSILON)
        );

        // Mutating the wrapper via setter also propagates.
        let live = cfg.threshold();
        let mut live_mut = live;
        live_mut.set_value(0.25);
        let snap2 = cfg.snapshot();
        assert!(
            matches!(snap2.threshold, RsThreshold::Relative(f) if (f - 0.25).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn detection_strategy_switch_preserves_inactive_branch_state() {
        let mut ds = DetectionStrategy::default_chess();
        // Pre-populate the radon branch while chess is active.
        let mut radon = ds.radon();
        radon.set_ray_radius(11);
        radon.set_image_upsample(2);
        // Flip discriminant to radon — branch state must survive.
        ds.use_radon();
        assert_eq!(ds.kind(), "radon");
        match ds.snapshot() {
            RsDetectionStrategy::Radon(r) => assert_eq!(r.ray_radius, 11),
            other => panic!("expected radon, got {other:?}"),
        }
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
        let mut new_refiner = RefinerConfig::new();
        new_refiner.set_kind(RefinementMethod::SaddlePoint);
        new_refiner.forstner().set_max_offset(3.5);

        cfg.set_refiner(&new_refiner);

        let snap = cfg.snapshot();
        assert_eq!(snap.refiner.kind, RsRefinementMethod::SaddlePoint);
        assert_eq!(snap.refiner.forstner.max_offset, 3.5);

        // Reseating must take effect for future getter calls.
        new_refiner.forstner().set_max_offset(4.5);
        assert_eq!(cfg.snapshot().refiner.forstner.max_offset, 4.5);
    }

    #[test]
    fn snapshot_returns_independent_state() {
        let cfg = ChessConfig::new();
        let t = Threshold::absolute(0.1);
        let mut cfg_mut = cfg;
        cfg_mut.set_threshold(&t);
        let snap = cfg_mut.snapshot();
        // Replace the threshold after snapshotting — snapshot must not move.
        let t2 = Threshold::absolute(0.9);
        cfg_mut.set_threshold(&t2);
        assert!(matches!(snap.threshold, RsThreshold::Absolute(v) if (v - 0.1).abs() < 1e-6));
    }

    #[test]
    fn orientation_method_round_trips_all_variants() {
        let cases = [
            (OrientationMethod::RingFit, RsOrientationMethod::RingFit),
            (OrientationMethod::DiskFit, RsOrientationMethod::DiskFit),
        ];

        for (wasm_variant, rs_variant) in cases {
            let mut cfg = ChessConfig::new();
            cfg.set_orientation_method(wasm_variant);
            assert_eq!(cfg.orientation_method(), wasm_variant);
            let snap = cfg.snapshot();
            assert_eq!(
                snap.orientation_method, rs_variant,
                "snapshot mismatch for {wasm_variant:?}"
            );
        }
    }
}
