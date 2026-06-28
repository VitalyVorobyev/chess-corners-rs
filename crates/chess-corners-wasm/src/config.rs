//! Typed `#[wasm_bindgen]` wrappers around `chess-corners` config
//! structs.
//!
//! ## Live nested edits
//!
//! Each wrapper stores its inner Rust value in a shared
//! `Rc<RefCell<T>>` cell, and compound wrappers (`DetectorConfig`,
//! `DetectionStrategy`, `ChessConfig`, `RadonConfig`, `ChessRefiner`,
//! `RadonRefiner`) hold `Rc` handles to their children's cells. A
//! getter returns a wrapper backed by the same cell as the parent,
//! so chained mutation propagates without a round-trip:
//!
//! ```js
//! const cfg = DetectorConfig.chessMultiscale();
//! cfg.strategy.chess.ring = ChessRing.Broad;             // works
//! cfg.strategy.chess.refiner.forstner.maxOffset = 2.0;   // works
//! cfg.detection.nmsRadius = 3;                           // works
//! cfg.multiscale = MultiscaleConfig.pyramid(4, 64, 3);   // works
//! ```
//!
//! Setters that take a nested wrapper (e.g. `cfg.strategy = newDs`)
//! reseat the parent's `Rc` to point at the new value's cell, so
//! future getter calls return wrappers backed by the new cell. Any
//! JS reference held to the *previous* nested wrapper still
//! observes the previous cell — matching natural attribute-
//! reassignment semantics in JS.
//!
//! ## Strategy discriminants
//!
//! [`DetectionStrategy`] is a tagged wrapper carrying both a
//! [`ChessConfig`] and a [`RadonConfig`] cell plus an internal
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
    ChessRefiner as RsChessRefiner, ChessRing as RsChessRing, DetectionParams as RsDetectionParams,
    DetectionStrategy as RsDetectionStrategy, DetectorConfig as RsDetectorConfig,
    ForstnerConfig as RsForstnerConfig, MultiscaleConfig as RsMultiscaleConfig,
    OrientationMethod as RsOrientationMethod, PeakFitMode as RsPeakFitMode,
    RadonConfig as RsRadonConfig, RadonPeakConfig as RsRadonPeakConfig,
    RadonRefiner as RsRadonRefiner, SaddlePointConfig as RsSaddlePointConfig,
    UpscaleConfig as RsUpscaleConfig,
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
// ChessRefiner discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChessRefinerKind {
    CenterOfMass,
    Forstner,
    SaddlePoint,
    #[cfg(feature = "ml-refiner")]
    Ml,
}

impl ChessRefinerKind {
    fn as_str(self) -> &'static str {
        match self {
            ChessRefinerKind::CenterOfMass => "center_of_mass",
            ChessRefinerKind::Forstner => "forstner",
            ChessRefinerKind::SaddlePoint => "saddle_point",
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => "ml",
        }
    }
}

// ---------------------------------------------------------------------------
// ChessRefiner (tagged class)
// ---------------------------------------------------------------------------

/// Subpixel refiner selection for the ChESS detector.
///
/// Construct one via the static factories
/// [`ChessRefiner::center_of_mass`], [`ChessRefiner::forstner`],
/// [`ChessRefiner::saddle_point`], or — when the `ml-refiner` Cargo
/// feature is enabled — `ChessRefiner::ml`. In JS the factory names
/// are `ChessRefiner.withCenterOfMass(...)`,
/// `ChessRefiner.withForstner(...)`,
/// `ChessRefiner.withSaddlePoint(...)`, and `ChessRefiner.withMl()`.
/// Instance getters (`refiner.centerOfMass`, `refiner.forstner`,
/// `refiner.saddlePoint`) hand back the corresponding payload
/// wrapper.
///
/// The wrapper holds a cell for every payload variant so payload edits
/// survive across `kind` flips. Only the cell matching the active
/// `kind` is read at snapshot time.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessRefiner {
    kind: Cell<ChessRefinerKind>,
    center_of_mass: Cell<RsCenterOfMassConfig>,
    forstner: Cell<RsForstnerConfig>,
    saddle_point: Cell<RsSaddlePointConfig>,
}

#[wasm_bindgen]
impl ChessRefiner {
    /// Center-of-mass refinement on the response map. JS:
    /// `ChessRefiner.withCenterOfMass(cfg)`.
    #[wasm_bindgen(js_name = withCenterOfMass)]
    pub fn center_of_mass(cfg: &CenterOfMassConfig) -> Self {
        Self {
            kind: cell(ChessRefinerKind::CenterOfMass),
            center_of_mass: cfg.share_cell(),
            forstner: cell(RsForstnerConfig::default()),
            saddle_point: cell(RsSaddlePointConfig::default()),
        }
    }

    /// Förstner structure-tensor refinement on the image patch. JS:
    /// `ChessRefiner.withForstner(cfg)`.
    #[wasm_bindgen(js_name = withForstner)]
    pub fn forstner(cfg: &ForstnerConfig) -> Self {
        Self {
            kind: cell(ChessRefinerKind::Forstner),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
            forstner: cfg.share_cell(),
            saddle_point: cell(RsSaddlePointConfig::default()),
        }
    }

    /// Quadratic saddle-point refinement. JS:
    /// `ChessRefiner.withSaddlePoint(cfg)`.
    #[wasm_bindgen(js_name = withSaddlePoint)]
    pub fn saddle_point(cfg: &SaddlePointConfig) -> Self {
        Self {
            kind: cell(ChessRefinerKind::SaddlePoint),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
            forstner: cell(RsForstnerConfig::default()),
            saddle_point: cfg.share_cell(),
        }
    }

    /// ML-backed refinement (requires the `ml-refiner` Cargo feature).
    /// JS: `ChessRefiner.withMl()`.
    #[cfg(feature = "ml-refiner")]
    #[wasm_bindgen(js_name = withMl)]
    pub fn ml() -> Self {
        Self {
            kind: cell(ChessRefinerKind::Ml),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
            forstner: cell(RsForstnerConfig::default()),
            saddle_point: cell(RsSaddlePointConfig::default()),
        }
    }

    /// Discriminant tag: one of `"center_of_mass"`, `"forstner"`,
    /// `"saddle_point"`, `"ml"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// Center-of-mass payload wrapper. Edits propagate even when the
    /// active `kind` is something else (the cell is preserved for when
    /// the user flips back).
    #[wasm_bindgen(getter, js_name = centerOfMass)]
    pub fn get_center_of_mass(&self) -> CenterOfMassConfig {
        CenterOfMassConfig::from_cell(Rc::clone(&self.center_of_mass))
    }
    #[wasm_bindgen(setter, js_name = centerOfMass)]
    pub fn set_center_of_mass(&mut self, v: &CenterOfMassConfig) {
        // Copy contents into the existing cell so cell-sharing
        // observers (any parent ChessConfig referencing this Rc) see
        // the update.
        *self.center_of_mass.borrow_mut() = *v.cell.borrow();
        *self.kind.borrow_mut() = ChessRefinerKind::CenterOfMass;
    }

    /// Förstner payload wrapper.
    #[wasm_bindgen(getter, js_name = forstner)]
    pub fn get_forstner(&self) -> ForstnerConfig {
        ForstnerConfig::from_cell(Rc::clone(&self.forstner))
    }
    #[wasm_bindgen(setter, js_name = forstner)]
    pub fn set_forstner(&mut self, v: &ForstnerConfig) {
        *self.forstner.borrow_mut() = *v.cell.borrow();
        *self.kind.borrow_mut() = ChessRefinerKind::Forstner;
    }

    /// Saddle-point payload wrapper.
    #[wasm_bindgen(getter, js_name = saddlePoint)]
    pub fn get_saddle_point(&self) -> SaddlePointConfig {
        SaddlePointConfig::from_cell(Rc::clone(&self.saddle_point))
    }
    #[wasm_bindgen(setter, js_name = saddlePoint)]
    pub fn set_saddle_point(&mut self, v: &SaddlePointConfig) {
        *self.saddle_point.borrow_mut() = *v.cell.borrow();
        *self.kind.borrow_mut() = ChessRefinerKind::SaddlePoint;
    }

    /// Switch the discriminant to `"ml"` without otherwise mutating the
    /// payload cells. Available only when the `ml-refiner` feature is
    /// enabled.
    #[cfg(feature = "ml-refiner")]
    #[wasm_bindgen(js_name = useMl)]
    pub fn use_ml(&mut self) {
        *self.kind.borrow_mut() = ChessRefinerKind::Ml;
    }
}

impl Default for ChessRefiner {
    fn default() -> Self {
        Self {
            kind: cell(ChessRefinerKind::CenterOfMass),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
            forstner: cell(RsForstnerConfig::default()),
            saddle_point: cell(RsSaddlePointConfig::default()),
        }
    }
}

impl ChessRefiner {
    fn from_value(value: RsChessRefiner) -> Self {
        let me = Self::default();
        match value {
            RsChessRefiner::CenterOfMass(cfg) => {
                *me.kind.borrow_mut() = ChessRefinerKind::CenterOfMass;
                *me.center_of_mass.borrow_mut() = cfg;
            }
            RsChessRefiner::Forstner(cfg) => {
                *me.kind.borrow_mut() = ChessRefinerKind::Forstner;
                *me.forstner.borrow_mut() = cfg;
            }
            RsChessRefiner::SaddlePoint(cfg) => {
                *me.kind.borrow_mut() = ChessRefinerKind::SaddlePoint;
                *me.saddle_point.borrow_mut() = cfg;
            }
            #[cfg(feature = "ml-refiner")]
            RsChessRefiner::Ml => {
                *me.kind.borrow_mut() = ChessRefinerKind::Ml;
            }
            _ => {}
        }
        me
    }

    fn snapshot(&self) -> RsChessRefiner {
        match *self.kind.borrow() {
            ChessRefinerKind::CenterOfMass => {
                RsChessRefiner::CenterOfMass(*self.center_of_mass.borrow())
            }
            ChessRefinerKind::Forstner => RsChessRefiner::Forstner(*self.forstner.borrow()),
            ChessRefinerKind::SaddlePoint => {
                RsChessRefiner::SaddlePoint(*self.saddle_point.borrow())
            }
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => RsChessRefiner::Ml,
        }
    }
}

// ---------------------------------------------------------------------------
// RadonRefiner discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RadonRefinerKind {
    RadonPeak,
    CenterOfMass,
}

impl RadonRefinerKind {
    fn as_str(self) -> &'static str {
        match self {
            RadonRefinerKind::RadonPeak => "radon_peak",
            RadonRefinerKind::CenterOfMass => "center_of_mass",
        }
    }
}

// ---------------------------------------------------------------------------
// RadonRefiner (tagged class)
// ---------------------------------------------------------------------------

/// Subpixel refiner selection for the whole-image Radon detector.
///
/// Construct one via [`RadonRefiner::radon_peak`] or
/// [`RadonRefiner::center_of_mass`]. In JS these are
/// `RadonRefiner.withRadonPeak(...)` and
/// `RadonRefiner.withCenterOfMass(...)`. Instance getters
/// (`refiner.radonPeak`, `refiner.centerOfMass`) hand back the
/// corresponding payload wrapper.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonRefiner {
    kind: Cell<RadonRefinerKind>,
    radon_peak: Cell<RsRadonPeakConfig>,
    center_of_mass: Cell<RsCenterOfMassConfig>,
}

#[wasm_bindgen]
impl RadonRefiner {
    /// Radon-projection refinement along candidate axes. JS:
    /// `RadonRefiner.withRadonPeak(cfg)`.
    #[wasm_bindgen(js_name = withRadonPeak)]
    pub fn radon_peak(cfg: &RadonPeakConfig) -> Self {
        Self {
            kind: cell(RadonRefinerKind::RadonPeak),
            radon_peak: cfg.share_cell(),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
        }
    }

    /// Center-of-mass refinement on the response map. JS:
    /// `RadonRefiner.withCenterOfMass(cfg)`.
    #[wasm_bindgen(js_name = withCenterOfMass)]
    pub fn center_of_mass(cfg: &CenterOfMassConfig) -> Self {
        Self {
            kind: cell(RadonRefinerKind::CenterOfMass),
            radon_peak: cell(RsRadonPeakConfig::default()),
            center_of_mass: cfg.share_cell(),
        }
    }

    /// Discriminant tag: `"radon_peak"` or `"center_of_mass"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// Radon-peak payload wrapper.
    #[wasm_bindgen(getter, js_name = radonPeak)]
    pub fn get_radon_peak(&self) -> RadonPeakConfig {
        RadonPeakConfig::from_cell(Rc::clone(&self.radon_peak))
    }
    #[wasm_bindgen(setter, js_name = radonPeak)]
    pub fn set_radon_peak(&mut self, v: &RadonPeakConfig) {
        *self.radon_peak.borrow_mut() = *v.cell.borrow();
        *self.kind.borrow_mut() = RadonRefinerKind::RadonPeak;
    }

    /// Center-of-mass payload wrapper.
    #[wasm_bindgen(getter, js_name = centerOfMass)]
    pub fn get_center_of_mass(&self) -> CenterOfMassConfig {
        CenterOfMassConfig::from_cell(Rc::clone(&self.center_of_mass))
    }
    #[wasm_bindgen(setter, js_name = centerOfMass)]
    pub fn set_center_of_mass(&mut self, v: &CenterOfMassConfig) {
        *self.center_of_mass.borrow_mut() = *v.cell.borrow();
        *self.kind.borrow_mut() = RadonRefinerKind::CenterOfMass;
    }
}

impl Default for RadonRefiner {
    fn default() -> Self {
        Self {
            kind: cell(RadonRefinerKind::RadonPeak),
            radon_peak: cell(RsRadonPeakConfig::default()),
            center_of_mass: cell(RsCenterOfMassConfig::default()),
        }
    }
}

impl RadonRefiner {
    fn from_value(value: RsRadonRefiner) -> Self {
        let me = Self::default();
        match value {
            RsRadonRefiner::RadonPeak(cfg) => {
                *me.kind.borrow_mut() = RadonRefinerKind::RadonPeak;
                *me.radon_peak.borrow_mut() = cfg;
            }
            RsRadonRefiner::CenterOfMass(cfg) => {
                *me.kind.borrow_mut() = RadonRefinerKind::CenterOfMass;
                *me.center_of_mass.borrow_mut() = cfg;
            }
            _ => {}
        }
        me
    }

    fn snapshot(&self) -> RsRadonRefiner {
        match *self.kind.borrow() {
            RadonRefinerKind::RadonPeak => RsRadonRefiner::RadonPeak(*self.radon_peak.borrow()),
            RadonRefinerKind::CenterOfMass => {
                RsRadonRefiner::CenterOfMass(*self.center_of_mass.borrow())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MultiscaleConfig discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MultiscaleKind {
    SingleScale,
    Pyramid,
}

impl MultiscaleKind {
    fn as_str(self) -> &'static str {
        match self {
            MultiscaleKind::SingleScale => "single_scale",
            MultiscaleKind::Pyramid => "pyramid",
        }
    }
}

// ---------------------------------------------------------------------------
// MultiscaleConfig (tagged class)
// ---------------------------------------------------------------------------

/// Multiscale pipeline configuration. Mirrors
/// [`chess_corners::MultiscaleConfig`].
///
/// Construct via [`MultiscaleConfig::single_scale`] (default) or
/// [`MultiscaleConfig::pyramid`] (in JS: `MultiscaleConfig.singleScale()`
/// and `MultiscaleConfig.pyramid(...)`). Honoured by both ChESS and
/// Radon strategies.
///
/// The `kind` getter returns `"single_scale"` or `"pyramid"`. Per-variant
/// getters (`levels`, `minSize`, `refinementRadius`) are only meaningful
/// when `kind == "pyramid"`; setting them when `kind == "single_scale"`
/// throws a JS error.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct MultiscaleConfig {
    kind: Cell<MultiscaleKind>,
    levels: Cell<u8>,
    min_size: Cell<usize>,
    refinement_radius: Cell<u32>,
}

#[wasm_bindgen]
impl MultiscaleConfig {
    /// Construct a single-scale configuration (no pyramid).
    #[wasm_bindgen(js_name = singleScale)]
    pub fn single_scale() -> Self {
        // Carry sensible defaults for the pyramid payload so flipping
        // `kind` later via a setter does not require the JS user to
        // populate every field.
        Self {
            kind: cell(MultiscaleKind::SingleScale),
            levels: cell(3),
            min_size: cell(128),
            refinement_radius: cell(3),
        }
    }

    /// Library-default three-level pyramid configuration (levels = 3,
    /// minSize = 128, refinementRadius = 3). Matches the preset used by
    /// [`DetectorConfig::chess_multiscale`] and
    /// [`DetectorConfig::radon_multiscale`]. JS: `MultiscaleConfig.pyramidDefault()`.
    #[wasm_bindgen(js_name = pyramidDefault)]
    pub fn pyramid_default() -> Self {
        Self::pyramid(3, 128, 3)
    }

    /// Construct a coarse-to-fine pyramid configuration with the given
    /// number of levels, minimum short-edge size in pixels, and
    /// refinement ROI half-radius in coarse-level pixels. JS:
    /// `MultiscaleConfig.pyramid(levels, minSize, refinementRadius)`
    /// (positional arguments).
    pub fn pyramid(levels: u8, min_size: u32, refinement_radius: u32) -> Self {
        Self {
            kind: cell(MultiscaleKind::Pyramid),
            levels: cell(levels),
            min_size: cell(min_size as usize),
            refinement_radius: cell(refinement_radius),
        }
    }

    /// Discriminant tag: `"single_scale"` or `"pyramid"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// Number of pyramid levels (≥ 1). Throws when `kind ==
    /// "single_scale"`.
    #[wasm_bindgen(getter)]
    pub fn levels(&self) -> Result<u8, JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => Ok(*self.levels.borrow()),
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "levels is only valid when kind == \"pyramid\"",
            )),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_levels(&mut self, v: u8) -> Result<(), JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => {
                *self.levels.borrow_mut() = v;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "cannot set levels when kind == \"single_scale\"; use MultiscaleConfig.pyramid(...) first",
            )),
        }
    }

    /// Minimum short-edge length in pixels. Throws when `kind ==
    /// "single_scale"`.
    #[wasm_bindgen(getter, js_name = minSize)]
    pub fn min_size(&self) -> Result<u32, JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => Ok(*self.min_size.borrow() as u32),
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "minSize is only valid when kind == \"pyramid\"",
            )),
        }
    }
    #[wasm_bindgen(setter, js_name = minSize)]
    pub fn set_min_size(&mut self, v: u32) -> Result<(), JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => {
                *self.min_size.borrow_mut() = v as usize;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "cannot set minSize when kind == \"single_scale\"; use MultiscaleConfig.pyramid(...) first",
            )),
        }
    }

    /// Refinement ROI half-radius in coarse-level pixels. Throws when
    /// `kind == "single_scale"`.
    #[wasm_bindgen(getter, js_name = refinementRadius)]
    pub fn refinement_radius(&self) -> Result<u32, JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => Ok(*self.refinement_radius.borrow()),
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "refinementRadius is only valid when kind == \"pyramid\"",
            )),
        }
    }
    #[wasm_bindgen(setter, js_name = refinementRadius)]
    pub fn set_refinement_radius(&mut self, v: u32) -> Result<(), JsValue> {
        match *self.kind.borrow() {
            MultiscaleKind::Pyramid => {
                *self.refinement_radius.borrow_mut() = v;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(JsValue::from_str(
                "cannot set refinementRadius when kind == \"single_scale\"; use MultiscaleConfig.pyramid(...) first",
            )),
        }
    }
}

impl Default for MultiscaleConfig {
    fn default() -> Self {
        Self::single_scale()
    }
}

impl MultiscaleConfig {
    fn from_value(value: RsMultiscaleConfig) -> Self {
        match value {
            RsMultiscaleConfig::SingleScale => Self::single_scale(),
            RsMultiscaleConfig::Pyramid {
                levels,
                min_size,
                refinement_radius,
            } => Self {
                kind: cell(MultiscaleKind::Pyramid),
                levels: cell(levels),
                min_size: cell(min_size),
                refinement_radius: cell(refinement_radius),
            },
            _ => Self::single_scale(),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers see the update.
    fn copy_from(&self, other: &MultiscaleConfig) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        *self.levels.borrow_mut() = *other.levels.borrow();
        *self.min_size.borrow_mut() = *other.min_size.borrow();
        *self.refinement_radius.borrow_mut() = *other.refinement_radius.borrow();
    }

    fn snapshot(&self) -> RsMultiscaleConfig {
        match *self.kind.borrow() {
            MultiscaleKind::SingleScale => RsMultiscaleConfig::SingleScale,
            MultiscaleKind::Pyramid => RsMultiscaleConfig::Pyramid {
                levels: *self.levels.borrow(),
                min_size: *self.min_size.borrow(),
                refinement_radius: *self.refinement_radius.borrow(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// UpscaleConfig discriminant (internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpscaleKind {
    Disabled,
    Fixed,
}

impl UpscaleKind {
    fn as_str(self) -> &'static str {
        match self {
            UpscaleKind::Disabled => "disabled",
            UpscaleKind::Fixed => "fixed",
        }
    }
}

// ---------------------------------------------------------------------------
// UpscaleConfig (tagged class)
// ---------------------------------------------------------------------------

/// Pre-pipeline integer-factor upscaling configuration. Mirrors
/// [`chess_corners::UpscaleConfig`].
///
/// Construct via [`UpscaleConfig::disabled`] (default) or
/// [`UpscaleConfig::fixed`]. Allowed integer factors are `2`, `3`, `4`.
///
/// The `kind` getter returns `"disabled"` or `"fixed"`. The `factor`
/// getter is only meaningful when `kind == "fixed"`; setting it when
/// `kind == "disabled"` throws a JS error.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct UpscaleConfig {
    kind: Cell<UpscaleKind>,
    factor: Cell<u32>,
}

#[wasm_bindgen]
impl UpscaleConfig {
    /// Construct a disabled upscale configuration (no upscaling).
    pub fn disabled() -> Self {
        Self {
            kind: cell(UpscaleKind::Disabled),
            factor: cell(1),
        }
    }

    /// Construct a fixed-factor upscale configuration. Allowed factors
    /// are `2`, `3`, `4`; validation happens at detector-construction
    /// time.
    pub fn fixed(factor: u32) -> Self {
        Self {
            kind: cell(UpscaleKind::Fixed),
            factor: cell(factor),
        }
    }

    /// Discriminant tag: `"disabled"` or `"fixed"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.borrow().as_str().into()
    }

    /// Integer upscale factor. Throws when `kind == "disabled"`.
    #[wasm_bindgen(getter)]
    pub fn factor(&self) -> Result<u32, JsValue> {
        match *self.kind.borrow() {
            UpscaleKind::Fixed => Ok(*self.factor.borrow()),
            UpscaleKind::Disabled => Err(JsValue::from_str(
                "factor is only valid when kind == \"fixed\"",
            )),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_factor(&mut self, v: u32) -> Result<(), JsValue> {
        match *self.kind.borrow() {
            UpscaleKind::Fixed => {
                *self.factor.borrow_mut() = v;
                Ok(())
            }
            UpscaleKind::Disabled => Err(JsValue::from_str(
                "cannot set factor when kind == \"disabled\"; use UpscaleConfig.fixed(...) first",
            )),
        }
    }
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl UpscaleConfig {
    fn from_value(value: RsUpscaleConfig) -> Self {
        match value {
            RsUpscaleConfig::Disabled => Self::disabled(),
            RsUpscaleConfig::Fixed(k) => Self::fixed(k),
            _ => Self::disabled(),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers see the update.
    fn copy_from(&self, other: &UpscaleConfig) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        *self.factor.borrow_mut() = *other.factor.borrow();
    }

    fn snapshot(&self) -> RsUpscaleConfig {
        match *self.kind.borrow() {
            UpscaleKind::Disabled => RsUpscaleConfig::Disabled,
            UpscaleKind::Fixed => RsUpscaleConfig::Fixed(*self.factor.borrow()),
        }
    }
}

// ---------------------------------------------------------------------------
// ChessConfig
// ---------------------------------------------------------------------------

/// ChESS-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::ChessConfig`].
///
/// All fields — including the refiner discriminant and payloads — are
/// stored as shared `Rc<RefCell<…>>` cells. The `refiner` getter
/// assembles a [`ChessRefiner`] wrapper out of clones of those cells,
/// so edits chained as `cfg.strategy.chess.refiner.forstner.maxOffset
/// = 2.5` propagate through the chess config without a round-trip.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChessConfig {
    ring: Cell<RsChessRing>,
    refiner_kind: Cell<ChessRefinerKind>,
    refiner_center_of_mass: Cell<RsCenterOfMassConfig>,
    refiner_forstner: Cell<RsForstnerConfig>,
    refiner_saddle_point: Cell<RsSaddlePointConfig>,
}

#[wasm_bindgen]
impl ChessConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsChessConfig::default())
    }

    #[wasm_bindgen(getter)]
    pub fn ring(&self) -> ChessRing {
        (*self.ring.borrow()).into()
    }
    #[wasm_bindgen(setter)]
    pub fn set_ring(&mut self, v: ChessRing) {
        *self.ring.borrow_mut() = v.into();
    }

    /// Subpixel refiner. Returns a wrapper that shares cells with this
    /// config; edits propagate without a round-trip.
    #[wasm_bindgen(getter)]
    pub fn refiner(&self) -> ChessRefiner {
        ChessRefiner {
            kind: Rc::clone(&self.refiner_kind),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
            forstner: Rc::clone(&self.refiner_forstner),
            saddle_point: Rc::clone(&self.refiner_saddle_point),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_refiner(&mut self, v: &ChessRefiner) {
        // Copy v's cell contents into this config's existing cells.
        // Cell-sharing observers (any clone of this ChessConfig
        // referencing the same Rc) see the update.
        *self.refiner_kind.borrow_mut() = *v.kind.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *v.center_of_mass.borrow();
        *self.refiner_forstner.borrow_mut() = *v.forstner.borrow();
        *self.refiner_saddle_point.borrow_mut() = *v.saddle_point.borrow();
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ChessConfig {
    fn from_value(value: RsChessConfig) -> Self {
        let refiner = ChessRefiner::from_value(value.refiner);
        Self {
            ring: cell(value.ring),
            refiner_kind: refiner.kind,
            refiner_center_of_mass: refiner.center_of_mass,
            refiner_forstner: refiner.forstner,
            refiner_saddle_point: refiner.saddle_point,
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`)
    /// see the update.
    fn copy_from(&self, other: &ChessConfig) {
        *self.ring.borrow_mut() = *other.ring.borrow();
        *self.refiner_kind.borrow_mut() = *other.refiner_kind.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *other.refiner_center_of_mass.borrow();
        *self.refiner_forstner.borrow_mut() = *other.refiner_forstner.borrow();
        *self.refiner_saddle_point.borrow_mut() = *other.refiner_saddle_point.borrow();
    }

    fn snapshot(&self) -> RsChessConfig {
        // Build a transient ChessRefiner wrapper sharing this config's
        // cells, then read its snapshot. The wrapper does no work on
        // construction (cell-handle clones only).
        let refiner_view = ChessRefiner {
            kind: Rc::clone(&self.refiner_kind),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
            forstner: Rc::clone(&self.refiner_forstner),
            saddle_point: Rc::clone(&self.refiner_saddle_point),
        };
        let mut s = RsChessConfig::default();
        s.ring = *self.ring.borrow();
        s.refiner = refiner_view.snapshot();
        s
    }
}

// ---------------------------------------------------------------------------
// RadonConfig
// ---------------------------------------------------------------------------

/// Radon-detector branch of [`DetectionStrategy`]. Mirrors
/// [`chess_corners::RadonConfig`]. All radii / counts are in
/// **working-resolution** pixels (i.e. after `imageUpsample`).
///
/// All fields — including the refiner discriminant and payloads — are
/// stored as shared `Rc<RefCell<…>>` cells. The `refiner` getter
/// assembles a [`RadonRefiner`] wrapper out of clones of those cells,
/// so nested chained edits propagate through the radon config.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct RadonConfig {
    ray_radius: Cell<u32>,
    image_upsample: Cell<u32>,
    response_blur_radius: Cell<u32>,
    peak_fit: Cell<RsPeakFitMode>,
    refiner_kind: Cell<RadonRefinerKind>,
    refiner_radon_peak: Cell<RsRadonPeakConfig>,
    refiner_center_of_mass: Cell<RsCenterOfMassConfig>,
}

#[wasm_bindgen]
impl RadonConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsRadonConfig::default())
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

    /// Subpixel refiner. Returns a wrapper that shares cells with this
    /// config; edits propagate without a round-trip.
    #[wasm_bindgen(getter)]
    pub fn refiner(&self) -> RadonRefiner {
        RadonRefiner {
            kind: Rc::clone(&self.refiner_kind),
            radon_peak: Rc::clone(&self.refiner_radon_peak),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
        }
    }
    #[wasm_bindgen(setter)]
    pub fn set_refiner(&mut self, v: &RadonRefiner) {
        // Copy v's cell contents into this config's existing cells.
        *self.refiner_kind.borrow_mut() = *v.kind.borrow();
        *self.refiner_radon_peak.borrow_mut() = *v.radon_peak.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *v.center_of_mass.borrow();
    }
}

impl Default for RadonConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RadonConfig {
    fn from_value(value: RsRadonConfig) -> Self {
        let refiner = RadonRefiner::from_value(value.refiner);
        Self {
            ray_radius: cell(value.ray_radius),
            image_upsample: cell(value.image_upsample),
            response_blur_radius: cell(value.response_blur_radius),
            peak_fit: cell(value.peak_fit),
            refiner_kind: refiner.kind,
            refiner_radon_peak: refiner.radon_peak,
            refiner_center_of_mass: refiner.center_of_mass,
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`)
    /// see the update.
    fn copy_from(&self, other: &RadonConfig) {
        *self.ray_radius.borrow_mut() = *other.ray_radius.borrow();
        *self.image_upsample.borrow_mut() = *other.image_upsample.borrow();
        *self.response_blur_radius.borrow_mut() = *other.response_blur_radius.borrow();
        *self.peak_fit.borrow_mut() = *other.peak_fit.borrow();
        *self.refiner_kind.borrow_mut() = *other.refiner_kind.borrow();
        *self.refiner_radon_peak.borrow_mut() = *other.refiner_radon_peak.borrow();
        *self.refiner_center_of_mass.borrow_mut() = *other.refiner_center_of_mass.borrow();
    }

    fn snapshot(&self) -> RsRadonConfig {
        let refiner_view = RadonRefiner {
            kind: Rc::clone(&self.refiner_kind),
            radon_peak: Rc::clone(&self.refiner_radon_peak),
            center_of_mass: Rc::clone(&self.refiner_center_of_mass),
        };
        let mut s = RsRadonConfig::default();
        s.ray_radius = *self.ray_radius.borrow();
        s.image_upsample = *self.image_upsample.borrow();
        s.response_blur_radius = *self.response_blur_radius.borrow();
        s.peak_fit = *self.peak_fit.borrow();
        s.refiner = refiner_view.snapshot();
        s
    }
}

// ---------------------------------------------------------------------------
// DetectionParams (shared NMS / clustering knobs)
// ---------------------------------------------------------------------------

/// Shared non-maximum-suppression and peak-clustering thresholds.
/// Mirrors [`chess_corners::DetectionParams`]. Honoured by both the
/// ChESS and Radon strategies; lives on
/// [`DetectorConfig`](DetectorConfig)'s `detection` field. All counts
/// are in working-resolution pixels.
///
/// Cell-backed so the `detection` getter returns a wrapper sharing the
/// parent's cells; edits chained as `cfg.detection.nmsRadius = 3`
/// propagate without a round-trip.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectionParams {
    nms_radius: Cell<u32>,
    min_cluster_size: Cell<u32>,
}

#[wasm_bindgen]
impl DetectionParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_value(RsDetectionParams::default())
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

impl Default for DetectionParams {
    fn default() -> Self {
        Self::new()
    }
}

impl DetectionParams {
    fn from_value(value: RsDetectionParams) -> Self {
        Self {
            nms_radius: cell(value.nms_radius),
            min_cluster_size: cell(value.min_cluster_size),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers (any clone of `self`) see
    /// the update.
    fn copy_from(&self, other: &DetectionParams) {
        *self.nms_radius.borrow_mut() = *other.nms_radius.borrow();
        *self.min_cluster_size.borrow_mut() = *other.min_cluster_size.borrow();
    }

    fn snapshot(&self) -> RsDetectionParams {
        let mut s = RsDetectionParams::default();
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
/// The wrapper carries both a [`ChessConfig`] and a [`RadonConfig`]
/// cell internally. The `kind` discriminant selects which one is
/// snapshotted into the Rust value. Field edits on the inactive
/// variant are preserved in its cell — they become live again when
/// the discriminant flips.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DetectionStrategy {
    kind: Cell<StrategyKind>,
    chess: ChessConfig,
    radon: RadonConfig,
}

#[wasm_bindgen]
impl DetectionStrategy {
    /// Construct a [`DetectionStrategy`] in the ChESS branch from a
    /// given [`ChessConfig`]. JS: `DetectionStrategy.fromChess(c)`.
    #[wasm_bindgen(js_name = fromChess)]
    pub fn from_chess(strategy: &ChessConfig) -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: strategy.clone(),
            radon: RadonConfig::default(),
        }
    }

    /// Construct a [`DetectionStrategy`] in the Radon branch from a
    /// given [`RadonConfig`]. JS: `DetectionStrategy.fromRadon(c)`.
    #[wasm_bindgen(js_name = fromRadon)]
    pub fn from_radon(strategy: &RadonConfig) -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessConfig::default(),
            radon: strategy.clone(),
        }
    }

    /// Default ChESS-branch strategy. Equivalent to
    /// `DetectionStrategy.fromChess(new ChessConfig())`.
    #[wasm_bindgen(js_name = defaultChess)]
    pub fn default_chess() -> Self {
        Self {
            kind: cell(StrategyKind::Chess),
            chess: ChessConfig::default(),
            radon: RadonConfig::default(),
        }
    }

    /// Default Radon-branch strategy. Equivalent to
    /// `DetectionStrategy.fromRadon(new RadonConfig())`.
    #[wasm_bindgen(js_name = defaultRadon)]
    pub fn default_radon() -> Self {
        Self {
            kind: cell(StrategyKind::Radon),
            chess: ChessConfig::default(),
            radon: RadonConfig::default(),
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
    pub fn chess(&self) -> ChessConfig {
        self.chess.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_chess(&mut self, v: &ChessConfig) {
        // Copy v's contents into this strategy's chess cells so
        // cell-sharing observers see the update.
        self.chess.copy_from(v);
        // Setting a chess config implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Radon branch wrapper. Edits propagate even when the active kind
    /// is ChESS.
    #[wasm_bindgen(getter)]
    pub fn radon(&self) -> RadonConfig {
        self.radon.clone()
    }
    #[wasm_bindgen(setter)]
    pub fn set_radon(&mut self, v: &RadonConfig) {
        self.radon.copy_from(v);
        // Setting a radon config implies that branch is now active.
        *self.kind.borrow_mut() = StrategyKind::Radon;
    }

    /// Switch the discriminant to ChESS without otherwise mutating
    /// the wrapper. The previously stored [`ChessConfig`] cell
    /// becomes the live source again.
    #[wasm_bindgen(js_name = useChess)]
    pub fn use_chess(&mut self) {
        *self.kind.borrow_mut() = StrategyKind::Chess;
    }

    /// Switch the discriminant to Radon without otherwise mutating
    /// the wrapper. The previously stored [`RadonConfig`] cell
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
                chess: ChessConfig::from_value(c),
                radon: RadonConfig::default(),
            },
            RsDetectionStrategy::Radon(r) => Self {
                kind: cell(StrategyKind::Radon),
                chess: ChessConfig::default(),
                radon: RadonConfig::from_value(r),
            },
            _ => Self::default_chess(),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// strategy's cells. Cell-sharing observers see the update.
    fn copy_from(&self, other: &DetectionStrategy) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        self.chess.copy_from(&other.chess);
        self.radon.copy_from(&other.radon);
    }

    pub(crate) fn snapshot(&self) -> RsDetectionStrategy {
        match *self.kind.borrow() {
            StrategyKind::Chess => RsDetectionStrategy::Chess(self.chess.snapshot()),
            StrategyKind::Radon => RsDetectionStrategy::Radon(self.radon.snapshot()),
        }
    }
}

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

    /// Deprecated — use `DetectorConfig.chess()` instead.
    ///
    /// Logs a `console.warn` and returns the same config as [`Self::chess`].
    #[wasm_bindgen(js_name = singleScale)]
    pub fn single_scale() -> Self {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::warn_1(
            &"DetectorConfig.singleScale() is deprecated; use DetectorConfig.chess() instead"
                .into(),
        );
        Self::chess()
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

    /// Return a copy of this config with the Radon refiner replaced.
    ///
    /// Use this instead of the `refiner` key in `withRadon({})`.
    /// JS: `cfg.withRadonRefiner(RadonRefiner.withCenterOfMass(new CenterOfMassConfig()))`.
    #[wasm_bindgen(js_name = withRadonRefiner)]
    pub fn with_radon_refiner(&self, refiner: &RadonRefiner) -> Self {
        let mut out = self.deep_clone();
        if out.strategy.kind() != "radon" {
            out.strategy.use_radon();
        }
        out.strategy.radon().set_refiner(refiner);
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
    ///
    /// To set the refiner use the typed [`Self::with_radon_refiner`] builder
    /// instead.
    ///
    /// Unknown keys throw `Error("unexpected option: '<key>'")`.
    /// JS: `cfg.withRadon({ rayRadius: 6, imageUpsample: 2 })`.
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
                "refiner" => {
                    apply_radon_refiner_from_js(&mut out, val)?;
                }
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
/// `withChessRefiner` / `withRadonRefiner` instead.
fn apply_chess_refiner_from_js(_cfg: &mut DetectorConfig, _val: JsValue) -> Result<(), JsValue> {
    Err(JsValue::from_str(
        "refiner cannot be set via the options object; use .withChessRefiner(refiner) instead",
    ))
}

fn apply_radon_refiner_from_js(_cfg: &mut DetectorConfig, _val: JsValue) -> Result<(), JsValue> {
    Err(JsValue::from_str(
        "refiner cannot be set via the options object; use .withRadonRefiner(refiner) instead",
    ))
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
        ChessRing as RsChessRingCheck, DetectionStrategy as RsDetectionStrategy,
        MultiscaleConfig as RsMultiscaleConfig, UpscaleConfig as RsUpscaleConfig,
    };

    #[test]
    fn leaf_edits_propagate_through_chess_refiner_chain() {
        // Models the JS chain
        //   cfg.strategy.chess.refiner.forstner.maxOffset = 2.5
        // by walking the same getter chain in Rust and mutating the
        // leaf forstner cell. Every intermediate wrapper is a clone
        // that shares the underlying `Rc<RefCell<_>>` cells, so the
        // edit reaches the cfg's chess refiner cell.
        let cfg = DetectorConfig::new();
        // Flip kind to Forstner first (via set_refiner copy-into-cell).
        let mut chess = cfg.strategy().chess();
        chess.set_refiner(&ChessRefiner::forstner(&ForstnerConfig::new()));
        // Edit the leaf forstner cell through the chain.
        cfg.strategy()
            .chess()
            .refiner()
            .get_forstner()
            .set_max_offset(3.5);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        match c.refiner {
            chess_corners::ChessRefiner::Forstner(forstner) => {
                assert!((forstner.max_offset - 3.5).abs() < f32::EPSILON);
            }
            other => panic!("expected forstner refiner, got {other:?}"),
        }
    }

    #[test]
    fn chess_config_field_edits_propagate() {
        let cfg = DetectorConfig::single_scale();
        let mut chess = cfg.strategy().chess();
        chess.set_ring(ChessRing::Broad);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(s) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert_eq!(s.ring, RsChessRingCheck::Broad);
    }

    #[test]
    fn detection_field_edits_propagate() {
        // Shared NMS / clustering knobs live on `detection` and are
        // honoured regardless of the active strategy.
        let cfg = DetectorConfig::single_scale();
        let mut detection = cfg.detection();
        detection.set_nms_radius(7);
        detection.set_min_cluster_size(3);

        let snap = cfg.snapshot();
        assert_eq!(snap.detection.nms_radius, 7);
        assert_eq!(snap.detection.min_cluster_size, 3);
    }

    #[test]
    fn multiscale_pyramid_factory_round_trips_to_snapshot() {
        // The `levels`/`minSize`/`refinementRadius` getters return
        // `Result<_, JsValue>` and panic on host targets because
        // `JsValue` is wasm32-only. So we round-trip through the
        // snapshot, which exercises the same internal storage cells
        // without crossing the wasm-bindgen boundary.
        let mut cfg = DetectorConfig::single_scale();
        cfg.set_multiscale(&MultiscaleConfig::pyramid(4, 64, 5));
        assert_eq!(cfg.multiscale().kind(), "pyramid");

        let snap = cfg.snapshot();
        let RsMultiscaleConfig::Pyramid {
            levels,
            min_size,
            refinement_radius,
        } = snap.multiscale
        else {
            panic!("expected pyramid")
        };
        assert_eq!(levels, 4);
        assert_eq!(min_size, 64);
        assert_eq!(refinement_radius, 5);
    }

    #[test]
    fn multiscale_assigning_wrapper_swaps_to_single_scale() {
        let mut cfg = DetectorConfig::chess_multiscale();
        assert_eq!(cfg.multiscale().kind(), "pyramid");
        cfg.set_multiscale(&MultiscaleConfig::single_scale());
        assert_eq!(cfg.multiscale().kind(), "single_scale");

        let snap = cfg.snapshot();
        assert!(matches!(snap.multiscale, RsMultiscaleConfig::SingleScale));
    }

    #[test]
    fn radon_config_field_edits_propagate() {
        let cfg = DetectorConfig::radon();
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
    fn threshold_numeric_round_trip() {
        let mut cfg = DetectorConfig::new();
        cfg.set_threshold(3.5);
        assert!((cfg.threshold() - 3.5).abs() < f32::EPSILON);
        let snap = cfg.snapshot();
        assert!((snap.threshold - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn cfg_threshold_propagates() {
        let mut cfg = DetectorConfig::new();
        cfg.set_threshold(0.15);
        let snap = cfg.snapshot();
        assert!((snap.threshold - 0.15).abs() < f32::EPSILON);

        cfg.set_threshold(0.25);
        let snap2 = cfg.snapshot();
        assert!((snap2.threshold - 0.25).abs() < f32::EPSILON);
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
    fn upscale_fixed_factor_round_trips() {
        // The `factor` getter/setter returns `Result<_, JsValue>` and
        // panics on host targets; round-trip through the snapshot.
        let mut cfg = DetectorConfig::new();
        cfg.set_upscale(&UpscaleConfig::fixed(3));
        assert_eq!(cfg.upscale().kind(), "fixed");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(3)));
    }

    #[test]
    fn upscale_disabled_round_trips() {
        let cfg = DetectorConfig::new();
        assert_eq!(cfg.upscale().kind(), "disabled");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Disabled));
    }

    #[test]
    fn upscale_assigning_wrapper_swaps_to_fixed() {
        let mut cfg = DetectorConfig::new();
        cfg.set_upscale(&UpscaleConfig::fixed(4));
        assert_eq!(cfg.upscale().kind(), "fixed");
        let snap = cfg.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(4)));
    }

    #[test]
    fn assigning_chess_refiner_copies_into_existing_cells() {
        // `cfg.strategy.chess.refiner = newRefiner` (modelled here as
        // `chess.set_refiner(&new_refiner)`) copies the new refiner's
        // payload contents into the cfg's chess refiner cells, so
        // future `cfg.strategy.chess.refiner.*` getter calls observe
        // the new state without any re-binding.
        let cfg = DetectorConfig::new();
        let mut chess = cfg.strategy().chess();
        let mut wrapped = SaddlePointConfig::new();
        wrapped.set_max_offset(3.5);
        let new_refiner = ChessRefiner::saddle_point(&wrapped);
        chess.set_refiner(&new_refiner);

        let snap = cfg.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        match c.refiner {
            chess_corners::ChessRefiner::SaddlePoint(s) => {
                assert!((s.max_offset - 3.5).abs() < f32::EPSILON);
            }
            other => panic!("expected saddle_point, got {other:?}"),
        }
    }

    #[test]
    fn snapshot_returns_independent_state() {
        let cfg = DetectorConfig::new();
        let mut cfg_mut = cfg;
        cfg_mut.set_threshold(0.1);
        let snap = cfg_mut.snapshot();
        // Replace the threshold after snapshotting — snapshot must not move.
        cfg_mut.set_threshold(0.9);
        assert!((snap.threshold - 0.1).abs() < 1e-6);
    }

    #[test]
    fn orientation_method_round_trips_all_variants() {
        let cases = [
            (OrientationMethod::RingFit, RsOrientationMethod::RingFit),
            (OrientationMethod::DiskFit, RsOrientationMethod::DiskFit),
        ];

        for (wasm_variant, rs_variant) in cases {
            let mut cfg = DetectorConfig::new();
            cfg.set_orientation_method(Some(wasm_variant));
            assert_eq!(cfg.orientation_method(), Some(wasm_variant));
            let snap = cfg.snapshot();
            assert_eq!(
                snap.orientation_method,
                Some(rs_variant),
                "snapshot mismatch for {wasm_variant:?}"
            );
        }
    }

    #[test]
    fn without_orientation_clears_method() {
        let cfg = DetectorConfig::chess().without_orientation();
        assert_eq!(cfg.orientation_method(), None);
        assert_eq!(cfg.snapshot().orientation_method, None);
    }

    #[test]
    fn radon_multiscale_preset_has_radon_strategy_and_multiscale() {
        let cfg = DetectorConfig::radon_multiscale();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Radon(_)),
            "radon_multiscale preset must use Radon strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::Pyramid { .. }),
            "radon_multiscale preset must enable pyramid multiscale"
        );
    }

    #[test]
    fn radon_refiner_center_of_mass_round_trips() {
        let cfg = DetectorConfig::radon();
        let mut radon = cfg.strategy().radon();
        let cm = CenterOfMassConfig::new();
        radon.set_refiner(&RadonRefiner::center_of_mass(&cm));
        let snap = cfg.snapshot();
        let RsDetectionStrategy::Radon(r) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert!(matches!(
            r.refiner,
            chess_corners::RadonRefiner::CenterOfMass(_)
        ));
    }

    // ---- New 0.10.0 API ----

    #[test]
    fn chess_preset_is_single_scale() {
        let cfg = DetectorConfig::chess();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "chess() preset must use Chess strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::SingleScale),
            "chess() preset must be single-scale"
        );
    }

    #[test]
    fn chess_multiscale_preset_has_pyramid() {
        let cfg = DetectorConfig::chess_multiscale();
        let snap = cfg.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "chess_multiscale() preset must use Chess strategy"
        );
        assert!(
            matches!(snap.multiscale, RsMultiscaleConfig::Pyramid { .. }),
            "chess_multiscale() preset must enable pyramid multiscale"
        );
    }

    #[test]
    fn single_scale_delegates_to_chess() {
        // Deprecated shim must return an identical snapshot to chess().
        let snap_compat = DetectorConfig::single_scale().snapshot();
        let snap_new = DetectorConfig::chess().snapshot();
        // Both must be Chess/SingleScale.
        assert!(matches!(
            snap_compat.strategy,
            RsDetectionStrategy::Chess(_)
        ));
        assert!(matches!(
            snap_compat.multiscale,
            RsMultiscaleConfig::SingleScale
        ));
        // Threshold must match.
        assert_eq!(snap_compat.threshold, snap_new.threshold);
    }

    #[test]
    fn pyramid_default_matches_chess_multiscale_preset() {
        let ms = MultiscaleConfig::pyramid_default();
        assert_eq!(ms.kind(), "pyramid");
        // levels / min_size / refinement_radius must equal the facade defaults (3/128/3).
        let snap = ms.snapshot();
        assert!(
            matches!(
                snap,
                RsMultiscaleConfig::Pyramid {
                    levels: 3,
                    min_size: 128,
                    refinement_radius: 3,
                }
            ),
            "pyramid_default must equal (3, 128, 3); got {snap:?}"
        );
    }

    #[test]
    fn with_threshold_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let snap_before = cfg.snapshot();
        let before_val = snap_before.threshold;

        let cfg2 = cfg.with_threshold(0.12);
        let snap2 = cfg2.snapshot();
        assert!((snap2.threshold - 0.12).abs() < f32::EPSILON);

        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!((snap_orig.threshold - before_val).abs() < f32::EPSILON);
    }

    #[test]
    fn with_multiscale_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let ms = MultiscaleConfig::pyramid(4, 64, 5);
        let cfg2 = cfg.with_multiscale(&ms);
        let snap = cfg2.snapshot();
        assert!(
            matches!(
                snap.multiscale,
                RsMultiscaleConfig::Pyramid {
                    levels: 4,
                    min_size: 64,
                    refinement_radius: 5,
                }
            ),
            "with_multiscale must propagate to snapshot; got {snap:?}"
        );

        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!(matches!(
            snap_orig.multiscale,
            RsMultiscaleConfig::SingleScale
        ));
    }

    #[test]
    fn with_upscale_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let up = UpscaleConfig::fixed(2);
        let cfg2 = cfg.with_upscale(&up);
        let snap = cfg2.snapshot();
        assert!(matches!(snap.upscale, RsUpscaleConfig::Fixed(2)));
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!(matches!(snap_orig.upscale, RsUpscaleConfig::Disabled));
    }

    #[test]
    fn with_orientation_method_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let cfg2 = cfg.with_orientation_method(OrientationMethod::DiskFit);
        let snap = cfg2.snapshot();
        assert_eq!(snap.orientation_method, Some(RsOrientationMethod::DiskFit));
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert_eq!(
            snap_orig.orientation_method,
            Some(RsOrientationMethod::RingFit)
        );
    }

    #[test]
    fn with_merge_radius_builder_returns_new_config() {
        let cfg = DetectorConfig::chess();
        let cfg2 = cfg.with_merge_radius(7.5);
        let snap = cfg2.snapshot();
        assert!((snap.merge_radius - 7.5).abs() < f32::EPSILON);
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        assert!((snap_orig.merge_radius - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_chess_refiner_builder_sets_forstner() {
        let cfg = DetectorConfig::chess();
        let mut forstner = ForstnerConfig::new();
        forstner.set_max_offset(4.0);
        let refiner = ChessRefiner::forstner(&forstner);
        let cfg2 = cfg.with_chess_refiner(&refiner);
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert!(
            matches!(c.refiner, chess_corners::ChessRefiner::Forstner(f) if (f.max_offset - 4.0).abs() < f32::EPSILON),
            "with_chess_refiner must set Forstner with max_offset=4.0"
        );
        // Original is unchanged.
        let snap_orig = cfg.snapshot();
        let RsDetectionStrategy::Chess(c_orig) = snap_orig.strategy else {
            panic!("expected chess strategy")
        };
        assert!(
            !matches!(c_orig.refiner, chess_corners::ChessRefiner::Forstner(_)),
            "original config must not have Forstner refiner"
        );
    }

    #[test]
    fn with_radon_refiner_builder_sets_center_of_mass() {
        let cfg = DetectorConfig::radon();
        let cm = CenterOfMassConfig::new();
        let refiner = RadonRefiner::center_of_mass(&cm);
        let cfg2 = cfg.with_radon_refiner(&refiner);
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Radon(r) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert!(matches!(
            r.refiner,
            chess_corners::RadonRefiner::CenterOfMass(_)
        ));
    }

    // The following tests use js_sys::Object / js_sys::Reflect which panic on
    // non-wasm32 targets ("cannot call wasm-bindgen imported functions on
    // non-wasm targets"). They are gated to wasm32 so `cargo test` on the
    // host continues to work. They run under `wasm-pack test` / wasm-bindgen-test.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_ring() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("ring"),
            &JsValue::from_f64(ChessRing::Broad as u8 as f64),
        )
        .unwrap();
        let cfg2 = cfg.with_chess(&opts).expect("with_chess must succeed");
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Chess(c) = snap.strategy else {
            panic!("expected chess strategy")
        };
        assert_eq!(c.ring, chess_corners::ChessRing::Broad);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_detection_opts_set_shared_params() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("nmsRadius"),
            &JsValue::from_f64(7.0),
        )
        .unwrap();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("minClusterSize"),
            &JsValue::from_f64(3.0),
        )
        .unwrap();
        let cfg2 = cfg
            .with_detection(&opts)
            .expect("with_detection must succeed");
        let snap = cfg2.snapshot();
        assert_eq!(snap.detection.nms_radius, 7);
        assert_eq!(snap.detection.min_cluster_size, 3);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_unknown_key_returns_error() {
        let cfg = DetectorConfig::chess();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("unknownField"),
            &JsValue::from_f64(1.0),
        )
        .unwrap();
        let err = cfg
            .with_chess(&opts)
            .expect_err("unknown key must produce an error");
        let msg = err.as_string().unwrap_or_default();
        assert!(
            msg.contains("unexpected option: 'unknownField'"),
            "error message must name the unknown key; got: {msg}"
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_radon_opts_ray_radius_and_image_upsample() {
        let cfg = DetectorConfig::radon();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("rayRadius"),
            &JsValue::from_f64(6.0),
        )
        .unwrap();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("imageUpsample"),
            &JsValue::from_f64(2.0),
        )
        .unwrap();
        let cfg2 = cfg.with_radon(&opts).expect("with_radon must succeed");
        let snap = cfg2.snapshot();
        let RsDetectionStrategy::Radon(r) = snap.strategy else {
            panic!("expected radon strategy")
        };
        assert_eq!(r.ray_radius, 6);
        assert_eq!(r.image_upsample, 2);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_radon_opts_unknown_key_returns_error() {
        let cfg = DetectorConfig::radon();
        let opts = js_sys::Object::new();
        js_sys::Reflect::set(&opts, &JsValue::from_str("badKey"), &JsValue::from_f64(1.0)).unwrap();
        let err = cfg
            .with_radon(&opts)
            .expect_err("unknown key must produce an error");
        let msg = err.as_string().unwrap_or_default();
        assert!(
            msg.contains("unexpected option: 'badKey'"),
            "error message must name the unknown key; got: {msg}"
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn with_chess_opts_on_radon_config_switches_strategy() {
        // Calling with_chess on a radon config must flip strategy to Chess.
        let cfg = DetectorConfig::radon();
        let snap_before = cfg.snapshot();
        assert!(matches!(
            snap_before.strategy,
            RsDetectionStrategy::Radon(_)
        ));

        let opts = js_sys::Object::new();
        js_sys::Reflect::set(
            &opts,
            &JsValue::from_str("ring"),
            &JsValue::from_f64(ChessRing::Broad as u8 as f64),
        )
        .unwrap();
        let cfg2 = cfg.with_chess(&opts).expect("with_chess must succeed");
        let snap = cfg2.snapshot();
        assert!(
            matches!(snap.strategy, RsDetectionStrategy::Chess(_)),
            "strategy must flip to Chess"
        );
    }

    /// Pin the integer discriminants of every `#[wasm_bindgen]` numeric enum.
    ///
    /// wasm-bindgen exposes these values to JavaScript callers as plain
    /// numbers, so reordering or renumbering a variant is a breaking change
    /// for existing JS/TS consumers.  This test catches any accidental
    /// reordering at `cargo test` time, before the WASM package is published.
    #[test]
    fn enum_discriminants_are_pinned() {
        // ChessRing
        assert_eq!(ChessRing::Canonical as u32, 0);
        assert_eq!(ChessRing::Broad as u32, 1);

        // PeakFitMode
        assert_eq!(PeakFitMode::Parabolic as u32, 0);
        assert_eq!(PeakFitMode::Gaussian as u32, 1);

        // OrientationMethod
        assert_eq!(OrientationMethod::RingFit as u32, 0);
        assert_eq!(OrientationMethod::DiskFit as u32, 1);
    }
}
