//! `#[wasm_bindgen]` wrappers for subpixel refiner config types.

use std::rc::Rc;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessRefiner as RsChessRefiner,
    ForstnerConfig as RsForstnerConfig, SaddlePointConfig as RsSaddlePointConfig,
};
use wasm_bindgen::prelude::*;

use super::{cell, Cell};

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
// ChessRefinerKind (internal discriminant, pub(crate) for strategy.rs)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChessRefinerKind {
    CenterOfMass,
    Forstner,
    SaddlePoint,
    #[cfg(feature = "ml-refiner")]
    Ml,
}

impl ChessRefinerKind {
    pub(crate) fn as_str(self) -> &'static str {
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
    pub(crate) kind: Cell<ChessRefinerKind>,
    pub(crate) center_of_mass: Cell<RsCenterOfMassConfig>,
    pub(crate) forstner: Cell<RsForstnerConfig>,
    pub(crate) saddle_point: Cell<RsSaddlePointConfig>,
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
    pub(crate) fn from_value(value: RsChessRefiner) -> Self {
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

    pub(crate) fn snapshot(&self) -> RsChessRefiner {
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
