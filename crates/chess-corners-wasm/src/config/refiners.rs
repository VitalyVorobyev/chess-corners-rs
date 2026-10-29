//! `#[wasm_bindgen]` wrappers for the subpixel refiner config types.

use std::rc::Rc;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ImageRefiner as RsImageRefiner,
    SaddlePointConfig as RsSaddlePointConfig,
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
    pub(crate) fn from_cell(cell: Cell<RsSaddlePointConfig>) -> Self {
        Self { cell }
    }
}

// ---------------------------------------------------------------------------
// ImageRefiner (immutable value class)
// ---------------------------------------------------------------------------

/// The optional image-refinement stage. Mirrors
/// [`chess_corners::ImageRefiner`]. An immutable value: build one with
/// `ImageRefiner.saddlePoint()` or `ImageRefiner.saddlePointWith(cfg)` and
/// pass it to `DetectorConfig.withRefiner`.
///
/// Without an image refiner each detector keeps its own subpixel localizer
/// (ChESS: center of mass; Radon: peak fit). An image refiner reads only the
/// input image around each corner, so it works with both detectors. On ChESS
/// it replaces the center of mass, starts from the integer response peak and
/// drops the corners it rejects. On Radon it starts from the peak fit and
/// keeps the peak fit where it does not converge.
///
/// `saddlePoint` fits a quadratic surface to the image patch at the saddle
/// point and re-centres it until it converges; it reports a position
/// covariance on the corners it refines.
///
/// JSON: `{"saddle_point": {...}}`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRefiner {
    /// The saddle-point tuning: the only image refiner.
    inner: RsSaddlePointConfig,
}

#[wasm_bindgen]
impl ImageRefiner {
    /// Saddle-point refinement with default tuning.
    #[wasm_bindgen(js_name = saddlePoint)]
    pub fn saddle_point() -> Self {
        Self {
            inner: RsSaddlePointConfig::default(),
        }
    }

    /// Saddle-point refinement tuned by `cfg`. The payload is copied: later
    /// edits to `cfg` do not affect this value.
    #[wasm_bindgen(js_name = saddlePointWith)]
    pub fn saddle_point_with(cfg: &SaddlePointConfig) -> Self {
        Self {
            inner: *cfg.cell.borrow(),
        }
    }

    /// Variant tag: `"saddle_point"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        "saddle_point".into()
    }

    /// A copy of the saddle-point payload (named `saddlePointConfig` because
    /// `saddlePoint` is the factory).
    #[wasm_bindgen(getter, js_name = saddlePointConfig)]
    pub fn get_saddle_point(&self) -> SaddlePointConfig {
        SaddlePointConfig::from_cell(cell(self.inner))
    }
}

impl ImageRefiner {
    pub(crate) fn snapshot(self) -> RsImageRefiner {
        RsImageRefiner::SaddlePoint(self.inner)
    }
}

impl TryFrom<RsImageRefiner> for ImageRefiner {
    type Error = JsError;

    /// A variant these bindings do not know is an error, never mapped
    /// silently to another one.
    fn try_from(inner: RsImageRefiner) -> Result<Self, Self::Error> {
        match inner {
            RsImageRefiner::SaddlePoint(cfg) => Ok(Self { inner: cfg }),
            other => Err(JsError::new(&format!(
                "unsupported image refiner {other:?}"
            ))),
        }
    }
}
