//! `#[wasm_bindgen]` wrapper for `MultiscaleConfig`.

use chess_corners::MultiscaleConfig as RsMultiscaleConfig;
use wasm_bindgen::prelude::*;

use super::{cell, Cell};

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
    pub(crate) fn from_value(value: RsMultiscaleConfig) -> Self {
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
    pub(crate) fn copy_from(&self, other: &MultiscaleConfig) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        *self.levels.borrow_mut() = *other.levels.borrow();
        *self.min_size.borrow_mut() = *other.min_size.borrow();
        *self.refinement_radius.borrow_mut() = *other.refinement_radius.borrow();
    }

    pub(crate) fn snapshot(&self) -> RsMultiscaleConfig {
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
