//! `#[wasm_bindgen]` wrapper for `UpscaleConfig`.

use chess_corners::UpscaleConfig as RsUpscaleConfig;
use wasm_bindgen::prelude::*;

use super::{cell, Cell};

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
    pub(crate) fn from_value(value: RsUpscaleConfig) -> Self {
        match value {
            RsUpscaleConfig::Disabled => Self::disabled(),
            RsUpscaleConfig::Fixed(k) => Self::fixed(k),
            _ => Self::disabled(),
        }
    }

    /// Copy every cell-backed field's content from `other` into this
    /// config's cells. Cell-sharing observers see the update.
    pub(crate) fn copy_from(&self, other: &UpscaleConfig) {
        *self.kind.borrow_mut() = *other.kind.borrow();
        *self.factor.borrow_mut() = *other.factor.borrow();
    }

    pub(crate) fn snapshot(&self) -> RsUpscaleConfig {
        match *self.kind.borrow() {
            UpscaleKind::Disabled => RsUpscaleConfig::Disabled,
            UpscaleKind::Fixed => RsUpscaleConfig::Fixed(*self.factor.borrow()),
        }
    }
}
