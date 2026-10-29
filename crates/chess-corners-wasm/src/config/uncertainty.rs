//! `#[wasm_bindgen]` wrapper for `Uncertainty` and its noise level.

use chess_corners::{NoiseLevel as RsNoiseLevel, Uncertainty as RsUncertainty};
use wasm_bindgen::prelude::*;

/// Which position uncertainty each corner reports, and the image noise
/// level behind it. Mirrors [`chess_corners::Uncertainty`]. An immutable
/// value: build one with `Uncertainty.off()` or
/// `Uncertainty.propagated(sigma?)`.
///
/// `off()` *(default)* — no position covariance and no noise estimate;
/// all covariance values are `NaN`.
///
/// `propagated(sigma?)` — every corner whose localizer supports it reports
/// a position covariance: image noise propagated through the localizer
/// plus its systematic error, including a per-localizer floor measured on
/// real images that usually dominates it there (weighting by it has not
/// been shown to beat uniform weighting on real data). `sigma` is the
/// known image noise in gray levels (rounding included, what
/// `estimateNoiseSigma` measures); omit it to estimate the noise from each
/// image. A negative or non-finite `sigma` is rejected when the detector is
/// built.
///
/// Positions are identical under every variant. JSON: `"off"`,
/// `{"propagated": {"noise": "estimate"}}` or
/// `{"propagated": {"noise": {"known": 1.2}}}`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uncertainty {
    inner: RsUncertainty,
}

#[wasm_bindgen]
impl Uncertainty {
    /// No position covariance and no noise estimate (the default).
    pub fn off() -> Self {
        Self {
            inner: RsUncertainty::Off,
        }
    }

    /// A position covariance on every corner whose localizer reports one,
    /// with the known image noise `sigma` in gray levels, or the per-image
    /// estimate when `sigma` is omitted.
    pub fn propagated(sigma: Option<f32>) -> Self {
        let noise = match sigma {
            Some(sigma) => RsNoiseLevel::Known(sigma),
            None => RsNoiseLevel::Estimate,
        };
        Self {
            inner: RsUncertainty::Propagated { noise },
        }
    }

    /// Variant tag: `"off"` or `"propagated"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        match self.inner {
            RsUncertainty::Off => "off",
            _ => "propagated",
        }
        .into()
    }

    /// The known noise σ in gray levels, or `undefined` when it is
    /// estimated per image (or the variant is `off`).
    #[wasm_bindgen(getter, js_name = noiseSigma)]
    pub fn noise_sigma(&self) -> Option<f32> {
        match self.inner {
            RsUncertainty::Propagated {
                noise: RsNoiseLevel::Known(sigma),
            } => Some(sigma),
            _ => None,
        }
    }
}

impl Default for Uncertainty {
    fn default() -> Self {
        Self::off()
    }
}

impl Uncertainty {
    pub(crate) fn snapshot(self) -> RsUncertainty {
        self.inner
    }
}

impl TryFrom<RsUncertainty> for Uncertainty {
    type Error = JsError;

    /// A variant these bindings do not know is an error, never mapped
    /// silently to another one.
    fn try_from(inner: RsUncertainty) -> Result<Self, Self::Error> {
        match inner {
            RsUncertainty::Off
            | RsUncertainty::Propagated {
                noise: RsNoiseLevel::Estimate | RsNoiseLevel::Known(_),
            } => Ok(Self { inner }),
            other => Err(JsError::new(&format!("unsupported uncertainty {other:?}"))),
        }
    }
}
