//! PyO3 wrapper for `Uncertainty` and its noise level.

use chess_corners::{NoiseLevel as RsNoiseLevel, Uncertainty as RsUncertainty};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    accept_dict_or_bare_string, config_error, extract_float, reject_unknown_keys, require_dict,
    unknown_variant,
};

/// Which position uncertainty each corner reports, and the image noise
/// level behind it. An immutable value: build one with `off()` (the
/// default: no covariance, no noise estimate) or
/// `propagated(noise_sigma=None)` (a covariance on every corner whose
/// localizer reports one; `noise_sigma` is the known image noise in gray
/// levels, `None` to estimate it from each image).
///
/// Positions are identical under every variant.
#[pyclass(module = "chess_corners", frozen, eq, from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct Uncertainty {
    inner: RsUncertainty,
}

impl Uncertainty {
    /// The wrapper of a facade value; an error for a variant these bindings
    /// do not know.
    pub(crate) fn from_rs(inner: RsUncertainty) -> PyResult<Self> {
        // The facade enums are `#[non_exhaustive]`: the wildcard arm catches
        // variants added to the library after these bindings.
        match inner {
            RsUncertainty::Off
            | RsUncertainty::Propagated {
                noise: RsNoiseLevel::Estimate | RsNoiseLevel::Known(_),
            } => Ok(Self { inner }),
            other => Err(unknown_variant("Uncertainty", &other)),
        }
    }

    pub(crate) fn to_rs(&self) -> RsUncertainty {
        self.inner
    }

    /// The known noise σ, or `None` when the noise level is estimated (or
    /// there is none). Only called on values `from_rs` accepted.
    fn known_sigma(&self) -> Option<f32> {
        match self.inner {
            RsUncertainty::Propagated {
                noise: RsNoiseLevel::Known(sigma),
            } => Some(sigma),
            _ => None,
        }
    }

    /// Parse the noise level of a `propagated` payload: `"estimate"`,
    /// `{"estimate": None}` or `{"known": σ}`.
    fn parse_noise(value: &Bound<'_, PyAny>, path: &str) -> PyResult<RsNoiseLevel> {
        if accept_dict_or_bare_string(value, &["estimate"], &["known"], path)?.is_some() {
            return Ok(RsNoiseLevel::Estimate);
        }
        let dict = require_dict(value, path)?;
        reject_unknown_keys(&dict, &["estimate", "known"], path)?;
        match (dict.get_item("estimate")?, dict.len()) {
            (Some(v), 1) if v.is_none() => return Ok(RsNoiseLevel::Estimate),
            (Some(_), 1) => return Err(config_error(format!("{path}.estimate must be null"))),
            _ => {}
        }
        if dict.len() == 1 {
            if let Some(sigma) = extract_float(&dict, "known", path)? {
                return Ok(RsNoiseLevel::Known(sigma as f32));
            }
        }
        Err(config_error(format!(
            "{path} must have exactly one of: estimate, known"
        )))
    }
}

#[pymethods]
impl Uncertainty {
    /// No position covariance and no noise estimate (the default).
    #[classmethod]
    fn off(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RsUncertainty::Off,
        }
    }

    /// A position covariance on every corner whose localizer reports one.
    /// `noise_sigma` is the standard deviation of the image's pixel noise
    /// in gray levels, rounding included (what `estimate_noise_sigma`
    /// measures); `None` estimates it from each image. A negative or
    /// non-finite value is rejected when the `Detector` is built.
    #[classmethod]
    #[pyo3(signature = (noise_sigma=None))]
    fn propagated(_cls: &Bound<'_, PyType>, noise_sigma: Option<f32>) -> Self {
        let noise = match noise_sigma {
            Some(sigma) => RsNoiseLevel::Known(sigma),
            None => RsNoiseLevel::Estimate,
        };
        Self {
            inner: RsUncertainty::Propagated { noise },
        }
    }

    /// Variant tag: `"off"` or `"propagated"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            RsUncertainty::Off => "off",
            _ => "propagated",
        }
    }

    /// The known image noise σ in gray levels, or `None` when it is
    /// estimated per image (or the variant is `off`).
    #[getter]
    fn noise_sigma(&self) -> Option<f32> {
        self.known_sigma()
    }

    /// The config dict: `{"off": None}`, `{"propagated": {"noise":
    /// "estimate"}}` or `{"propagated": {"noise": {"known": σ}}}`.
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.inner {
            RsUncertainty::Off => d.set_item("off", py.None())?,
            _ => {
                let payload = PyDict::new(py);
                match self.known_sigma() {
                    Some(sigma) => {
                        let known = PyDict::new(py);
                        known.set_item("known", f64::from(sigma))?;
                        payload.set_item("noise", known)?;
                    }
                    None => payload.set_item("noise", "estimate")?,
                }
                d.set_item("propagated", payload)?;
            }
        }
        Ok(d.unbind())
    }

    /// Parse a config value: the bare string `"off"`, `{"off": None}`, or
    /// `{"propagated": {"noise": "estimate" | {"known": σ}}}`. Any other
    /// shape raises `ConfigError`.
    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        const PATH: &str = "uncertainty";
        if accept_dict_or_bare_string(data, &["off"], &["propagated"], PATH)?.is_some() {
            return Ok(Self {
                inner: RsUncertainty::Off,
            });
        }
        let dict = require_dict(data, PATH)?;
        reject_unknown_keys(&dict, &["off", "propagated"], PATH)?;
        if dict.len() != 1 {
            return Err(config_error(format!(
                "{PATH} must have exactly one of: off, propagated"
            )));
        }
        if let Some(value) = dict.get_item("off")? {
            if !value.is_none() {
                return Err(config_error(format!("{PATH}.off must be null")));
            }
            return Ok(Self {
                inner: RsUncertainty::Off,
            });
        }
        let path = format!("{PATH}.propagated");
        let payload = dict
            .get_item("propagated")?
            .ok_or_else(|| config_error(format!("{PATH} must have one of: off, propagated")))?;
        let payload = require_dict(&payload, &path)?;
        reject_unknown_keys(&payload, &["noise"], &path)?;
        let noise = payload
            .get_item("noise")?
            .ok_or_else(|| config_error(format!("{path}.noise is required")))?;
        let noise = Self::parse_noise(&noise, &format!("{path}.noise"))?;
        Ok(Self {
            inner: RsUncertainty::Propagated { noise },
        })
    }

    fn __repr__(&self) -> String {
        match (self.inner, self.known_sigma()) {
            (RsUncertainty::Off, _) => "Uncertainty.off()".to_string(),
            (_, Some(sigma)) => format!("Uncertainty.propagated(noise_sigma={sigma})"),
            (_, None) => "Uncertainty.propagated()".to_string(),
        }
    }
}
