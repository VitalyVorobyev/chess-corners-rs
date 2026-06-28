//! PyO3 wrapper for `UpscaleConfig`.

use chess_corners::UpscaleConfig as RsUpscaleConfig;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    accept_dict_or_bare_string, config_error, extract_int, json_dumps, json_loads,
    reject_unknown_keys,
};

// ---------------------------------------------------------------------------
// UpscaleConfig (variant-tagged wrapper).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum UpscaleKind {
    Disabled,
    Fixed,
}

impl UpscaleKind {
    fn as_str(&self) -> &'static str {
        match self {
            UpscaleKind::Disabled => "disabled",
            UpscaleKind::Fixed => "fixed",
        }
    }
}

/// Pre-pipeline integer upscaling. One of `disabled()` (no upscaling,
/// the default) or `fixed(factor)` (allowed factors: 2, 3, 4).
#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct UpscaleConfig {
    kind: UpscaleKind,
    factor: u32,
}

impl UpscaleConfig {
    pub(crate) fn from_rs(v: RsUpscaleConfig) -> Self {
        match v {
            RsUpscaleConfig::Disabled => Self {
                kind: UpscaleKind::Disabled,
                factor: 1,
            },
            RsUpscaleConfig::Fixed(k) => Self {
                kind: UpscaleKind::Fixed,
                factor: k,
            },
            _ => Self {
                kind: UpscaleKind::Disabled,
                factor: 1,
            },
        }
    }

    pub(crate) fn to_rs(&self) -> RsUpscaleConfig {
        match self.kind {
            UpscaleKind::Disabled => RsUpscaleConfig::Disabled,
            UpscaleKind::Fixed => RsUpscaleConfig::Fixed(self.factor),
        }
    }
}

#[pymethods]
impl UpscaleConfig {
    /// Default-construct: disabled (no upscaling).
    #[new]
    fn new() -> Self {
        Self::from_rs(RsUpscaleConfig::default())
    }

    /// Disabled variant: no upscaling.
    #[classmethod]
    fn disabled(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_rs(RsUpscaleConfig::Disabled)
    }

    /// Fixed-factor upscale. Allowed factors: 2, 3, 4. Factor is not
    /// validated here; the detector's constructor rejects invalid
    /// values.
    #[classmethod]
    fn fixed(_cls: &Bound<'_, PyType>, factor: u32) -> Self {
        Self {
            kind: UpscaleKind::Fixed,
            factor,
        }
    }

    /// Variant tag: `"disabled"` or `"fixed"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Integer upscale factor; only meaningful when `kind == "fixed"`.
    /// Raises `AttributeError` for the disabled variant.
    #[getter]
    fn factor(&self) -> PyResult<u32> {
        match self.kind {
            UpscaleKind::Fixed => Ok(self.factor),
            UpscaleKind::Disabled => Err(pyo3::exceptions::PyAttributeError::new_err(
                "factor is only available on the fixed variant",
            )),
        }
    }
    #[setter]
    fn set_factor(&mut self, v: u32) -> PyResult<()> {
        match self.kind {
            UpscaleKind::Fixed => {
                self.factor = v;
                Ok(())
            }
            UpscaleKind::Disabled => Err(pyo3::exceptions::PyAttributeError::new_err(
                "factor is only available on the fixed variant",
            )),
        }
    }

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.kind {
            UpscaleKind::Disabled => {
                d.set_item("disabled", py.None())?;
            }
            UpscaleKind::Fixed => {
                d.set_item("fixed", self.factor)?;
            }
        }
        Ok(d.unbind())
    }

    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Accept the externally-tagged shape produced by serde:
        //   "disabled"          (bare string — unit variant)
        //   { "disabled": null } (object form — also accepted)
        //   { "fixed": 2 }      (payload variant — object only)
        if let Some(tag) = accept_dict_or_bare_string(data, &["disabled"], &["fixed"], "upscale")? {
            debug_assert_eq!(tag, "disabled");
            return Ok(Self::from_rs(RsUpscaleConfig::Disabled));
        }
        let dict = data.cast::<PyDict>().cloned().unwrap();
        reject_unknown_keys(&dict, &["disabled", "fixed"], "upscale")?;
        let has_disabled = dict.get_item("disabled")?.is_some();
        let has_fixed = dict.get_item("fixed")?.is_some();
        if has_disabled && has_fixed {
            return Err(config_error(
                "upscale must have exactly one of: disabled, fixed",
            ));
        }
        if has_disabled {
            return Ok(Self::from_rs(RsUpscaleConfig::Disabled));
        }
        if has_fixed {
            if let Some(v) = extract_int(&dict, "fixed", "upscale")? {
                return Ok(Self {
                    kind: UpscaleKind::Fixed,
                    factor: v as u32,
                });
            }
            return Err(config_error("upscale.fixed must be an integer factor"));
        }
        Err(config_error("upscale must have one of: disabled, fixed"))
    }

    #[pyo3(signature = (*, indent=None, sort_keys=true))]
    fn to_json(&self, py: Python<'_>, indent: Option<i64>, sort_keys: bool) -> PyResult<String> {
        let dict = self.to_dict(py)?;
        json_dumps(py, dict.bind(py), indent, sort_keys)
    }

    #[classmethod]
    fn from_json(cls: &Bound<'_, PyType>, py: Python<'_>, text: &str) -> PyResult<Self> {
        let value = json_loads(py, text)
            .map_err(|e| config_error(format!("failed to parse config JSON: {e}")))?;
        Self::from_dict(cls, &value)
    }

    #[pyo3(signature = (*, indent=2, sort_keys=true))]
    fn pretty(&self, py: Python<'_>, indent: i64, sort_keys: bool) -> PyResult<String> {
        self.to_json(py, Some(indent), sort_keys)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.pretty(py, 2, true)
    }
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.pretty(py, 2, true)
    }
}
