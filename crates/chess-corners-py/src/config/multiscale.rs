//! PyO3 wrapper for `MultiscaleConfig`.

use chess_corners::MultiscaleConfig as RsMultiscaleConfig;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    accept_dict_or_bare_string, config_error, extract_int, json_dumps, json_loads,
    reject_unknown_keys, require_dict,
};

// ---------------------------------------------------------------------------
// MultiscaleConfig (variant-tagged wrapper).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum MultiscaleKind {
    SingleScale,
    Pyramid,
}

impl MultiscaleKind {
    fn as_str(&self) -> &'static str {
        match self {
            MultiscaleKind::SingleScale => "single_scale",
            MultiscaleKind::Pyramid => "pyramid",
        }
    }
}

/// Coarse-to-fine multiscale configuration. One of
/// `single_scale()` (no pyramid) or `pyramid(levels=, min_size=, refinement_radius=)`.
/// Both detectors honour the same enum.
#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct MultiscaleConfig {
    kind: MultiscaleKind,
    levels: u8,
    min_size: usize,
    refinement_radius: u32,
}

impl MultiscaleConfig {
    pub(crate) fn from_rs(v: RsMultiscaleConfig) -> Self {
        match v {
            RsMultiscaleConfig::SingleScale => Self {
                kind: MultiscaleKind::SingleScale,
                levels: 3,
                min_size: 128,
                refinement_radius: 3,
            },
            RsMultiscaleConfig::Pyramid {
                levels,
                min_size,
                refinement_radius,
            } => Self {
                kind: MultiscaleKind::Pyramid,
                levels,
                min_size,
                refinement_radius,
            },
            _ => Self {
                kind: MultiscaleKind::SingleScale,
                levels: 3,
                min_size: 128,
                refinement_radius: 3,
            },
        }
    }

    pub(crate) fn to_rs(&self) -> RsMultiscaleConfig {
        match self.kind {
            MultiscaleKind::SingleScale => RsMultiscaleConfig::SingleScale,
            MultiscaleKind::Pyramid => RsMultiscaleConfig::Pyramid {
                levels: self.levels,
                min_size: self.min_size,
                refinement_radius: self.refinement_radius,
            },
        }
    }
}

#[pymethods]
impl MultiscaleConfig {
    /// Default-construct: single-scale (no pyramid).
    #[new]
    fn new() -> Self {
        Self::from_rs(RsMultiscaleConfig::default())
    }

    /// Single-scale variant: run the detector once on the full image.
    #[classmethod]
    fn single_scale(_cls: &Bound<'_, PyType>) -> Self {
        Self::from_rs(RsMultiscaleConfig::SingleScale)
    }

    /// Pyramid variant with library-default parameters (levels=3, min_size=128,
    /// refinement_radius=3). Use `pyramid(levels=..., ...)` for custom settings.
    #[classmethod]
    fn pyramid_default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            kind: MultiscaleKind::Pyramid,
            levels: 3,
            min_size: 128,
            refinement_radius: 3,
        }
    }

    /// Pyramid variant: build an image pyramid and refine coarse seeds
    /// into the base image.
    #[classmethod]
    #[pyo3(signature = (*, levels=3, min_size=128, refinement_radius=3))]
    fn pyramid(
        _cls: &Bound<'_, PyType>,
        levels: u8,
        min_size: usize,
        refinement_radius: u32,
    ) -> Self {
        Self {
            kind: MultiscaleKind::Pyramid,
            levels,
            min_size,
            refinement_radius,
        }
    }

    /// Variant tag: `"single_scale"` or `"pyramid"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Pyramid levels; only meaningful when `kind == "pyramid"`.
    /// Raises `AttributeError` for the single-scale variant.
    #[getter]
    fn levels(&self) -> PyResult<u8> {
        match self.kind {
            MultiscaleKind::Pyramid => Ok(self.levels),
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "levels is only available on the pyramid variant",
            )),
        }
    }
    #[setter]
    fn set_levels(&mut self, v: u8) -> PyResult<()> {
        match self.kind {
            MultiscaleKind::Pyramid => {
                self.levels = v;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "levels is only available on the pyramid variant",
            )),
        }
    }

    /// Pyramid minimum short-edge length, in pixels. Only meaningful
    /// when `kind == "pyramid"`.
    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        match self.kind {
            MultiscaleKind::Pyramid => Ok(self.min_size),
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "min_size is only available on the pyramid variant",
            )),
        }
    }
    #[setter]
    fn set_min_size(&mut self, v: usize) -> PyResult<()> {
        match self.kind {
            MultiscaleKind::Pyramid => {
                self.min_size = v;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "min_size is only available on the pyramid variant",
            )),
        }
    }

    /// Coarse-level ROI half-radius used to refine each seed into the
    /// base image. Only meaningful when `kind == "pyramid"`.
    #[getter]
    fn refinement_radius(&self) -> PyResult<u32> {
        match self.kind {
            MultiscaleKind::Pyramid => Ok(self.refinement_radius),
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "refinement_radius is only available on the pyramid variant",
            )),
        }
    }
    #[setter]
    fn set_refinement_radius(&mut self, v: u32) -> PyResult<()> {
        match self.kind {
            MultiscaleKind::Pyramid => {
                self.refinement_radius = v;
                Ok(())
            }
            MultiscaleKind::SingleScale => Err(pyo3::exceptions::PyAttributeError::new_err(
                "refinement_radius is only available on the pyramid variant",
            )),
        }
    }

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.kind {
            MultiscaleKind::SingleScale => {
                d.set_item("single_scale", py.None())?;
            }
            MultiscaleKind::Pyramid => {
                let payload = PyDict::new(py);
                payload.set_item("levels", self.levels)?;
                payload.set_item("min_size", self.min_size)?;
                payload.set_item("refinement_radius", self.refinement_radius)?;
                d.set_item("pyramid", payload)?;
            }
        }
        Ok(d.unbind())
    }

    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Accept the externally-tagged shape produced by serde:
        //   "single_scale"          (bare string — unit variant)
        //   { "single_scale": null } (object form — also accepted)
        //   { "pyramid": { ... } }  (payload variant — object only)
        if let Some(tag) =
            accept_dict_or_bare_string(data, &["single_scale"], &["pyramid"], "multiscale")?
        {
            debug_assert_eq!(tag, "single_scale");
            return Ok(Self::from_rs(RsMultiscaleConfig::SingleScale));
        }
        let dict = data.cast::<PyDict>().cloned().unwrap();
        reject_unknown_keys(&dict, &["single_scale", "pyramid"], "multiscale")?;
        let has_single = dict.get_item("single_scale")?.is_some();
        let has_pyramid = dict.get_item("pyramid")?.is_some();
        if has_single && has_pyramid {
            return Err(config_error(
                "multiscale must have exactly one of: single_scale, pyramid",
            ));
        }
        if has_single {
            return Ok(Self::from_rs(RsMultiscaleConfig::SingleScale));
        }
        if has_pyramid {
            let value = dict
                .get_item("pyramid")?
                .ok_or_else(|| config_error("multiscale.pyramid missing"))?;
            if value.is_none() {
                return Err(config_error("multiscale.pyramid payload must be an object"));
            }
            let pdict = require_dict(&value, "multiscale.pyramid")?;
            reject_unknown_keys(
                &pdict,
                &["levels", "min_size", "refinement_radius"],
                "multiscale.pyramid",
            )?;
            let mut cfg = Self {
                kind: MultiscaleKind::Pyramid,
                levels: 3,
                min_size: 128,
                refinement_radius: 3,
            };
            if let Some(v) = extract_int(&pdict, "levels", "multiscale.pyramid")? {
                cfg.levels = v as u8;
            }
            if let Some(v) = extract_int(&pdict, "min_size", "multiscale.pyramid")? {
                cfg.min_size = v as usize;
            }
            if let Some(v) = extract_int(&pdict, "refinement_radius", "multiscale.pyramid")? {
                cfg.refinement_radius = v as u32;
            }
            return Ok(cfg);
        }
        Err(config_error(
            "multiscale must have one of: single_scale, pyramid",
        ))
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
