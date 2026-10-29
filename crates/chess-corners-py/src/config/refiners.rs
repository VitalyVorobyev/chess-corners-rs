//! PyO3 wrappers for subpixel refiner config types.

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ImageRefiner as RsImageRefiner,
    SaddlePointConfig as RsSaddlePointConfig,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    accept_dict_or_bare_string, config_error, extract_float, extract_int, json_dumps, json_loads,
    reject_unknown_keys, require_dict, unknown_variant,
};

// ---------------------------------------------------------------------------
// CenterOfMassConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
#[derive(Clone, Debug)]
pub struct CenterOfMassConfig {
    pub(crate) inner: RsCenterOfMassConfig,
}

#[pymethods]
impl CenterOfMassConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsCenterOfMassConfig::default(),
        }
    }

    #[getter]
    fn radius(&self) -> i32 {
        self.inner.radius
    }
    #[setter]
    fn set_radius(&mut self, v: i32) {
        self.inner.radius = v;
    }

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("radius", self.inner.radius)?;
        Ok(d.unbind())
    }

    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "center_of_mass")?;
        reject_unknown_keys(&dict, &["radius"], "center_of_mass")?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "radius", "center_of_mass")? {
            cfg.inner.radius = v as i32;
        }
        Ok(cfg)
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

// ---------------------------------------------------------------------------
// SaddlePointConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
#[derive(Clone, Debug)]
pub struct SaddlePointConfig {
    pub(crate) inner: RsSaddlePointConfig,
}

#[pymethods]
impl SaddlePointConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsSaddlePointConfig::default(),
        }
    }

    #[getter]
    fn radius(&self) -> i32 {
        self.inner.radius
    }
    #[setter]
    fn set_radius(&mut self, v: i32) {
        self.inner.radius = v;
    }

    #[getter]
    fn det_margin(&self) -> f32 {
        self.inner.det_margin
    }
    #[setter]
    fn set_det_margin(&mut self, v: f32) {
        self.inner.det_margin = v;
    }

    #[getter]
    fn max_offset(&self) -> f32 {
        self.inner.max_offset
    }
    #[setter]
    fn set_max_offset(&mut self, v: f32) {
        self.inner.max_offset = v;
    }

    #[getter]
    fn min_abs_det(&self) -> f32 {
        self.inner.min_abs_det
    }
    #[setter]
    fn set_min_abs_det(&mut self, v: f32) {
        self.inner.min_abs_det = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("radius", self.inner.radius)?;
        d.set_item("det_margin", self.inner.det_margin as f64)?;
        d.set_item("max_offset", self.inner.max_offset as f64)?;
        d.set_item("min_abs_det", self.inner.min_abs_det as f64)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "saddle_point")?;
        reject_unknown_keys(
            &dict,
            &["radius", "det_margin", "max_offset", "min_abs_det"],
            "saddle_point",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "radius", "saddle_point")? {
            cfg.inner.radius = v as i32;
        }
        if let Some(v) = extract_float(&dict, "det_margin", "saddle_point")? {
            cfg.inner.det_margin = v as f32;
        }
        if let Some(v) = extract_float(&dict, "max_offset", "saddle_point")? {
            cfg.inner.max_offset = v as f32;
        }
        if let Some(v) = extract_float(&dict, "min_abs_det", "saddle_point")? {
            cfg.inner.min_abs_det = v as f32;
        }
        Ok(cfg)
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

// ---------------------------------------------------------------------------
// ImageRefiner (frozen value class).
// ---------------------------------------------------------------------------

/// Subpixel refinement of each corner on the input image, replacing the
/// detector's own localizer. An immutable value: build one with
/// `ImageRefiner.saddle_point(cfg=None)` or `ImageRefiner.ml()`.
///
/// On the ChESS detector it replaces the center of mass, starts from the
/// integer peak and drops corners it rejects. On the Radon detector it
/// starts from the peak fit and keeps the peak fit where it does not
/// converge. The saddle-point refiner reports a position covariance. The
/// ML refiner works with both detectors, rejects no corner (one it cannot
/// refine keeps its seed), reports no covariance and is not supported by
/// `detect_roi`.
#[pyclass(module = "chess_corners", frozen, eq, from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRefiner {
    inner: RsImageRefiner,
}

impl ImageRefiner {
    /// The wrapper of a facade value; an error for a variant these bindings
    /// do not know.
    pub(crate) fn from_rs(inner: RsImageRefiner) -> PyResult<Self> {
        // The facade enum is `#[non_exhaustive]`: the wildcard arm catches
        // variants added to the library after these bindings.
        match inner {
            RsImageRefiner::SaddlePoint(_) => Ok(Self { inner }),
            #[cfg(feature = "ml-refiner")]
            RsImageRefiner::Ml => Ok(Self { inner }),
            other => Err(unknown_variant("ImageRefiner", &other)),
        }
    }

    pub(crate) fn to_rs(&self) -> RsImageRefiner {
        self.inner
    }
}

#[pymethods]
impl ImageRefiner {
    /// Quadratic surface fit at the saddle point of the image patch.
    #[classmethod]
    #[pyo3(signature = (cfg=None))]
    fn saddle_point(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        cfg: Option<Py<SaddlePointConfig>>,
    ) -> Self {
        let inner = cfg.map(|c| c.borrow(py).inner).unwrap_or_default();
        Self {
            inner: RsImageRefiner::SaddlePoint(inner),
        }
    }

    /// ML-backed subpixel refinement. Requires the `ml-refiner` feature.
    #[cfg(feature = "ml-refiner")]
    #[classmethod]
    fn ml(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RsImageRefiner::Ml,
        }
    }

    /// Variant tag: `"saddle_point"` or `"ml"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            RsImageRefiner::SaddlePoint(_) => "saddle_point",
            _ => "ml",
        }
    }

    /// Tuning payload: a fresh `SaddlePointConfig` copy for `saddle_point`,
    /// `None` for `ml`. Modifying the returned object does not affect the
    /// refiner; rebuild via `ImageRefiner.saddle_point(cfg)` to apply it.
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner {
            RsImageRefiner::SaddlePoint(inner) => {
                Ok(Py::new(py, SaddlePointConfig { inner })?.into_any())
            }
            _ => Ok(py.None()),
        }
    }

    /// The config dict: `{"saddle_point": {...}}` or `{"ml": None}`.
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.inner {
            RsImageRefiner::SaddlePoint(inner) => {
                d.set_item("saddle_point", SaddlePointConfig { inner }.to_dict(py)?)?;
            }
            _ => d.set_item("ml", py.None())?,
        }
        Ok(d.unbind())
    }

    /// Parse a config value: `{"saddle_point": {...}}`, and with the
    /// `ml-refiner` feature `"ml"` or `{"ml": None}`. Any other shape
    /// raises `ConfigError`.
    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        const PATH: &str = "refiner";
        #[cfg(feature = "ml-refiner")]
        const UNIT_TAGS: &[&str] = &["ml"];
        #[cfg(not(feature = "ml-refiner"))]
        const UNIT_TAGS: &[&str] = &[];
        #[cfg(feature = "ml-refiner")]
        const ALLOWED: &[&str] = &["saddle_point", "ml"];
        #[cfg(not(feature = "ml-refiner"))]
        const ALLOWED: &[&str] = &["saddle_point"];

        if accept_dict_or_bare_string(data, UNIT_TAGS, &["saddle_point"], PATH)?.is_some() {
            #[cfg(feature = "ml-refiner")]
            return Ok(Self {
                inner: RsImageRefiner::Ml,
            });
            #[cfg(not(feature = "ml-refiner"))]
            return Err(config_error(format!("{PATH}: no unit variants available")));
        }
        let dict = require_dict(data, PATH)?;
        reject_unknown_keys(&dict, ALLOWED, PATH)?;
        if dict.len() != 1 {
            return Err(config_error(format!(
                "{PATH} must have exactly one of: {}",
                ALLOWED.join(", ")
            )));
        }
        if let Some(value) = dict.get_item("saddle_point")? {
            let cls = data.py().get_type::<SaddlePointConfig>();
            let cfg = SaddlePointConfig::from_dict(&cls, &value)?;
            return Ok(Self {
                inner: RsImageRefiner::SaddlePoint(cfg.inner),
            });
        }
        #[cfg(feature = "ml-refiner")]
        if let Some(value) = dict.get_item("ml")? {
            if !value.is_none() {
                return Err(config_error(format!("{PATH}.ml must be null")));
            }
            return Ok(Self {
                inner: RsImageRefiner::Ml,
            });
        }
        Err(config_error(format!(
            "{PATH} must have one of: {}",
            ALLOWED.join(", ")
        )))
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RsImageRefiner::SaddlePoint(c) => format!(
                "ImageRefiner.saddle_point(radius={}, det_margin={}, max_offset={}, min_abs_det={})",
                c.radius, c.det_margin, c.max_offset, c.min_abs_det
            ),
            _ => "ImageRefiner.ml()".to_string(),
        }
    }
}
