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
/// `ImageRefiner.saddle_point(cfg=None)`.
///
/// On the ChESS detector it replaces the center of mass, starts from the
/// integer peak and drops corners it rejects. On the Radon detector it
/// starts from the peak fit and keeps the peak fit where it does not
/// converge. The saddle-point refiner reports a position covariance.
#[pyclass(module = "chess_corners", frozen, eq, from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRefiner {
    /// The saddle-point tuning: the only image refiner.
    inner: RsSaddlePointConfig,
}

impl ImageRefiner {
    /// The wrapper of a facade value; an error for a variant these bindings
    /// do not know.
    pub(crate) fn from_rs(inner: RsImageRefiner) -> PyResult<Self> {
        // The facade enum is `#[non_exhaustive]`: the wildcard arm catches
        // variants added to the library after these bindings.
        match inner {
            RsImageRefiner::SaddlePoint(cfg) => Ok(Self { inner: cfg }),
            other => Err(unknown_variant("ImageRefiner", &other)),
        }
    }

    pub(crate) fn to_rs(&self) -> RsImageRefiner {
        RsImageRefiner::SaddlePoint(self.inner)
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
        Self { inner }
    }

    /// Variant tag: `"saddle_point"`.
    #[getter]
    fn kind(&self) -> &'static str {
        "saddle_point"
    }

    /// Tuning payload: a fresh `SaddlePointConfig` copy. Modifying the
    /// returned object does not affect the refiner; rebuild via
    /// `ImageRefiner.saddle_point(cfg)` to apply it.
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, SaddlePointConfig { inner: self.inner })?.into_any())
    }

    /// The config dict: `{"saddle_point": {...}}`.
    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item(
            "saddle_point",
            SaddlePointConfig { inner: self.inner }.to_dict(py)?,
        )?;
        Ok(d.unbind())
    }

    /// Parse a config value: `{"saddle_point": {...}}`. Any other shape
    /// raises `ConfigError`.
    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        const PATH: &str = "refiner";
        const ALLOWED: &[&str] = &["saddle_point"];

        // No variant is a bare string: this only rejects strings, naming the
        // payload a bare `"saddle_point"` lacks.
        accept_dict_or_bare_string(data, &[], ALLOWED, PATH)?;
        let dict = require_dict(data, PATH)?;
        reject_unknown_keys(&dict, ALLOWED, PATH)?;
        if dict.len() != 1 {
            return Err(config_error(format!(
                "{PATH} must have exactly one of: {}",
                ALLOWED.join(", ")
            )));
        }
        let Some(value) = dict.get_item("saddle_point")? else {
            return Err(config_error(format!(
                "{PATH} must have one of: {}",
                ALLOWED.join(", ")
            )));
        };
        let cls = data.py().get_type::<SaddlePointConfig>();
        let cfg = SaddlePointConfig::from_dict(&cls, &value)?;
        Ok(Self { inner: cfg.inner })
    }

    fn __repr__(&self) -> String {
        let c = self.inner;
        format!(
            "ImageRefiner.saddle_point(radius={}, det_margin={}, max_offset={}, min_abs_det={})",
            c.radius, c.det_margin, c.max_offset, c.min_abs_det
        )
    }
}
