//! PyO3 wrappers for subpixel refiner config types.

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessRefiner as RsChessRefiner,
    ForstnerConfig as RsForstnerConfig, SaddlePointConfig as RsSaddlePointConfig,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    config_error, extract_float, extract_int, json_dumps, json_loads, reject_unknown_keys,
    require_dict,
};

// ---------------------------------------------------------------------------
// CenterOfMassConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("radius", self.inner.radius)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
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
// ForstnerConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ForstnerConfig {
    pub(crate) inner: RsForstnerConfig,
}

#[pymethods]
impl ForstnerConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsForstnerConfig::default(),
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
    fn min_trace(&self) -> f32 {
        self.inner.min_trace
    }
    #[setter]
    fn set_min_trace(&mut self, v: f32) {
        self.inner.min_trace = v;
    }

    #[getter]
    fn min_det(&self) -> f32 {
        self.inner.min_det
    }
    #[setter]
    fn set_min_det(&mut self, v: f32) {
        self.inner.min_det = v;
    }

    #[getter]
    fn max_condition_number(&self) -> f32 {
        self.inner.max_condition_number
    }
    #[setter]
    fn set_max_condition_number(&mut self, v: f32) {
        self.inner.max_condition_number = v;
    }

    #[getter]
    fn max_offset(&self) -> f32 {
        self.inner.max_offset
    }
    #[setter]
    fn set_max_offset(&mut self, v: f32) {
        self.inner.max_offset = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("radius", self.inner.radius)?;
        d.set_item("min_trace", self.inner.min_trace as f64)?;
        d.set_item("min_det", self.inner.min_det as f64)?;
        d.set_item(
            "max_condition_number",
            self.inner.max_condition_number as f64,
        )?;
        d.set_item("max_offset", self.inner.max_offset as f64)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "forstner")?;
        reject_unknown_keys(
            &dict,
            &[
                "radius",
                "min_trace",
                "min_det",
                "max_condition_number",
                "max_offset",
            ],
            "forstner",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "radius", "forstner")? {
            cfg.inner.radius = v as i32;
        }
        if let Some(v) = extract_float(&dict, "min_trace", "forstner")? {
            cfg.inner.min_trace = v as f32;
        }
        if let Some(v) = extract_float(&dict, "min_det", "forstner")? {
            cfg.inner.min_det = v as f32;
        }
        if let Some(v) = extract_float(&dict, "max_condition_number", "forstner")? {
            cfg.inner.max_condition_number = v as f32;
        }
        if let Some(v) = extract_float(&dict, "max_offset", "forstner")? {
            cfg.inner.max_offset = v as f32;
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

#[pyclass(module = "chess_corners", skip_from_py_object)]
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
// ChessRefiner (variant-tagged wrapper).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum ChessRefinerKind {
    CenterOfMass,
    Forstner,
    SaddlePoint,
    #[cfg(feature = "ml-refiner")]
    Ml,
}

impl ChessRefinerKind {
    fn as_str(&self) -> &'static str {
        match self {
            ChessRefinerKind::CenterOfMass => "center_of_mass",
            ChessRefinerKind::Forstner => "forstner",
            ChessRefinerKind::SaddlePoint => "saddle_point",
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => "ml",
        }
    }
}

/// Subpixel refiner selection for the ChESS detector. Build via one
/// of `ChessRefiner.center_of_mass(...)`, `.forstner(...)`,
/// `.saddle_point(...)`, or `.ml()` (requires the `ml-refiner`
/// feature). Each variant carries its own tuning struct as a
/// payload, so switching variants cannot leave a stale field behind.
#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ChessRefiner {
    kind: ChessRefinerKind,
    center_of_mass: RsCenterOfMassConfig,
    forstner: RsForstnerConfig,
    saddle_point: RsSaddlePointConfig,
}

impl ChessRefiner {
    pub(crate) fn from_rs(v: RsChessRefiner) -> Self {
        let mut cfg = Self {
            kind: ChessRefinerKind::CenterOfMass,
            center_of_mass: RsCenterOfMassConfig::default(),
            forstner: RsForstnerConfig::default(),
            saddle_point: RsSaddlePointConfig::default(),
        };
        match v {
            RsChessRefiner::CenterOfMass(c) => {
                cfg.kind = ChessRefinerKind::CenterOfMass;
                cfg.center_of_mass = c;
            }
            RsChessRefiner::Forstner(c) => {
                cfg.kind = ChessRefinerKind::Forstner;
                cfg.forstner = c;
            }
            RsChessRefiner::SaddlePoint(c) => {
                cfg.kind = ChessRefinerKind::SaddlePoint;
                cfg.saddle_point = c;
            }
            #[cfg(feature = "ml-refiner")]
            RsChessRefiner::Ml => {
                cfg.kind = ChessRefinerKind::Ml;
            }
            _ => {
                cfg.kind = ChessRefinerKind::CenterOfMass;
            }
        }
        cfg
    }

    pub(crate) fn to_rs(&self) -> RsChessRefiner {
        match self.kind {
            ChessRefinerKind::CenterOfMass => RsChessRefiner::CenterOfMass(self.center_of_mass),
            ChessRefinerKind::Forstner => RsChessRefiner::Forstner(self.forstner),
            ChessRefinerKind::SaddlePoint => RsChessRefiner::SaddlePoint(self.saddle_point),
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => RsChessRefiner::Ml,
        }
    }
}

#[pymethods]
impl ChessRefiner {
    /// Default-construct: center-of-mass with default tuning.
    #[new]
    fn new() -> Self {
        Self::from_rs(RsChessRefiner::default())
    }

    /// Center-of-mass refinement on the response map.
    #[classmethod]
    #[pyo3(name = "center_of_mass", signature = (cfg=None))]
    fn ctor_center_of_mass(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        cfg: Option<Py<CenterOfMassConfig>>,
    ) -> Self {
        let inner = cfg.map(|c| c.borrow(py).inner).unwrap_or_default();
        Self {
            kind: ChessRefinerKind::CenterOfMass,
            center_of_mass: inner,
            forstner: RsForstnerConfig::default(),
            saddle_point: RsSaddlePointConfig::default(),
        }
    }

    /// Förstner structure-tensor refinement on the image patch.
    #[classmethod]
    #[pyo3(name = "forstner", signature = (cfg=None))]
    fn ctor_forstner(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        cfg: Option<Py<ForstnerConfig>>,
    ) -> Self {
        let inner = cfg.map(|c| c.borrow(py).inner).unwrap_or_default();
        Self {
            kind: ChessRefinerKind::Forstner,
            center_of_mass: RsCenterOfMassConfig::default(),
            forstner: inner,
            saddle_point: RsSaddlePointConfig::default(),
        }
    }

    /// Quadratic surface fit at the saddle point of the image patch.
    #[classmethod]
    #[pyo3(name = "saddle_point", signature = (cfg=None))]
    fn ctor_saddle_point(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        cfg: Option<Py<SaddlePointConfig>>,
    ) -> Self {
        let inner = cfg.map(|c| c.borrow(py).inner).unwrap_or_default();
        Self {
            kind: ChessRefinerKind::SaddlePoint,
            center_of_mass: RsCenterOfMassConfig::default(),
            forstner: RsForstnerConfig::default(),
            saddle_point: inner,
        }
    }

    /// ML-backed subpixel refinement. Requires the `ml-refiner` feature.
    #[cfg(feature = "ml-refiner")]
    #[classmethod]
    #[pyo3(name = "ml")]
    fn ctor_ml(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            kind: ChessRefinerKind::Ml,
            center_of_mass: RsCenterOfMassConfig::default(),
            forstner: RsForstnerConfig::default(),
            saddle_point: RsSaddlePointConfig::default(),
        }
    }

    /// Variant tag: `"center_of_mass"`, `"forstner"`, `"saddle_point"`,
    /// or `"ml"` (with the `ml-refiner` feature).
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Tuning payload of the active variant. Returns the typed config
    /// struct (`CenterOfMassConfig`, `ForstnerConfig`, or
    /// `SaddlePointConfig`); returns `None` for the `ml` variant.
    /// Modifying the returned object does not affect the refiner;
    /// rebuild via the appropriate classmethod factory to apply
    /// changes.
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.kind {
            ChessRefinerKind::CenterOfMass => Ok(Py::new(
                py,
                CenterOfMassConfig {
                    inner: self.center_of_mass,
                },
            )?
            .into_any()),
            ChessRefinerKind::Forstner => Ok(Py::new(
                py,
                ForstnerConfig {
                    inner: self.forstner,
                },
            )?
            .into_any()),
            ChessRefinerKind::SaddlePoint => Ok(Py::new(
                py,
                SaddlePointConfig {
                    inner: self.saddle_point,
                },
            )?
            .into_any()),
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => Ok(py.None()),
        }
    }

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.kind {
            ChessRefinerKind::CenterOfMass => {
                let cfg = CenterOfMassConfig {
                    inner: self.center_of_mass,
                };
                d.set_item("center_of_mass", cfg.to_dict(py)?)?;
            }
            ChessRefinerKind::Forstner => {
                let cfg = ForstnerConfig {
                    inner: self.forstner,
                };
                d.set_item("forstner", cfg.to_dict(py)?)?;
            }
            ChessRefinerKind::SaddlePoint => {
                let cfg = SaddlePointConfig {
                    inner: self.saddle_point,
                };
                d.set_item("saddle_point", cfg.to_dict(py)?)?;
            }
            #[cfg(feature = "ml-refiner")]
            ChessRefinerKind::Ml => {
                d.set_item("ml", py.None())?;
            }
        }
        Ok(d.unbind())
    }

    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Permit only the four (or three, without ml-refiner) variants.
        #[cfg(feature = "ml-refiner")]
        let allowed: &[&str] = &["center_of_mass", "forstner", "saddle_point", "ml"];
        #[cfg(not(feature = "ml-refiner"))]
        let allowed: &[&str] = &["center_of_mass", "forstner", "saddle_point"];

        // Accept the externally-tagged shape produced by serde:
        //   "ml"          (bare string — unit variant, ml-refiner only)
        //   { "ml": null } (object form — also accepted)
        //   { "forstner": {...} } (payload variant — object only)
        #[cfg(feature = "ml-refiner")]
        let unit_tags: &[&str] = &["ml"];
        #[cfg(not(feature = "ml-refiner"))]
        let unit_tags: &[&str] = &[];
        let payload_tags: &[&str] = &["center_of_mass", "forstner", "saddle_point"];

        if let Some(_tag) =
            super::parse::accept_dict_or_bare_string(data, unit_tags, payload_tags, "refiner")?
        {
            #[cfg(feature = "ml-refiner")]
            return Ok(Self {
                kind: ChessRefinerKind::Ml,
                center_of_mass: RsCenterOfMassConfig::default(),
                forstner: RsForstnerConfig::default(),
                saddle_point: RsSaddlePointConfig::default(),
            });
            // Without ml-refiner the unit_tags slice is empty, so this branch
            // is unreachable; the cfg below silences the dead-code warning.
            #[cfg(not(feature = "ml-refiner"))]
            return Err(config_error("refiner: no unit variants available"));
        }

        let dict = data.cast::<PyDict>().cloned().unwrap();
        reject_unknown_keys(&dict, allowed, "refiner")?;

        let mut found: Option<&str> = None;
        for &key in allowed.iter() {
            if dict.get_item(key)?.is_some() {
                if found.is_some() {
                    return Err(config_error(format!(
                        "refiner must have exactly one of: {}",
                        allowed.join(", ")
                    )));
                }
                found = Some(key);
            }
        }
        let Some(key) = found else {
            return Err(config_error(format!(
                "refiner must have one of: {}",
                allowed.join(", ")
            )));
        };

        let value = dict
            .get_item(key)?
            .ok_or_else(|| config_error(format!("refiner.{key} missing")))?;

        let py = data.py();
        match key {
            "center_of_mass" => {
                let cls = py.get_type::<CenterOfMassConfig>();
                let cfg = CenterOfMassConfig::from_dict(&cls, &value)?;
                Ok(Self {
                    kind: ChessRefinerKind::CenterOfMass,
                    center_of_mass: cfg.inner,
                    forstner: RsForstnerConfig::default(),
                    saddle_point: RsSaddlePointConfig::default(),
                })
            }
            "forstner" => {
                let cls = py.get_type::<ForstnerConfig>();
                let cfg = ForstnerConfig::from_dict(&cls, &value)?;
                Ok(Self {
                    kind: ChessRefinerKind::Forstner,
                    center_of_mass: RsCenterOfMassConfig::default(),
                    forstner: cfg.inner,
                    saddle_point: RsSaddlePointConfig::default(),
                })
            }
            "saddle_point" => {
                let cls = py.get_type::<SaddlePointConfig>();
                let cfg = SaddlePointConfig::from_dict(&cls, &value)?;
                Ok(Self {
                    kind: ChessRefinerKind::SaddlePoint,
                    center_of_mass: RsCenterOfMassConfig::default(),
                    forstner: RsForstnerConfig::default(),
                    saddle_point: cfg.inner,
                })
            }
            #[cfg(feature = "ml-refiner")]
            "ml" => Ok(Self {
                kind: ChessRefinerKind::Ml,
                center_of_mass: RsCenterOfMassConfig::default(),
                forstner: RsForstnerConfig::default(),
                saddle_point: RsSaddlePointConfig::default(),
            }),
            other => Err(config_error(format!("refiner.{other} not supported"))),
        }
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
