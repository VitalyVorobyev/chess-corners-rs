//! Native PyO3 wrappers around `chess-corners` config structs.
//!
//! Each wrapper owns a Rust value (or `Py<...>` handles to nested
//! wrappers, in the case of compound types). Python-side users get
//! attribute access, chained mutation, classmethod factories, and
//! `to_dict`/`to_json`/`pretty`/`print` helpers identical to the
//! prior pure-Python dataclass surface. The FFI does not serialize
//! through JSON: `Detector(cfg).detect(image)` accepts the typed
//! [`DetectorConfig`] wrapper directly.
//!
//! The Rust source-of-truth structs in `chess-corners` /
//! `chess-corners-core` are unchanged; only the binding layer adds
//! these wrappers, preserving the workspace dependency rule that
//! core crates do not depend on `pyo3`.

use std::collections::BTreeSet;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessConfig as RsChessConfig,
    ChessRefiner as RsChessRefiner, ChessRing as RsChessRing, DetectionParams as RsDetectionParams,
    DetectionStrategy as RsDetectionStrategy, DetectorConfig as RsDetectorConfig,
    ForstnerConfig as RsForstnerConfig, MultiscaleConfig as RsMultiscaleConfig,
    OrientationMethod as RsOrientationMethod, PeakFitMode as RsPeakFitMode,
    RadonConfig as RsRadonConfig, RadonPeakConfig as RsRadonPeakConfig,
    SaddlePointConfig as RsSaddlePointConfig, UpscaleConfig as RsUpscaleConfig,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString, PyType};

create_exception!(_native, ConfigError, PyValueError);

// ---------------------------------------------------------------------------
// Helpers for dict → field deserialization with unknown-key rejection.
// ---------------------------------------------------------------------------

fn config_error(msg: impl Into<String>) -> PyErr {
    ConfigError::new_err(msg.into())
}

fn require_dict<'py>(value: &Bound<'py, PyAny>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    value
        .cast::<PyDict>()
        .cloned()
        .map_err(|_| config_error(format!("{path} must be an object")))
}

/// For an externally-tagged enum that has both unit variants (serialised by
/// serde as a bare string, e.g. `"single_scale"`) and payload variants
/// (serialised as `{"pyramid": {...}}`), accept either encoding on the
/// input side.
///
/// Returns:
/// - `Ok(None)` — `value` is a dict; the caller parses it normally.
/// - `Ok(Some(tag))` — `value` is the string `tag`, matched a unit-variant name.
/// - `Err(ConfigError)` — bare string that doesn't match any unit tag, or
///   a payload-only variant name used as a bare string.
fn accept_dict_or_bare_string<'py>(
    value: &Bound<'py, PyAny>,
    unit_tags: &[&str],
    payload_tags: &[&str],
    path: &str,
) -> PyResult<Option<String>> {
    if value.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    if value.is_instance_of::<PyString>() {
        let s: String = value.extract()?;
        if unit_tags.contains(&s.as_str()) {
            return Ok(Some(s));
        }
        if payload_tags.contains(&s.as_str()) {
            return Err(config_error(format!(
                "{path}: \"{s}\" requires a payload — use {{\"{}\":{{...}}}} instead",
                s
            )));
        }
        let all: Vec<&str> = unit_tags
            .iter()
            .chain(payload_tags.iter())
            .copied()
            .collect();
        return Err(config_error(format!(
            "{path}: got bare string \"{s}\", expected one of: {}",
            all.join(", ")
        )));
    }
    Err(config_error(format!("{path} must be an object or string")))
}

fn reject_unknown_keys(dict: &Bound<'_, PyDict>, allowed: &[&str], path: &str) -> PyResult<()> {
    let allowed: BTreeSet<&str> = allowed.iter().copied().collect();
    let mut unknown: Vec<String> = Vec::new();
    for key in dict.keys().iter() {
        let s: String = key.extract()?;
        if !allowed.contains(s.as_str()) {
            unknown.push(s);
        }
    }
    if !unknown.is_empty() {
        unknown.sort();
        return Err(config_error(format!(
            "{path} has unknown keys: {}",
            unknown.join(", ")
        )));
    }
    Ok(())
}

fn extract_int(dict: &Bound<'_, PyDict>, key: &str, path: &str) -> PyResult<Option<i64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(config_error(format!("{path}.{key} must be an integer")));
    }
    Ok(Some(value.extract::<i64>()?))
}

fn extract_float(dict: &Bound<'_, PyDict>, key: &str, path: &str) -> PyResult<Option<f64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
    {
        return Err(config_error(format!("{path}.{key} must be a number")));
    }
    Ok(Some(value.extract::<f64>()?))
}

fn extract_string(dict: &Bound<'_, PyDict>, key: &str, path: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if !value.is_instance_of::<PyString>() {
        return Err(config_error(format!("{path}.{key} must be a string")));
    }
    Ok(Some(value.extract::<String>()?))
}

// ---------------------------------------------------------------------------
// Enums.
// ---------------------------------------------------------------------------

macro_rules! py_enum {
    (
        $(#[$attr:meta])*
        $name:ident, $rs:path,
        [$( ($variant:ident, $py_name:literal, $rs_variant:ident) ),+ $(,)?]
    ) => {
        $(#[$attr])*
        #[pyclass(eq, eq_int, module = "chess_corners", from_py_object)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $(
                #[pyo3(name = $py_name)]
                $variant,
            )+
        }

        impl From<$name> for $rs {
            fn from(v: $name) -> Self {
                match v {
                    $( $name::$variant => <$rs>::$rs_variant, )+
                }
            }
        }

        impl From<$rs> for $name {
            #[allow(unreachable_patterns)]
            fn from(v: $rs) -> Self {
                match v {
                    $( <$rs>::$rs_variant => $name::$variant, )+
                    // Future-proof for `#[non_exhaustive]` source enums.
                    _ => $name::default_for_unknown(),
                }
            }
        }

        impl $name {
            fn default_for_unknown() -> Self {
                Self::__first()
            }
            fn __first() -> Self {
                #[allow(unreachable_code, clippy::needless_return)]
                {
                    $( return $name::$variant; )+
                }
            }
        }
    };
}

py_enum!(
    /// ChESS detector sampling-ring choice.
    /// `CANONICAL` is the paper-default radius-5 ring; `BROAD` uses
    /// the radius-10 ring for low-resolution or heavily blurred inputs.
    ChessRing, RsChessRing,
    [
        (Canonical, "CANONICAL", Canonical),
        (Broad, "BROAD", Broad),
    ]
);

py_enum!(
    /// Subpixel peak-fit mode used by the Radon refiner and detector.
    PeakFitMode, RsPeakFitMode,
    [
        (Parabolic, "PARABOLIC", Parabolic),
        (Gaussian, "GAUSSIAN", Gaussian),
    ]
);

py_enum!(
    /// Orientation-fit method used to estimate the two grid axes at
    /// each detected corner.
    ///
    /// `RING_FIT` *(default)* — fits the parametric two-axis chessboard
    /// intensity model to the 16-sample ring via Gauss-Newton, with
    /// per-axis 1σ uncertainties calibrated by a piecewise-linear lookup
    /// table. Suitable for the full range of standard chessboard images.
    ///
    /// `DISK_FIT` — full-disk crossing-line estimator. Samples all image
    /// pixels in a disk around the corner center and fits two possibly
    /// non-orthogonal axes. Falls back to `RING_FIT` on clean orthogonal
    /// corners (lazy gate) and near image borders. Use when corners are
    /// imaged under strong projective warp.
    OrientationMethod, RsOrientationMethod,
    [
        (RingFit, "RING_FIT", RingFit),
        (DiskFit, "DISK_FIT", DiskFit),
    ]
);

fn parse_enum<E>(value: &str, path: &str, allowed: &[(&str, E)]) -> PyResult<E>
where
    E: Copy,
{
    for (name, variant) in allowed {
        if *name == value {
            return Ok(*variant);
        }
    }
    let allowed_str = allowed
        .iter()
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(", ");
    Err(config_error(format!(
        "{path} must be one of: {allowed_str}"
    )))
}

fn parse_chess_ring(value: &str, path: &str) -> PyResult<RsChessRing> {
    parse_enum(
        value,
        path,
        &[
            ("canonical", RsChessRing::Canonical),
            ("broad", RsChessRing::Broad),
        ],
    )
}

fn parse_peak_fit_mode(value: &str, path: &str) -> PyResult<RsPeakFitMode> {
    parse_enum(
        value,
        path,
        &[
            ("parabolic", RsPeakFitMode::Parabolic),
            ("gaussian", RsPeakFitMode::Gaussian),
        ],
    )
}

fn parse_orientation_method(value: &str, path: &str) -> PyResult<RsOrientationMethod> {
    parse_enum(
        value,
        path,
        &[
            ("ring_fit", RsOrientationMethod::RingFit),
            ("disk_fit", RsOrientationMethod::DiskFit),
        ],
    )
}

fn chess_ring_str(v: RsChessRing) -> &'static str {
    match v {
        RsChessRing::Canonical => "canonical",
        RsChessRing::Broad => "broad",
        _ => "canonical",
    }
}

fn peak_fit_mode_str(v: RsPeakFitMode) -> &'static str {
    match v {
        RsPeakFitMode::Parabolic => "parabolic",
        RsPeakFitMode::Gaussian => "gaussian",
        _ => "gaussian",
    }
}

fn orientation_method_str(v: RsOrientationMethod) -> &'static str {
    match v {
        RsOrientationMethod::RingFit => "ring_fit",
        RsOrientationMethod::DiskFit => "disk_fit",
        _ => "ring_fit",
    }
}

// ---------------------------------------------------------------------------
// Shared json helpers.
// ---------------------------------------------------------------------------

fn json_dumps(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    indent: Option<i64>,
    sort_keys: bool,
) -> PyResult<String> {
    let json = py.import("json")?;
    let kwargs = PyDict::new(py);
    if let Some(i) = indent {
        kwargs.set_item("indent", i)?;
    }
    kwargs.set_item("sort_keys", sort_keys)?;
    let result = json.call_method("dumps", (obj,), Some(&kwargs))?;
    result.extract::<String>()
}

fn json_loads<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    let json = py.import("json")?;
    json.call_method1("loads", (text,))
}

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
// RadonPeakConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct RadonPeakConfig {
    pub(crate) inner: RsRadonPeakConfig,
}

#[pymethods]
impl RadonPeakConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsRadonPeakConfig::default(),
        }
    }

    #[getter]
    fn ray_radius(&self) -> u32 {
        self.inner.ray_radius
    }
    #[setter]
    fn set_ray_radius(&mut self, v: u32) {
        self.inner.ray_radius = v;
    }

    #[getter]
    fn patch_radius(&self) -> u32 {
        self.inner.patch_radius
    }
    #[setter]
    fn set_patch_radius(&mut self, v: u32) {
        self.inner.patch_radius = v;
    }

    #[getter]
    fn image_upsample(&self) -> u32 {
        self.inner.image_upsample
    }
    #[setter]
    fn set_image_upsample(&mut self, v: u32) {
        self.inner.image_upsample = v;
    }

    #[getter]
    fn response_blur_radius(&self) -> u32 {
        self.inner.response_blur_radius
    }
    #[setter]
    fn set_response_blur_radius(&mut self, v: u32) {
        self.inner.response_blur_radius = v;
    }

    #[getter]
    fn peak_fit(&self) -> PeakFitMode {
        self.inner.peak_fit.into()
    }
    #[setter]
    fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.inner.peak_fit = v.into();
    }

    #[getter]
    fn min_response(&self) -> f32 {
        self.inner.min_response
    }
    #[setter]
    fn set_min_response(&mut self, v: f32) {
        self.inner.min_response = v;
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
        d.set_item("ray_radius", self.inner.ray_radius)?;
        d.set_item("patch_radius", self.inner.patch_radius)?;
        d.set_item("image_upsample", self.inner.image_upsample)?;
        d.set_item("response_blur_radius", self.inner.response_blur_radius)?;
        d.set_item("peak_fit", peak_fit_mode_str(self.inner.peak_fit))?;
        d.set_item("min_response", self.inner.min_response as f64)?;
        d.set_item("max_offset", self.inner.max_offset as f64)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "radon_peak")?;
        reject_unknown_keys(
            &dict,
            &[
                "ray_radius",
                "patch_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "min_response",
                "max_offset",
            ],
            "radon_peak",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "ray_radius", "radon_peak")? {
            cfg.inner.ray_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "patch_radius", "radon_peak")? {
            cfg.inner.patch_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "image_upsample", "radon_peak")? {
            cfg.inner.image_upsample = v as u32;
        }
        if let Some(v) = extract_int(&dict, "response_blur_radius", "radon_peak")? {
            cfg.inner.response_blur_radius = v as u32;
        }
        if let Some(s) = extract_string(&dict, "peak_fit", "radon_peak")? {
            cfg.inner.peak_fit = parse_peak_fit_mode(&s, "radon_peak.peak_fit")?;
        }
        if let Some(v) = extract_float(&dict, "min_response", "radon_peak")? {
            cfg.inner.min_response = v as f32;
        }
        if let Some(v) = extract_float(&dict, "max_offset", "radon_peak")? {
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
    fn from_rs(v: RsMultiscaleConfig) -> Self {
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

    fn to_rs(&self) -> RsMultiscaleConfig {
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
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
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
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
    fn from_rs(v: RsUpscaleConfig) -> Self {
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

    fn to_rs(&self) -> RsUpscaleConfig {
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
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
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
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
    fn from_rs(v: RsChessRefiner) -> Self {
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
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
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
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

        if let Some(_tag) = accept_dict_or_bare_string(data, unit_tags, payload_tags, "refiner")? {
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

// ---------------------------------------------------------------------------
// ChessConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct ChessConfig {
    pub(crate) ring: RsChessRing,
    pub(crate) refiner: Py<ChessRefiner>,
}

impl ChessConfig {
    fn from_rs(py: Python<'_>, v: RsChessConfig) -> PyResult<Self> {
        Ok(Self {
            ring: v.ring,
            refiner: Py::new(py, ChessRefiner::from_rs(v.refiner))?,
        })
    }

    pub(crate) fn to_rs(&self, py: Python<'_>) -> RsChessConfig {
        let mut cfg = RsChessConfig::default();
        cfg.ring = self.ring;
        cfg.refiner = self.refiner.borrow(py).to_rs();
        cfg
    }
}

#[pymethods]
impl ChessConfig {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::default())
    }

    #[getter]
    fn ring(&self) -> ChessRing {
        self.ring.into()
    }
    #[setter]
    fn set_ring(&mut self, v: ChessRing) {
        self.ring = v.into();
    }

    #[getter]
    fn refiner(&self, py: Python<'_>) -> Py<ChessRefiner> {
        self.refiner.clone_ref(py)
    }
    #[setter]
    fn set_refiner(&mut self, v: Py<ChessRefiner>) {
        self.refiner = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ring", chess_ring_str(self.ring))?;
        d.set_item("refiner", self.refiner.borrow(py).to_dict(py)?)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "chess")?;
        reject_unknown_keys(&dict, &["ring", "refiner"], "chess")?;
        let mut cfg = Self::from_rs(py, RsChessConfig::default())?;
        if let Some(s) = extract_string(&dict, "ring", "chess")? {
            cfg.ring = parse_chess_ring(&s, "chess.ring")?;
        }
        if let Some(value) = dict.get_item("refiner")? {
            let cls = py.get_type::<ChessRefiner>();
            cfg.refiner = Py::new(py, ChessRefiner::from_dict(&cls, &value)?)?;
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
        Self::from_dict(cls, py, &value)
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
// RadonConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct RadonConfig {
    pub(crate) ray_radius: u32,
    pub(crate) image_upsample: u32,
    pub(crate) response_blur_radius: u32,
    pub(crate) peak_fit: RsPeakFitMode,
}

impl RadonConfig {
    fn from_rs(_py: Python<'_>, v: RsRadonConfig) -> PyResult<Self> {
        Ok(Self {
            ray_radius: v.ray_radius,
            image_upsample: v.image_upsample,
            response_blur_radius: v.response_blur_radius,
            peak_fit: v.peak_fit,
        })
    }

    pub(crate) fn to_rs(&self, _py: Python<'_>) -> RsRadonConfig {
        let mut cfg = RsRadonConfig::default();
        cfg.ray_radius = self.ray_radius;
        cfg.image_upsample = self.image_upsample;
        cfg.response_blur_radius = self.response_blur_radius;
        cfg.peak_fit = self.peak_fit;
        cfg
    }
}

#[pymethods]
impl RadonConfig {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsRadonConfig::default())
    }

    #[getter]
    fn ray_radius(&self) -> u32 {
        self.ray_radius
    }
    #[setter]
    fn set_ray_radius(&mut self, v: u32) {
        self.ray_radius = v;
    }

    #[getter]
    fn image_upsample(&self) -> u32 {
        self.image_upsample
    }
    #[setter]
    fn set_image_upsample(&mut self, v: u32) {
        self.image_upsample = v;
    }

    #[getter]
    fn response_blur_radius(&self) -> u32 {
        self.response_blur_radius
    }
    #[setter]
    fn set_response_blur_radius(&mut self, v: u32) {
        self.response_blur_radius = v;
    }

    #[getter]
    fn peak_fit(&self) -> PeakFitMode {
        self.peak_fit.into()
    }
    #[setter]
    fn set_peak_fit(&mut self, v: PeakFitMode) {
        self.peak_fit = v.into();
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ray_radius", self.ray_radius)?;
        d.set_item("image_upsample", self.image_upsample)?;
        d.set_item("response_blur_radius", self.response_blur_radius)?;
        d.set_item("peak_fit", peak_fit_mode_str(self.peak_fit))?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "radon")?;
        reject_unknown_keys(
            &dict,
            &[
                "ray_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "refiner",
            ],
            "radon",
        )?;
        let mut cfg = Self::from_rs(py, RsRadonConfig::default())?;
        if let Some(v) = extract_int(&dict, "ray_radius", "radon")? {
            cfg.ray_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "image_upsample", "radon")? {
            cfg.image_upsample = v as u32;
        }
        if let Some(v) = extract_int(&dict, "response_blur_radius", "radon")? {
            cfg.response_blur_radius = v as u32;
        }
        if let Some(s) = extract_string(&dict, "peak_fit", "radon")? {
            cfg.peak_fit = parse_peak_fit_mode(&s, "radon.peak_fit")?;
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
        Self::from_dict(cls, py, &value)
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
// DetectionParams (shared NMS / clustering knobs)
// ---------------------------------------------------------------------------

/// Shared non-maximum-suppression and peak-clustering thresholds applied
/// by both the ChESS and Radon detectors. Lives on
/// [`DetectorConfig.detection`](DetectorConfig); set it via attribute
/// access or `DetectorConfig.with_detection(...)`.
#[pyclass(module = "chess_corners")]
pub struct DetectionParams {
    pub(crate) inner: RsDetectionParams,
}

impl DetectionParams {
    fn from_rs(v: RsDetectionParams) -> Self {
        Self { inner: v }
    }

    pub(crate) fn to_rs(&self) -> RsDetectionParams {
        self.inner
    }
}

#[pymethods]
impl DetectionParams {
    #[new]
    fn py_new() -> Self {
        Self::from_rs(RsDetectionParams::default())
    }

    #[getter]
    fn nms_radius(&self) -> u32 {
        self.inner.nms_radius
    }
    #[setter]
    fn set_nms_radius(&mut self, v: u32) {
        self.inner.nms_radius = v;
    }

    #[getter]
    fn min_cluster_size(&self) -> u32 {
        self.inner.min_cluster_size
    }
    #[setter]
    fn set_min_cluster_size(&mut self, v: u32) {
        self.inner.min_cluster_size = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("nms_radius", self.inner.nms_radius)?;
        d.set_item("min_cluster_size", self.inner.min_cluster_size)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "detection")?;
        reject_unknown_keys(&dict, &["nms_radius", "min_cluster_size"], "detection")?;
        let mut cfg = Self::py_new();
        if let Some(v) = extract_int(&dict, "nms_radius", "detection")? {
            cfg.inner.nms_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "min_cluster_size", "detection")? {
            cfg.inner.min_cluster_size = v as u32;
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
// DetectionStrategy (variant-tagged wrapper).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum StrategyKind {
    Chess,
    Radon,
}

impl StrategyKind {
    fn as_str(&self) -> &'static str {
        match self {
            StrategyKind::Chess => "chess",
            StrategyKind::Radon => "radon",
        }
    }
}

/// Top-level detector dispatch. One of `chess(ChessConfig)` or
/// `radon(RadonConfig)`. Carries all detector-specific tuning;
/// settings that don't apply to the active detector are unreachable.
#[pyclass(module = "chess_corners")]
pub struct DetectionStrategy {
    kind: StrategyKind,
    chess: Py<ChessConfig>,
    radon: Py<RadonConfig>,
}

impl DetectionStrategy {
    fn from_rs(py: Python<'_>, v: RsDetectionStrategy) -> PyResult<Self> {
        match v {
            RsDetectionStrategy::Chess(c) => Ok(Self {
                kind: StrategyKind::Chess,
                chess: Py::new(py, ChessConfig::from_rs(py, c)?)?,
                radon: Py::new(py, RadonConfig::from_rs(py, RsRadonConfig::default())?)?,
            }),
            RsDetectionStrategy::Radon(r) => Ok(Self {
                kind: StrategyKind::Radon,
                chess: Py::new(py, ChessConfig::from_rs(py, RsChessConfig::default())?)?,
                radon: Py::new(py, RadonConfig::from_rs(py, r)?)?,
            }),
            _ => Self::from_rs(py, RsDetectionStrategy::default()),
        }
    }

    pub(crate) fn to_rs(&self, py: Python<'_>) -> RsDetectionStrategy {
        match self.kind {
            StrategyKind::Chess => RsDetectionStrategy::Chess(self.chess.borrow(py).to_rs(py)),
            StrategyKind::Radon => RsDetectionStrategy::Radon(self.radon.borrow(py).to_rs(py)),
        }
    }
}

#[pymethods]
impl DetectionStrategy {
    /// Default-construct: ChESS strategy with library defaults.
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectionStrategy::default())
    }

    /// Build a ChESS-strategy dispatch carrying the given [`ChessConfig`].
    #[classmethod]
    fn from_chess(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        strategy: Py<ChessConfig>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: StrategyKind::Chess,
            chess: strategy,
            radon: Py::new(py, RadonConfig::from_rs(py, RsRadonConfig::default())?)?,
        })
    }

    /// Build a Radon-strategy dispatch carrying the given [`RadonConfig`].
    #[classmethod]
    fn from_radon(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        strategy: Py<RadonConfig>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: StrategyKind::Radon,
            chess: Py::new(py, ChessConfig::from_rs(py, RsChessConfig::default())?)?,
            radon: strategy,
        })
    }

    /// Variant tag: `"chess"` or `"radon"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// The carried `ChessConfig` when the active variant is `chess`,
    /// otherwise `None`.
    #[getter]
    fn chess(&self, py: Python<'_>) -> Option<Py<ChessConfig>> {
        match self.kind {
            StrategyKind::Chess => Some(self.chess.clone_ref(py)),
            _ => None,
        }
    }

    /// The carried `RadonConfig` when the active variant is `radon`,
    /// otherwise `None`.
    #[getter]
    fn radon(&self, py: Python<'_>) -> Option<Py<RadonConfig>> {
        match self.kind {
            StrategyKind::Radon => Some(self.radon.clone_ref(py)),
            _ => None,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        match self.kind {
            StrategyKind::Chess => {
                d.set_item("chess", self.chess.borrow(py).to_dict(py)?)?;
            }
            StrategyKind::Radon => {
                d.set_item("radon", self.radon.borrow(py).to_dict(py)?)?;
            }
        }
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "strategy")?;
        reject_unknown_keys(&dict, &["chess", "radon"], "strategy")?;
        let has_chess = dict.get_item("chess")?.is_some();
        let has_radon = dict.get_item("radon")?.is_some();
        if has_chess && has_radon {
            return Err(config_error(
                "strategy must have exactly one of: chess, radon",
            ));
        }
        if has_chess {
            let value = dict
                .get_item("chess")?
                .ok_or_else(|| config_error("strategy.chess missing"))?;
            let chess_type = py.get_type::<ChessConfig>();
            let chess = Py::new(py, ChessConfig::from_dict(&chess_type, py, &value)?)?;
            return Ok(Self {
                kind: StrategyKind::Chess,
                chess,
                radon: Py::new(py, RadonConfig::from_rs(py, RsRadonConfig::default())?)?,
            });
        }
        if has_radon {
            let value = dict
                .get_item("radon")?
                .ok_or_else(|| config_error("strategy.radon missing"))?;
            let radon_type = py.get_type::<RadonConfig>();
            let radon = Py::new(py, RadonConfig::from_dict(&radon_type, py, &value)?)?;
            return Ok(Self {
                kind: StrategyKind::Radon,
                chess: Py::new(py, ChessConfig::from_rs(py, RsChessConfig::default())?)?,
                radon,
            });
        }
        Err(config_error("strategy must have one of: chess, radon"))
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
        Self::from_dict(cls, py, &value)
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
// DetectorConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct DetectorConfig {
    pub(crate) strategy: Py<DetectionStrategy>,
    pub(crate) threshold: f32,
    pub(crate) detection: Py<DetectionParams>,
    pub(crate) multiscale: Py<MultiscaleConfig>,
    pub(crate) upscale: Py<UpscaleConfig>,
    pub(crate) orientation_method: Option<RsOrientationMethod>,
    pub(crate) merge_radius: f32,
}

impl DetectorConfig {
    pub(crate) fn from_rs(py: Python<'_>, src: RsDetectorConfig) -> PyResult<Self> {
        Ok(Self {
            strategy: Py::new(py, DetectionStrategy::from_rs(py, src.strategy)?)?,
            threshold: src.threshold,
            detection: Py::new(py, DetectionParams::from_rs(src.detection))?,
            multiscale: Py::new(py, MultiscaleConfig::from_rs(src.multiscale))?,
            upscale: Py::new(py, UpscaleConfig::from_rs(src.upscale))?,
            orientation_method: src.orientation_method,
            merge_radius: src.merge_radius,
        })
    }

    fn build(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectorConfig::default())
    }

    /// Deep-clone this wrapper into a new owned `DetectorConfig`.
    fn clone_inner(&self, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, self.to_inner(py))
    }

    /// Convert into the Rust facade's `DetectorConfig`.
    pub(crate) fn to_inner(&self, py: Python<'_>) -> RsDetectorConfig {
        let mut cfg = RsDetectorConfig::default();
        cfg.strategy = self.strategy.borrow(py).to_rs(py);
        cfg.threshold = self.threshold;
        cfg.detection = self.detection.borrow(py).to_rs();
        cfg.multiscale = self.multiscale.borrow(py).to_rs();
        cfg.upscale = self.upscale.borrow(py).to_rs();
        cfg.orientation_method = self.orientation_method;
        cfg.merge_radius = self.merge_radius;
        cfg
    }
}

#[pymethods]
impl DetectorConfig {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::build(py)
    }

    /// Single-scale ChESS preset.
    #[classmethod]
    fn chess(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectorConfig::chess())
    }

    /// Three-level coarse-to-fine ChESS preset.
    #[classmethod]
    fn chess_multiscale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectorConfig::chess_multiscale())
    }

    /// Whole-image Radon detector preset.
    #[classmethod]
    fn radon(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectorConfig::radon())
    }

    /// Coarse-to-fine Radon preset.
    #[classmethod]
    fn radon_multiscale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsDetectorConfig::radon_multiscale())
    }

    // ---- chainable builder methods ----

    /// Return a new `DetectorConfig` with the threshold replaced.
    fn with_threshold(&self, py: Python<'_>, threshold: f32) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.threshold = threshold;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the multiscale config replaced.
    fn with_multiscale(&self, py: Python<'_>, multiscale: Py<MultiscaleConfig>) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.multiscale = multiscale;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the upscale config replaced.
    fn with_upscale(&self, py: Python<'_>, upscale: Py<UpscaleConfig>) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.upscale = upscale;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the orientation method replaced.
    fn with_orientation_method(&self, py: Python<'_>, method: OrientationMethod) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.orientation_method = Some(method.into());
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the per-corner orientation fit
    /// skipped. Detection still yields positions and responses, but the
    /// four axis columns of the output array are `NaN`.
    fn without_orientation(&self, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.orientation_method = None;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the merge radius replaced.
    fn with_merge_radius(&self, py: Python<'_>, radius: f32) -> PyResult<Self> {
        let mut cfg = self.clone_inner(py)?;
        cfg.merge_radius = radius;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the ChESS strategy fields updated
    /// from the provided keyword arguments. If the current strategy is Radon,
    /// it is replaced with a default `ChessConfig` before applying kwargs.
    /// Top-level fields (threshold, multiscale, etc.) are preserved.
    ///
    /// Accepted kwargs: `refiner`, `ring`. The shared
    /// NMS / clustering knobs moved to `with_detection`.
    #[pyo3(signature = (**kwargs))]
    fn with_chess(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        const CHESS_FIELDS: &[&str] = &["refiner", "ring"];

        let mut cfg = self.clone_inner(py)?;

        // Determine the base chess sub-config.
        let base_chess = match cfg.strategy.borrow(py).kind {
            StrategyKind::Chess => cfg.strategy.borrow(py).chess.borrow(py).to_rs(py),
            StrategyKind::Radon => RsChessConfig::default(),
        };
        let mut chess = ChessConfig::from_rs(py, base_chess)?;

        if let Some(kw) = kwargs {
            for key in kw.keys().iter() {
                let key_str: String = key.extract()?;
                if !CHESS_FIELDS.contains(&key_str.as_str()) {
                    return Err(PyTypeError::new_err(format!(
                        "unexpected keyword argument: '{key_str}'"
                    )));
                }
            }
            if let Some(v) = kw.get_item("refiner")? {
                chess.refiner = v.extract::<Py<ChessRefiner>>()?;
            }
            if let Some(v) = kw.get_item("ring")? {
                let ring: ChessRing = v.extract()?;
                chess.ring = ring.into();
            }
        }

        let strategy = DetectionStrategy {
            kind: StrategyKind::Chess,
            chess: Py::new(py, chess)?,
            radon: Py::new(py, RadonConfig::from_rs(py, RsRadonConfig::default())?)?,
        };
        cfg.strategy = Py::new(py, strategy)?;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the Radon strategy fields updated
    /// from the provided keyword arguments. If the current strategy is ChESS,
    /// it is replaced with a default `RadonConfig` before applying kwargs.
    /// Top-level fields (threshold, multiscale, etc.) are preserved.
    ///
    /// Accepted kwargs: `refiner`, `ray_radius`, `image_upsample`,
    /// `response_blur_radius`, `peak_fit`. The shared NMS / clustering
    /// knobs moved to `with_detection`.
    #[pyo3(signature = (**kwargs))]
    fn with_radon(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        const RADON_FIELDS: &[&str] = &[
            "refiner",
            "ray_radius",
            "image_upsample",
            "response_blur_radius",
            "peak_fit",
        ];

        let mut cfg = self.clone_inner(py)?;

        let base_radon = match cfg.strategy.borrow(py).kind {
            StrategyKind::Radon => cfg.strategy.borrow(py).radon.borrow(py).to_rs(py),
            StrategyKind::Chess => RsRadonConfig::default(),
        };
        let mut radon = RadonConfig::from_rs(py, base_radon)?;

        if let Some(kw) = kwargs {
            for key in kw.keys().iter() {
                let key_str: String = key.extract()?;
                if !RADON_FIELDS.contains(&key_str.as_str()) {
                    return Err(PyTypeError::new_err(format!(
                        "unexpected keyword argument: '{key_str}'"
                    )));
                }
            }
            if let Some(v) = kw.get_item("ray_radius")? {
                radon.ray_radius = v.extract::<u32>()?;
            }
            if let Some(v) = kw.get_item("image_upsample")? {
                radon.image_upsample = v.extract::<u32>()?;
            }
            if let Some(v) = kw.get_item("response_blur_radius")? {
                radon.response_blur_radius = v.extract::<u32>()?;
            }
            if let Some(v) = kw.get_item("peak_fit")? {
                let pf: PeakFitMode = v.extract()?;
                radon.peak_fit = pf.into();
            }
        }

        let strategy = DetectionStrategy {
            kind: StrategyKind::Radon,
            chess: Py::new(py, ChessConfig::from_rs(py, RsChessConfig::default())?)?,
            radon: Py::new(py, radon)?,
        };
        cfg.strategy = Py::new(py, strategy)?;
        Ok(cfg)
    }

    /// Return a new `DetectorConfig` with the shared detection params
    /// (NMS / clustering thresholds honoured by both strategies) updated
    /// from the provided keyword arguments. Top-level fields and the
    /// active strategy are preserved.
    ///
    /// Accepted kwargs: `nms_radius`, `min_cluster_size`.
    #[pyo3(signature = (**kwargs))]
    fn with_detection(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        const DETECTION_FIELDS: &[&str] = &["nms_radius", "min_cluster_size"];

        let cfg = self.clone_inner(py)?;
        if let Some(kw) = kwargs {
            for key in kw.keys().iter() {
                let key_str: String = key.extract()?;
                if !DETECTION_FIELDS.contains(&key_str.as_str()) {
                    return Err(PyTypeError::new_err(format!(
                        "unexpected keyword argument: '{key_str}'"
                    )));
                }
            }
            let mut detection = cfg.detection.borrow_mut(py);
            if let Some(v) = kw.get_item("nms_radius")? {
                detection.inner.nms_radius = v.extract::<u32>()?;
            }
            if let Some(v) = kw.get_item("min_cluster_size")? {
                detection.inner.min_cluster_size = v.extract::<u32>()?;
            }
        }
        Ok(cfg)
    }

    // ---- nested wrappers (returned by reference) ----

    #[getter]
    fn strategy(&self, py: Python<'_>) -> Py<DetectionStrategy> {
        self.strategy.clone_ref(py)
    }
    #[setter]
    fn set_strategy(&mut self, v: Py<DetectionStrategy>) {
        self.strategy = v;
    }

    #[getter]
    fn threshold(&self) -> f32 {
        self.threshold
    }
    #[setter]
    fn set_threshold(&mut self, v: f32) {
        self.threshold = v;
    }

    #[getter]
    fn detection(&self, py: Python<'_>) -> Py<DetectionParams> {
        self.detection.clone_ref(py)
    }
    #[setter]
    fn set_detection(&mut self, v: Py<DetectionParams>) {
        self.detection = v;
    }

    #[getter]
    fn multiscale(&self, py: Python<'_>) -> Py<MultiscaleConfig> {
        self.multiscale.clone_ref(py)
    }
    #[setter]
    fn set_multiscale(&mut self, v: Py<MultiscaleConfig>) {
        self.multiscale = v;
    }

    #[getter]
    fn upscale(&self, py: Python<'_>) -> Py<UpscaleConfig> {
        self.upscale.clone_ref(py)
    }
    #[setter]
    fn set_upscale(&mut self, v: Py<UpscaleConfig>) {
        self.upscale = v;
    }

    #[getter]
    fn orientation_method(&self) -> Option<OrientationMethod> {
        self.orientation_method.map(Into::into)
    }
    #[setter]
    fn set_orientation_method(&mut self, v: Option<OrientationMethod>) {
        self.orientation_method = v.map(Into::into);
    }

    #[getter]
    fn merge_radius(&self) -> f32 {
        self.merge_radius
    }
    #[setter]
    fn set_merge_radius(&mut self, v: f32) {
        self.merge_radius = v;
    }

    // ---- serialization ----

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("strategy", self.strategy.borrow(py).to_dict(py)?)?;
        d.set_item("threshold", self.threshold as f64)?;
        d.set_item("detection", self.detection.borrow(py).to_dict(py)?)?;
        d.set_item("multiscale", self.multiscale.borrow(py).to_dict(py)?)?;
        d.set_item("upscale", self.upscale.borrow(py).to_dict(py)?)?;
        // `None` (skip orientation) serialises to a JSON `null`.
        d.set_item(
            "orientation_method",
            self.orientation_method.map(orientation_method_str),
        )?;
        d.set_item("merge_radius", self.merge_radius as f64)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "config")?;
        if dict.get_item("refiner")?.is_some() {
            return Err(config_error(
                "refiner moved into strategy.{chess,radon}.refiner in 0.10.0",
            ));
        }
        reject_unknown_keys(
            &dict,
            &[
                "strategy",
                "threshold",
                "detection",
                "multiscale",
                "upscale",
                "orientation_method",
                "merge_radius",
            ],
            "config",
        )?;
        let mut cfg = Self::build(py)?;
        if let Some(value) = dict.get_item("strategy")? {
            let cls = py.get_type::<DetectionStrategy>();
            cfg.strategy = Py::new(py, DetectionStrategy::from_dict(&cls, py, &value)?)?;
        }
        if let Some(value) = dict.get_item("threshold")? {
            cfg.threshold = value
                .extract::<f64>()
                .map(|v| v as f32)
                .or_else(|_| value.extract::<f32>())
                .map_err(|_| config_error("threshold must be a number"))?;
        }
        if let Some(value) = dict.get_item("detection")? {
            let cls = py.get_type::<DetectionParams>();
            cfg.detection = Py::new(py, DetectionParams::from_dict(&cls, &value)?)?;
        }
        if let Some(value) = dict.get_item("multiscale")? {
            let cls = py.get_type::<MultiscaleConfig>();
            cfg.multiscale = Py::new(py, MultiscaleConfig::from_dict(&cls, &value)?)?;
        }
        if let Some(value) = dict.get_item("upscale")? {
            let cls = py.get_type::<UpscaleConfig>();
            cfg.upscale = Py::new(py, UpscaleConfig::from_dict(&cls, &value)?)?;
        }
        // A JSON `null` selects "skip orientation"; a string selects a
        // method; an absent key keeps the default.
        if let Some(value) = dict.get_item("orientation_method")? {
            if value.is_none() {
                cfg.orientation_method = None;
            } else if value.is_instance_of::<PyString>() {
                let s = value.extract::<String>()?;
                cfg.orientation_method =
                    Some(parse_orientation_method(&s, "config.orientation_method")?);
            } else {
                return Err(config_error(
                    "config.orientation_method must be a string or null",
                ));
            }
        }
        if let Some(v) = extract_float(&dict, "merge_radius", "config")? {
            cfg.merge_radius = v as f32;
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
        Self::from_dict(cls, py, &value)
    }

    #[classmethod]
    fn from_json_file(
        cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let path_str: String = py
            .import("os")?
            .call_method1("fspath", (path,))?
            .extract()?;
        let text = std::fs::read_to_string(&path_str)
            .map_err(|e| config_error(format!("failed to read config {path_str}: {e}")))?;
        Self::from_json(cls, py, &text)
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
