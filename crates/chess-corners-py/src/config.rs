//! Native PyO3 wrappers around `chess-corners` config structs.
//!
//! Each wrapper owns a Rust value (or `Py<...>` handles to nested
//! wrappers, in the case of compound types). Python-side users get
//! attribute access, chained mutation, classmethod factories, and
//! `to_dict`/`to_json`/`pretty`/`print` helpers identical to the
//! prior pure-Python dataclass surface — but the FFI no longer
//! serializes through JSON: `find_chess_corners(image, cfg)` accepts
//! the typed [`ChessConfig`] wrapper directly.
//!
//! The Rust source-of-truth structs in `chess-corners` /
//! `chess-corners-core` are unchanged; only the binding layer adds
//! these wrappers, preserving the workspace dependency rule that
//! core crates do not depend on `pyo3`.

use std::collections::BTreeSet;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessConfig as RsChessConfig,
    DescriptorMode as RsDescriptorMode, DetectorMode as RsDetectorMode,
    ForstnerConfig as RsForstnerConfig, PeakFitMode as RsPeakFitMode,
    RadonDetectorParams as RsRadonDetectorParams, RadonPeakConfig as RsRadonPeakConfig,
    RefinementMethod as RsRefinementMethod, RefinerConfig as RsRefinerConfig,
    SaddlePointConfig as RsSaddlePointConfig, ThresholdMode as RsThresholdMode,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
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
    /// Detector kernel selection.
    DetectorMode, RsDetectorMode,
    [
        (Canonical, "CANONICAL", Canonical),
        (Broad, "BROAD", Broad),
        (Radon, "RADON", Radon),
    ]
);

py_enum!(
    /// Descriptor sampling override.
    DescriptorMode, RsDescriptorMode,
    [
        (FollowDetector, "FOLLOW_DETECTOR", FollowDetector),
        (Canonical, "CANONICAL", Canonical),
        (Broad, "BROAD", Broad),
    ]
);

py_enum!(
    /// Threshold interpretation for candidate selection.
    ThresholdMode, RsThresholdMode,
    [
        (Relative, "RELATIVE", Relative),
        (Absolute, "ABSOLUTE", Absolute),
    ]
);

py_enum!(
    /// Subpixel refinement algorithm applied to each candidate.
    RefinementMethod, RsRefinementMethod,
    [
        (CenterOfMass, "CENTER_OF_MASS", CenterOfMass),
        (Forstner, "FORSTNER", Forstner),
        (SaddlePoint, "SADDLE_POINT", SaddlePoint),
        (RadonPeak, "RADON_PEAK", RadonPeak),
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

fn parse_detector_mode(value: &str, path: &str) -> PyResult<RsDetectorMode> {
    parse_enum(
        value,
        path,
        &[
            ("canonical", RsDetectorMode::Canonical),
            ("broad", RsDetectorMode::Broad),
            ("radon", RsDetectorMode::Radon),
        ],
    )
}

fn parse_descriptor_mode(value: &str, path: &str) -> PyResult<RsDescriptorMode> {
    parse_enum(
        value,
        path,
        &[
            ("follow_detector", RsDescriptorMode::FollowDetector),
            ("canonical", RsDescriptorMode::Canonical),
            ("broad", RsDescriptorMode::Broad),
        ],
    )
}

fn parse_threshold_mode(value: &str, path: &str) -> PyResult<RsThresholdMode> {
    parse_enum(
        value,
        path,
        &[
            ("relative", RsThresholdMode::Relative),
            ("absolute", RsThresholdMode::Absolute),
        ],
    )
}

fn parse_refinement_method(value: &str, path: &str) -> PyResult<RsRefinementMethod> {
    parse_enum(
        value,
        path,
        &[
            ("center_of_mass", RsRefinementMethod::CenterOfMass),
            ("forstner", RsRefinementMethod::Forstner),
            ("saddle_point", RsRefinementMethod::SaddlePoint),
            ("radon_peak", RsRefinementMethod::RadonPeak),
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

fn detector_mode_str(v: RsDetectorMode) -> &'static str {
    match v {
        RsDetectorMode::Canonical => "canonical",
        RsDetectorMode::Broad => "broad",
        RsDetectorMode::Radon => "radon",
        _ => "canonical",
    }
}

fn descriptor_mode_str(v: RsDescriptorMode) -> &'static str {
    match v {
        RsDescriptorMode::FollowDetector => "follow_detector",
        RsDescriptorMode::Canonical => "canonical",
        RsDescriptorMode::Broad => "broad",
        _ => "follow_detector",
    }
}

fn threshold_mode_str(v: RsThresholdMode) -> &'static str {
    match v {
        RsThresholdMode::Relative => "relative",
        RsThresholdMode::Absolute => "absolute",
        _ => "absolute",
    }
}

fn refinement_method_str(v: RsRefinementMethod) -> &'static str {
    match v {
        RsRefinementMethod::CenterOfMass => "center_of_mass",
        RsRefinementMethod::Forstner => "forstner",
        RsRefinementMethod::SaddlePoint => "saddle_point",
        RsRefinementMethod::RadonPeak => "radon_peak",
        _ => "center_of_mass",
    }
}

fn peak_fit_mode_str(v: RsPeakFitMode) -> &'static str {
    match v {
        RsPeakFitMode::Parabolic => "parabolic",
        RsPeakFitMode::Gaussian => "gaussian",
        _ => "gaussian",
    }
}

// ---------------------------------------------------------------------------
// Shared dict / json / pretty / print methods.
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
// RadonDetectorParams
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct RadonDetectorParams {
    pub(crate) inner: RsRadonDetectorParams,
}

#[pymethods]
impl RadonDetectorParams {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsRadonDetectorParams::default(),
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
    fn threshold_rel(&self) -> f32 {
        self.inner.threshold_rel
    }
    #[setter]
    fn set_threshold_rel(&mut self, v: f32) {
        self.inner.threshold_rel = v;
    }

    #[getter]
    fn threshold_abs(&self) -> Option<f32> {
        self.inner.threshold_abs
    }
    #[setter]
    fn set_threshold_abs(&mut self, v: Option<f32>) {
        self.inner.threshold_abs = v;
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
        d.set_item("ray_radius", self.inner.ray_radius)?;
        d.set_item("image_upsample", self.inner.image_upsample)?;
        d.set_item("response_blur_radius", self.inner.response_blur_radius)?;
        d.set_item("peak_fit", peak_fit_mode_str(self.inner.peak_fit))?;
        d.set_item("threshold_rel", self.inner.threshold_rel as f64)?;
        d.set_item("threshold_abs", self.inner.threshold_abs.map(|x| x as f64))?;
        d.set_item("nms_radius", self.inner.nms_radius)?;
        d.set_item("min_cluster_size", self.inner.min_cluster_size)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "radon_detector")?;
        reject_unknown_keys(
            &dict,
            &[
                "ray_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "threshold_rel",
                "threshold_abs",
                "nms_radius",
                "min_cluster_size",
            ],
            "radon_detector",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "ray_radius", "radon_detector")? {
            cfg.inner.ray_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "image_upsample", "radon_detector")? {
            cfg.inner.image_upsample = v as u32;
        }
        if let Some(v) = extract_int(&dict, "response_blur_radius", "radon_detector")? {
            cfg.inner.response_blur_radius = v as u32;
        }
        if let Some(s) = extract_string(&dict, "peak_fit", "radon_detector")? {
            cfg.inner.peak_fit = parse_peak_fit_mode(&s, "radon_detector.peak_fit")?;
        }
        if let Some(v) = extract_float(&dict, "threshold_rel", "radon_detector")? {
            cfg.inner.threshold_rel = v as f32;
        }
        if let Some(value) = dict.get_item("threshold_abs")? {
            if value.is_none() {
                cfg.inner.threshold_abs = None;
            } else if value.is_instance_of::<PyBool>()
                || !(value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
            {
                return Err(config_error(
                    "radon_detector.threshold_abs must be a number or null",
                ));
            } else {
                cfg.inner.threshold_abs = Some(value.extract::<f64>()? as f32);
            }
        }
        if let Some(v) = extract_int(&dict, "nms_radius", "radon_detector")? {
            cfg.inner.nms_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "min_cluster_size", "radon_detector")? {
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
// RefinerConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct RefinerConfig {
    /// `kind` is stored on the inner Rust value because it's a scalar
    /// enum. Per-variant configs are stored as Python objects so that
    /// chained mutation (`cfg.refiner.forstner.max_offset = 2.0`)
    /// behaves like real reference semantics — the same Python
    /// object is returned by every getter call.
    pub(crate) kind: RsRefinementMethod,
    pub(crate) center_of_mass: Py<CenterOfMassConfig>,
    pub(crate) forstner: Py<ForstnerConfig>,
    pub(crate) saddle_point: Py<SaddlePointConfig>,
    pub(crate) radon_peak: Py<RadonPeakConfig>,
}

impl RefinerConfig {
    pub(crate) fn build(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            kind: RsRefinementMethod::default(),
            center_of_mass: Py::new(py, CenterOfMassConfig::new())?,
            forstner: Py::new(py, ForstnerConfig::new())?,
            saddle_point: Py::new(py, SaddlePointConfig::new())?,
            radon_peak: Py::new(py, RadonPeakConfig::new())?,
        })
    }

    pub(crate) fn to_inner(&self, py: Python<'_>) -> RsRefinerConfig {
        RsRefinerConfig::build(
            self.kind,
            self.center_of_mass.borrow(py).inner,
            self.forstner.borrow(py).inner,
            self.saddle_point.borrow(py).inner,
            self.radon_peak.borrow(py).inner,
        )
    }
}

#[pymethods]
impl RefinerConfig {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::build(py)
    }

    #[classmethod]
    #[pyo3(name = "center_of_mass_config")]
    fn center_of_mass_config(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.kind = RsRefinementMethod::CenterOfMass;
        Ok(cfg)
    }
    #[classmethod]
    #[pyo3(name = "forstner_config")]
    fn forstner_config(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.kind = RsRefinementMethod::Forstner;
        Ok(cfg)
    }
    #[classmethod]
    #[pyo3(name = "saddle_point_config")]
    fn saddle_point_config(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.kind = RsRefinementMethod::SaddlePoint;
        Ok(cfg)
    }
    #[classmethod]
    #[pyo3(name = "radon_peak_config")]
    fn radon_peak_config(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.kind = RsRefinementMethod::RadonPeak;
        Ok(cfg)
    }

    #[getter]
    fn kind(&self) -> RefinementMethod {
        self.kind.into()
    }
    #[setter]
    fn set_kind(&mut self, v: RefinementMethod) {
        self.kind = v.into();
    }

    #[getter]
    fn center_of_mass(&self, py: Python<'_>) -> Py<CenterOfMassConfig> {
        self.center_of_mass.clone_ref(py)
    }
    #[setter]
    fn set_center_of_mass(&mut self, v: Py<CenterOfMassConfig>) {
        self.center_of_mass = v;
    }

    #[getter]
    fn forstner(&self, py: Python<'_>) -> Py<ForstnerConfig> {
        self.forstner.clone_ref(py)
    }
    #[setter]
    fn set_forstner(&mut self, v: Py<ForstnerConfig>) {
        self.forstner = v;
    }

    #[getter]
    fn saddle_point(&self, py: Python<'_>) -> Py<SaddlePointConfig> {
        self.saddle_point.clone_ref(py)
    }
    #[setter]
    fn set_saddle_point(&mut self, v: Py<SaddlePointConfig>) {
        self.saddle_point = v;
    }

    #[getter]
    fn radon_peak(&self, py: Python<'_>) -> Py<RadonPeakConfig> {
        self.radon_peak.clone_ref(py)
    }
    #[setter]
    fn set_radon_peak(&mut self, v: Py<RadonPeakConfig>) {
        self.radon_peak = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("kind", refinement_method_str(self.kind))?;
        d.set_item(
            "center_of_mass",
            self.center_of_mass.borrow(py).to_dict(py)?,
        )?;
        d.set_item("forstner", self.forstner.borrow(py).to_dict(py)?)?;
        d.set_item("saddle_point", self.saddle_point.borrow(py).to_dict(py)?)?;
        d.set_item("radon_peak", self.radon_peak.borrow(py).to_dict(py)?)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "refiner")?;
        reject_unknown_keys(
            &dict,
            &[
                "kind",
                "center_of_mass",
                "forstner",
                "saddle_point",
                "radon_peak",
            ],
            "refiner",
        )?;
        let mut cfg = Self::build(py)?;
        if let Some(s) = extract_string(&dict, "kind", "refiner")? {
            cfg.kind = parse_refinement_method(&s, "refiner.kind")?;
        }
        let com_type = py.get_type::<CenterOfMassConfig>();
        let forstner_type = py.get_type::<ForstnerConfig>();
        let saddle_type = py.get_type::<SaddlePointConfig>();
        let radon_type = py.get_type::<RadonPeakConfig>();
        if let Some(value) = dict.get_item("center_of_mass")? {
            cfg.center_of_mass = Py::new(py, CenterOfMassConfig::from_dict(&com_type, &value)?)?;
        }
        if let Some(value) = dict.get_item("forstner")? {
            cfg.forstner = Py::new(py, ForstnerConfig::from_dict(&forstner_type, &value)?)?;
        }
        if let Some(value) = dict.get_item("saddle_point")? {
            cfg.saddle_point = Py::new(py, SaddlePointConfig::from_dict(&saddle_type, &value)?)?;
        }
        if let Some(value) = dict.get_item("radon_peak")? {
            cfg.radon_peak = Py::new(py, RadonPeakConfig::from_dict(&radon_type, &value)?)?;
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
// ChessConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct ChessConfig {
    pub(crate) detector_mode: RsDetectorMode,
    pub(crate) descriptor_mode: RsDescriptorMode,
    pub(crate) threshold_mode: RsThresholdMode,
    pub(crate) threshold_value: f32,
    pub(crate) nms_radius: u32,
    pub(crate) min_cluster_size: u32,
    pub(crate) refiner: Py<RefinerConfig>,
    pub(crate) pyramid_levels: u8,
    pub(crate) pyramid_min_size: usize,
    pub(crate) refinement_radius: u32,
    pub(crate) merge_radius: f32,
    pub(crate) radon_detector: Py<RadonDetectorParams>,
}

impl ChessConfig {
    fn build(py: Python<'_>) -> PyResult<Self> {
        let defaults = RsChessConfig::default();
        Ok(Self {
            detector_mode: defaults.detector_mode,
            descriptor_mode: defaults.descriptor_mode,
            // Match the prior Python dataclass defaults (Relative/0.2)
            // rather than the Rust default (Absolute/0.0). Existing
            // Python tests assume the relative-fraction default.
            threshold_mode: RsThresholdMode::Relative,
            threshold_value: 0.2,
            nms_radius: defaults.nms_radius,
            min_cluster_size: defaults.min_cluster_size,
            refiner: Py::new(py, RefinerConfig::build(py)?)?,
            pyramid_levels: defaults.pyramid_levels,
            pyramid_min_size: defaults.pyramid_min_size,
            refinement_radius: defaults.refinement_radius,
            merge_radius: defaults.merge_radius,
            radon_detector: Py::new(py, RadonDetectorParams::new())?,
        })
    }

    /// Convert into the Rust facade's `ChessConfig` (consuming reads
    /// of the nested Python wrappers).
    pub(crate) fn to_inner(&self, py: Python<'_>) -> RsChessConfig {
        let mut cfg = RsChessConfig::default();
        cfg.detector_mode = self.detector_mode;
        cfg.descriptor_mode = self.descriptor_mode;
        cfg.threshold_mode = self.threshold_mode;
        cfg.threshold_value = self.threshold_value;
        cfg.nms_radius = self.nms_radius;
        cfg.min_cluster_size = self.min_cluster_size;
        cfg.refiner = self.refiner.borrow(py).to_inner(py);
        cfg.pyramid_levels = self.pyramid_levels;
        cfg.pyramid_min_size = self.pyramid_min_size;
        cfg.refinement_radius = self.refinement_radius;
        cfg.merge_radius = self.merge_radius;
        cfg.radon_detector = self.radon_detector.borrow(py).inner;
        cfg
    }
}

#[pymethods]
impl ChessConfig {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::build(py)
    }

    #[classmethod]
    fn single_scale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::build(py)
    }

    #[classmethod]
    fn multiscale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.pyramid_levels = 3;
        cfg.pyramid_min_size = 128;
        Ok(cfg)
    }

    #[classmethod]
    fn radon(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        let mut cfg = Self::build(py)?;
        cfg.detector_mode = RsDetectorMode::Radon;
        cfg.pyramid_levels = 1;
        Ok(cfg)
    }

    // ---- scalar fields ----

    #[getter]
    fn detector_mode(&self) -> DetectorMode {
        self.detector_mode.into()
    }
    #[setter]
    fn set_detector_mode(&mut self, v: DetectorMode) {
        self.detector_mode = v.into();
    }

    #[getter]
    fn descriptor_mode(&self) -> DescriptorMode {
        self.descriptor_mode.into()
    }
    #[setter]
    fn set_descriptor_mode(&mut self, v: DescriptorMode) {
        self.descriptor_mode = v.into();
    }

    #[getter]
    fn threshold_mode(&self) -> ThresholdMode {
        self.threshold_mode.into()
    }
    #[setter]
    fn set_threshold_mode(&mut self, v: ThresholdMode) {
        self.threshold_mode = v.into();
    }

    #[getter]
    fn threshold_value(&self) -> f32 {
        self.threshold_value
    }
    #[setter]
    fn set_threshold_value(&mut self, v: f32) {
        self.threshold_value = v;
    }

    #[getter]
    fn nms_radius(&self) -> u32 {
        self.nms_radius
    }
    #[setter]
    fn set_nms_radius(&mut self, v: u32) {
        self.nms_radius = v;
    }

    #[getter]
    fn min_cluster_size(&self) -> u32 {
        self.min_cluster_size
    }
    #[setter]
    fn set_min_cluster_size(&mut self, v: u32) {
        self.min_cluster_size = v;
    }

    #[getter]
    fn pyramid_levels(&self) -> u8 {
        self.pyramid_levels
    }
    #[setter]
    fn set_pyramid_levels(&mut self, v: u8) {
        self.pyramid_levels = v;
    }

    #[getter]
    fn pyramid_min_size(&self) -> usize {
        self.pyramid_min_size
    }
    #[setter]
    fn set_pyramid_min_size(&mut self, v: usize) {
        self.pyramid_min_size = v;
    }

    #[getter]
    fn refinement_radius(&self) -> u32 {
        self.refinement_radius
    }
    #[setter]
    fn set_refinement_radius(&mut self, v: u32) {
        self.refinement_radius = v;
    }

    #[getter]
    fn merge_radius(&self) -> f32 {
        self.merge_radius
    }
    #[setter]
    fn set_merge_radius(&mut self, v: f32) {
        self.merge_radius = v;
    }

    // ---- nested wrappers (returned by reference) ----

    #[getter]
    fn refiner(&self, py: Python<'_>) -> Py<RefinerConfig> {
        self.refiner.clone_ref(py)
    }
    #[setter]
    fn set_refiner(&mut self, v: Py<RefinerConfig>) {
        self.refiner = v;
    }

    #[getter]
    fn radon_detector(&self, py: Python<'_>) -> Py<RadonDetectorParams> {
        self.radon_detector.clone_ref(py)
    }
    #[setter]
    fn set_radon_detector(&mut self, v: Py<RadonDetectorParams>) {
        self.radon_detector = v;
    }

    // ---- serialization ----

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("detector_mode", detector_mode_str(self.detector_mode))?;
        d.set_item("descriptor_mode", descriptor_mode_str(self.descriptor_mode))?;
        d.set_item("threshold_mode", threshold_mode_str(self.threshold_mode))?;
        d.set_item("threshold_value", self.threshold_value as f64)?;
        d.set_item("nms_radius", self.nms_radius)?;
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        d.set_item("refiner", self.refiner.borrow(py).to_dict(py)?)?;
        d.set_item("pyramid_levels", self.pyramid_levels)?;
        d.set_item("pyramid_min_size", self.pyramid_min_size)?;
        d.set_item("refinement_radius", self.refinement_radius)?;
        d.set_item("merge_radius", self.merge_radius as f64)?;
        d.set_item(
            "radon_detector",
            self.radon_detector.borrow(py).to_dict(py)?,
        )?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "config")?;
        reject_unknown_keys(
            &dict,
            &[
                "detector_mode",
                "descriptor_mode",
                "threshold_mode",
                "threshold_value",
                "nms_radius",
                "min_cluster_size",
                "refiner",
                "pyramid_levels",
                "pyramid_min_size",
                "refinement_radius",
                "merge_radius",
                "radon_detector",
            ],
            "config",
        )?;
        let mut cfg = Self::build(py)?;
        if let Some(s) = extract_string(&dict, "detector_mode", "config")? {
            cfg.detector_mode = parse_detector_mode(&s, "config.detector_mode")?;
        }
        if let Some(s) = extract_string(&dict, "descriptor_mode", "config")? {
            cfg.descriptor_mode = parse_descriptor_mode(&s, "config.descriptor_mode")?;
        }
        if let Some(s) = extract_string(&dict, "threshold_mode", "config")? {
            cfg.threshold_mode = parse_threshold_mode(&s, "config.threshold_mode")?;
        }
        if let Some(v) = extract_float(&dict, "threshold_value", "config")? {
            cfg.threshold_value = v as f32;
        }
        if let Some(v) = extract_int(&dict, "nms_radius", "config")? {
            cfg.nms_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "min_cluster_size", "config")? {
            cfg.min_cluster_size = v as u32;
        }
        if let Some(value) = dict.get_item("refiner")? {
            let refiner_type = py.get_type::<RefinerConfig>();
            cfg.refiner = Py::new(py, RefinerConfig::from_dict(&refiner_type, py, &value)?)?;
        }
        if let Some(v) = extract_int(&dict, "pyramid_levels", "config")? {
            cfg.pyramid_levels = v as u8;
        }
        if let Some(v) = extract_int(&dict, "pyramid_min_size", "config")? {
            cfg.pyramid_min_size = v as usize;
        }
        if let Some(v) = extract_int(&dict, "refinement_radius", "config")? {
            cfg.refinement_radius = v as u32;
        }
        if let Some(v) = extract_float(&dict, "merge_radius", "config")? {
            cfg.merge_radius = v as f32;
        }
        if let Some(value) = dict.get_item("radon_detector")? {
            let radon_type = py.get_type::<RadonDetectorParams>();
            cfg.radon_detector = Py::new(py, RadonDetectorParams::from_dict(&radon_type, &value)?)?;
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
