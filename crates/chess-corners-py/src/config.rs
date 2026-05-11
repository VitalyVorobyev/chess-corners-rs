//! Native PyO3 wrappers around `chess-corners` config structs.
//!
//! Each wrapper owns a Rust value (or `Py<...>` handles to nested
//! wrappers, in the case of compound types). Python-side users get
//! attribute access, chained mutation, classmethod factories, and
//! `to_dict`/`to_json`/`pretty`/`print` helpers identical to the
//! prior pure-Python dataclass surface — but the FFI no longer
//! serializes through JSON: `Detector(cfg).detect(image)` accepts
//! the typed [`ChessConfig`] wrapper directly.
//!
//! The Rust source-of-truth structs in `chess-corners` /
//! `chess-corners-core` are unchanged; only the binding layer adds
//! these wrappers, preserving the workspace dependency rule that
//! core crates do not depend on `pyo3`.

use std::collections::BTreeSet;

use chess_corners::{
    CenterOfMassConfig as RsCenterOfMassConfig, ChessConfig as RsChessConfig,
    ChessRing as RsChessRing, ChessStrategy as RsChessStrategy, DescriptorMode as RsDescriptorMode,
    DetectionStrategy as RsDetectionStrategy, ForstnerConfig as RsForstnerConfig,
    MultiscaleParams as RsMultiscaleParams, OrientationMethod as RsOrientationMethod,
    PeakFitMode as RsPeakFitMode, RadonPeakConfig as RsRadonPeakConfig,
    RadonStrategy as RsRadonStrategy, RefinementMethod as RsRefinementMethod,
    RefinerConfig as RsRefinerConfig, SaddlePointConfig as RsSaddlePointConfig,
    Threshold as RsThreshold, UpscaleConfig as RsUpscaleConfig, UpscaleMode as RsUpscaleMode,
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
    /// Descriptor sampling override.
    DescriptorMode, RsDescriptorMode,
    [
        (FollowDetector, "FOLLOW_DETECTOR", FollowDetector),
        (Canonical, "CANONICAL", Canonical),
        (Broad, "BROAD", Broad),
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

py_enum!(
    /// Pre-pipeline upscale mode.
    /// `DISABLED` — no upscaling (default).
    /// `FIXED` — integer-factor bilinear upscaling ahead of the pyramid.
    UpscaleMode, RsUpscaleMode,
    [
        (Disabled, "DISABLED", Disabled),
        (Fixed, "FIXED", Fixed),
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

fn parse_upscale_mode(value: &str, path: &str) -> PyResult<RsUpscaleMode> {
    parse_enum(
        value,
        path,
        &[
            ("disabled", RsUpscaleMode::Disabled),
            ("fixed", RsUpscaleMode::Fixed),
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

fn descriptor_mode_str(v: RsDescriptorMode) -> &'static str {
    match v {
        RsDescriptorMode::FollowDetector => "follow_detector",
        RsDescriptorMode::Canonical => "canonical",
        RsDescriptorMode::Broad => "broad",
        _ => "follow_detector",
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

fn orientation_method_str(v: RsOrientationMethod) -> &'static str {
    match v {
        RsOrientationMethod::RingFit => "ring_fit",
        RsOrientationMethod::DiskFit => "disk_fit",
        _ => "ring_fit",
    }
}

fn upscale_mode_str(v: RsUpscaleMode) -> &'static str {
    match v {
        RsUpscaleMode::Disabled => "disabled",
        RsUpscaleMode::Fixed => "fixed",
        _ => "disabled",
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
// UpscaleConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct UpscaleConfig {
    pub(crate) inner: RsUpscaleConfig,
}

#[pymethods]
impl UpscaleConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsUpscaleConfig::default(),
        }
    }

    /// Factory for a disabled upscale config (default).
    #[classmethod]
    fn disabled(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RsUpscaleConfig::disabled(),
        }
    }

    /// Factory for a fixed integer-factor upscale (2, 3, or 4).
    #[classmethod]
    fn fixed(_cls: &Bound<'_, PyType>, factor: u32) -> Self {
        Self {
            inner: RsUpscaleConfig::fixed(factor),
        }
    }

    #[getter]
    fn mode(&self) -> UpscaleMode {
        self.inner.mode.into()
    }
    #[setter]
    fn set_mode(&mut self, v: UpscaleMode) {
        self.inner.mode = v.into();
    }

    #[getter]
    fn factor(&self) -> u32 {
        self.inner.factor
    }
    #[setter]
    fn set_factor(&mut self, v: u32) {
        self.inner.factor = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("mode", upscale_mode_str(self.inner.mode))?;
        d.set_item("factor", self.inner.factor)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "upscale")?;
        reject_unknown_keys(&dict, &["mode", "factor"], "upscale")?;
        let mut cfg = Self::new();
        if let Some(s) = extract_string(&dict, "mode", "upscale")? {
            cfg.inner.mode = parse_upscale_mode(&s, "upscale.mode")?;
        }
        if let Some(v) = extract_int(&dict, "factor", "upscale")? {
            cfg.inner.factor = v as u32;
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
// Threshold (variant-tagged wrapper — PyO3 enums can't carry payloads).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum ThresholdKind {
    Absolute,
    Relative,
}

impl ThresholdKind {
    fn as_str(&self) -> &'static str {
        match self {
            ThresholdKind::Absolute => "absolute",
            ThresholdKind::Relative => "relative",
        }
    }
}

/// Detector acceptance threshold. One of `absolute(value)` or
/// `relative(frac)`. Both detectors honour the same enum.
#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Threshold {
    kind: ThresholdKind,
    value: f32,
}

impl Threshold {
    fn from_rs(v: RsThreshold) -> Self {
        match v {
            RsThreshold::Absolute(v) => Self {
                kind: ThresholdKind::Absolute,
                value: v,
            },
            RsThreshold::Relative(v) => Self {
                kind: ThresholdKind::Relative,
                value: v,
            },
            _ => Self {
                kind: ThresholdKind::Absolute,
                value: 0.0,
            },
        }
    }

    fn to_rs(&self) -> RsThreshold {
        match self.kind {
            ThresholdKind::Absolute => RsThreshold::Absolute(self.value),
            ThresholdKind::Relative => RsThreshold::Relative(self.value),
        }
    }
}

#[pymethods]
impl Threshold {
    /// Default-construct: absolute threshold at 0.0 (paper's `R > 0`).
    #[new]
    fn new() -> Self {
        Self::from_rs(RsThreshold::default())
    }

    /// Build an absolute threshold: accept responses ≥ `value` in the
    /// detector's native score units.
    #[classmethod]
    fn absolute(_cls: &Bound<'_, PyType>, value: f32) -> Self {
        Self {
            kind: ThresholdKind::Absolute,
            value,
        }
    }

    /// Build a relative threshold: accept responses ≥ `frac · max(response)`
    /// in the current frame. `frac` is a fraction in `[0.0, 1.0]`.
    #[classmethod]
    fn relative(_cls: &Bound<'_, PyType>, frac: f32) -> Self {
        Self {
            kind: ThresholdKind::Relative,
            value: frac,
        }
    }

    /// Variant tag: `"absolute"` or `"relative"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Numeric payload of the active variant.
    #[getter]
    fn value(&self) -> f32 {
        self.value
    }
    #[setter]
    fn set_value(&mut self, v: f32) {
        self.value = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item(self.kind.as_str(), self.value as f64)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "threshold")?;
        reject_unknown_keys(&dict, &["absolute", "relative"], "threshold")?;
        // Externally-tagged: exactly one of {absolute, relative} present.
        let mut found: Option<(ThresholdKind, f32)> = None;
        for (key, kind) in [
            ("absolute", ThresholdKind::Absolute),
            ("relative", ThresholdKind::Relative),
        ] {
            if let Some(value) = extract_float(&dict, key, "threshold")? {
                if found.is_some() {
                    return Err(config_error(
                        "threshold must have exactly one of: absolute, relative",
                    ));
                }
                found = Some((kind, value as f32));
            }
        }
        let Some((kind, value)) = found else {
            return Err(config_error(
                "threshold must have one of: absolute, relative",
            ));
        };
        Ok(Self { kind, value })
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
// MultiscaleParams
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct MultiscaleParams {
    pub(crate) inner: RsMultiscaleParams,
}

#[pymethods]
impl MultiscaleParams {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMultiscaleParams::default(),
        }
    }

    #[getter]
    fn pyramid_levels(&self) -> u8 {
        self.inner.pyramid_levels
    }
    #[setter]
    fn set_pyramid_levels(&mut self, v: u8) {
        self.inner.pyramid_levels = v;
    }

    #[getter]
    fn pyramid_min_size(&self) -> usize {
        self.inner.pyramid_min_size
    }
    #[setter]
    fn set_pyramid_min_size(&mut self, v: usize) {
        self.inner.pyramid_min_size = v;
    }

    #[getter]
    fn refinement_radius(&self) -> u32 {
        self.inner.refinement_radius
    }
    #[setter]
    fn set_refinement_radius(&mut self, v: u32) {
        self.inner.refinement_radius = v;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("pyramid_levels", self.inner.pyramid_levels)?;
        d.set_item("pyramid_min_size", self.inner.pyramid_min_size)?;
        d.set_item("refinement_radius", self.inner.refinement_radius)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "multiscale")?;
        reject_unknown_keys(
            &dict,
            &["pyramid_levels", "pyramid_min_size", "refinement_radius"],
            "multiscale",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "pyramid_levels", "multiscale")? {
            cfg.inner.pyramid_levels = v as u8;
        }
        if let Some(v) = extract_int(&dict, "pyramid_min_size", "multiscale")? {
            cfg.inner.pyramid_min_size = v as usize;
        }
        if let Some(v) = extract_int(&dict, "refinement_radius", "multiscale")? {
            cfg.inner.refinement_radius = v as u32;
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
// ChessStrategy
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct ChessStrategy {
    pub(crate) ring: RsChessRing,
    pub(crate) nms_radius: u32,
    pub(crate) min_cluster_size: u32,
}

impl ChessStrategy {
    fn from_rs(_py: Python<'_>, v: RsChessStrategy) -> PyResult<Self> {
        Ok(Self {
            ring: v.ring,
            nms_radius: v.nms_radius,
            min_cluster_size: v.min_cluster_size,
        })
    }

    pub(crate) fn to_rs(&self, _py: Python<'_>) -> RsChessStrategy {
        let mut s = RsChessStrategy::default();
        s.ring = self.ring;
        s.nms_radius = self.nms_radius;
        s.min_cluster_size = self.min_cluster_size;
        s
    }
}

#[pymethods]
impl ChessStrategy {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessStrategy::default())
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ring", chess_ring_str(self.ring))?;
        d.set_item("nms_radius", self.nms_radius)?;
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dict = require_dict(data, "chess_strategy")?;
        reject_unknown_keys(
            &dict,
            &["ring", "nms_radius", "min_cluster_size"],
            "chess_strategy",
        )?;
        let mut cfg = Self::from_rs(py, RsChessStrategy::default())?;
        if let Some(s) = extract_string(&dict, "ring", "chess_strategy")? {
            cfg.ring = parse_chess_ring(&s, "chess_strategy.ring")?;
        }
        if let Some(v) = extract_int(&dict, "nms_radius", "chess_strategy")? {
            cfg.nms_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "min_cluster_size", "chess_strategy")? {
            cfg.min_cluster_size = v as u32;
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
// RadonStrategy
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct RadonStrategy {
    pub(crate) inner: RsRadonStrategy,
}

#[pymethods]
impl RadonStrategy {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsRadonStrategy::default(),
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
        d.set_item("nms_radius", self.inner.nms_radius)?;
        d.set_item("min_cluster_size", self.inner.min_cluster_size)?;
        Ok(d.unbind())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = require_dict(data, "radon_strategy")?;
        reject_unknown_keys(
            &dict,
            &[
                "ray_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "nms_radius",
                "min_cluster_size",
            ],
            "radon_strategy",
        )?;
        let mut cfg = Self::new();
        if let Some(v) = extract_int(&dict, "ray_radius", "radon_strategy")? {
            cfg.inner.ray_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "image_upsample", "radon_strategy")? {
            cfg.inner.image_upsample = v as u32;
        }
        if let Some(v) = extract_int(&dict, "response_blur_radius", "radon_strategy")? {
            cfg.inner.response_blur_radius = v as u32;
        }
        if let Some(s) = extract_string(&dict, "peak_fit", "radon_strategy")? {
            cfg.inner.peak_fit = parse_peak_fit_mode(&s, "radon_strategy.peak_fit")?;
        }
        if let Some(v) = extract_int(&dict, "nms_radius", "radon_strategy")? {
            cfg.inner.nms_radius = v as u32;
        }
        if let Some(v) = extract_int(&dict, "min_cluster_size", "radon_strategy")? {
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
// DetectionStrategy (variant-tagged wrapper — PyO3 enums can't carry payloads).
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

/// Top-level detector dispatch. One of `chess(ChessStrategy)` or
/// `radon(RadonStrategy)`. Carries all detector-specific tuning;
/// settings that don't apply to the active detector are unreachable.
#[pyclass(module = "chess_corners")]
pub struct DetectionStrategy {
    kind: StrategyKind,
    chess: Py<ChessStrategy>,
    radon: Py<RadonStrategy>,
}

impl DetectionStrategy {
    fn from_rs(py: Python<'_>, v: RsDetectionStrategy) -> PyResult<Self> {
        match v {
            RsDetectionStrategy::Chess(c) => Ok(Self {
                kind: StrategyKind::Chess,
                chess: Py::new(py, ChessStrategy::from_rs(py, c)?)?,
                radon: Py::new(py, RadonStrategy::new())?,
            }),
            RsDetectionStrategy::Radon(r) => Ok(Self {
                kind: StrategyKind::Radon,
                chess: Py::new(py, ChessStrategy::from_rs(py, RsChessStrategy::default())?)?,
                radon: Py::new(py, RadonStrategy { inner: r })?,
            }),
            _ => Self::from_rs(py, RsDetectionStrategy::default()),
        }
    }

    pub(crate) fn to_rs(&self, py: Python<'_>) -> RsDetectionStrategy {
        match self.kind {
            StrategyKind::Chess => RsDetectionStrategy::Chess(self.chess.borrow(py).to_rs(py)),
            StrategyKind::Radon => RsDetectionStrategy::Radon(self.radon.borrow(py).inner),
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

    /// Build a ChESS-strategy dispatch carrying the given
    /// [`ChessStrategy`]. Mirrors the Rust `DetectionStrategy::Chess`
    /// variant; the factory name avoids colliding with the `.chess`
    /// property accessor.
    #[classmethod]
    fn from_chess(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        strategy: Py<ChessStrategy>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: StrategyKind::Chess,
            chess: strategy,
            radon: Py::new(py, RadonStrategy::new())?,
        })
    }

    /// Build a Radon-strategy dispatch carrying the given
    /// [`RadonStrategy`]. Mirrors the Rust `DetectionStrategy::Radon`
    /// variant; the factory name avoids colliding with the `.radon`
    /// property accessor.
    #[classmethod]
    fn from_radon(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        strategy: Py<RadonStrategy>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: StrategyKind::Radon,
            chess: Py::new(py, ChessStrategy::from_rs(py, RsChessStrategy::default())?)?,
            radon: strategy,
        })
    }

    /// Variant tag: `"chess"` or `"radon"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// The carried `ChessStrategy` when the active variant is `chess`,
    /// otherwise `None`.
    #[getter]
    fn chess(&self, py: Python<'_>) -> Option<Py<ChessStrategy>> {
        match self.kind {
            StrategyKind::Chess => Some(self.chess.clone_ref(py)),
            _ => None,
        }
    }

    /// The carried `RadonStrategy` when the active variant is `radon`,
    /// otherwise `None`.
    #[getter]
    fn radon(&self, py: Python<'_>) -> Option<Py<RadonStrategy>> {
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
            let value = dict.get_item("chess")?.unwrap();
            let chess_type = py.get_type::<ChessStrategy>();
            let chess = Py::new(py, ChessStrategy::from_dict(&chess_type, py, &value)?)?;
            return Ok(Self {
                kind: StrategyKind::Chess,
                chess,
                radon: Py::new(py, RadonStrategy::new())?,
            });
        }
        if has_radon {
            let value = dict.get_item("radon")?.unwrap();
            let radon_type = py.get_type::<RadonStrategy>();
            let radon = Py::new(py, RadonStrategy::from_dict(&radon_type, &value)?)?;
            return Ok(Self {
                kind: StrategyKind::Radon,
                chess: Py::new(py, ChessStrategy::from_rs(py, RsChessStrategy::default())?)?,
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
// DetectorConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct DetectorConfig {
    pub(crate) strategy: Py<DetectionStrategy>,
    pub(crate) threshold: Py<Threshold>,
    pub(crate) multiscale: Option<Py<MultiscaleParams>>,
    pub(crate) refiner: Py<RefinerConfig>,
    pub(crate) orientation_method: RsOrientationMethod,
    pub(crate) descriptor_mode: RsDescriptorMode,
    pub(crate) upscale: Py<UpscaleConfig>,
    pub(crate) merge_radius: f32,
}

impl DetectorConfig {
    fn from_rs(py: Python<'_>, src: RsChessConfig) -> PyResult<Self> {
        let multiscale = match src.multiscale {
            Some(ms) => Some(Py::new(py, MultiscaleParams { inner: ms })?),
            None => None,
        };
        Ok(Self {
            strategy: Py::new(py, DetectionStrategy::from_rs(py, src.strategy)?)?,
            threshold: Py::new(py, Threshold::from_rs(src.threshold))?,
            multiscale,
            refiner: Py::new(
                py,
                RefinerConfig {
                    kind: src.refiner.kind,
                    center_of_mass: Py::new(
                        py,
                        CenterOfMassConfig {
                            inner: src.refiner.center_of_mass,
                        },
                    )?,
                    forstner: Py::new(
                        py,
                        ForstnerConfig {
                            inner: src.refiner.forstner,
                        },
                    )?,
                    saddle_point: Py::new(
                        py,
                        SaddlePointConfig {
                            inner: src.refiner.saddle_point,
                        },
                    )?,
                    radon_peak: Py::new(
                        py,
                        RadonPeakConfig {
                            inner: src.refiner.radon_peak,
                        },
                    )?,
                },
            )?,
            orientation_method: src.orientation_method,
            descriptor_mode: src.descriptor_mode,
            upscale: Py::new(py, UpscaleConfig { inner: src.upscale })?,
            merge_radius: src.merge_radius,
        })
    }

    fn build(py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::default())
    }

    /// Convert into the Rust facade's `DetectorConfig` (consuming reads
    /// of the nested Python wrappers).
    pub(crate) fn to_inner(&self, py: Python<'_>) -> RsChessConfig {
        let mut cfg = RsChessConfig::default();
        cfg.strategy = self.strategy.borrow(py).to_rs(py);
        cfg.threshold = self.threshold.borrow(py).to_rs();
        cfg.multiscale = self.multiscale.as_ref().map(|m| m.borrow(py).inner);
        cfg.refiner = self.refiner.borrow(py).to_inner(py);
        cfg.orientation_method = self.orientation_method;
        cfg.descriptor_mode = self.descriptor_mode;
        cfg.upscale = self.upscale.borrow(py).inner;
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
    fn single_scale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::single_scale())
    }

    /// Three-level coarse-to-fine ChESS preset.
    #[classmethod]
    fn multiscale_preset(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::multiscale())
    }

    /// Whole-image Radon detector preset.
    #[classmethod]
    fn radon(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::radon())
    }

    /// Coarse-to-fine Radon preset.
    #[classmethod]
    fn radon_multiscale(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        Self::from_rs(py, RsChessConfig::radon_multiscale())
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
    fn threshold(&self, py: Python<'_>) -> Py<Threshold> {
        self.threshold.clone_ref(py)
    }
    #[setter]
    fn set_threshold(&mut self, v: Py<Threshold>) {
        self.threshold = v;
    }

    #[getter]
    fn multiscale(&self, py: Python<'_>) -> Option<Py<MultiscaleParams>> {
        self.multiscale.as_ref().map(|m| m.clone_ref(py))
    }
    #[setter]
    fn set_multiscale(&mut self, v: Option<Py<MultiscaleParams>>) {
        self.multiscale = v;
    }

    #[getter]
    fn refiner(&self, py: Python<'_>) -> Py<RefinerConfig> {
        self.refiner.clone_ref(py)
    }
    #[setter]
    fn set_refiner(&mut self, v: Py<RefinerConfig>) {
        self.refiner = v;
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
    fn orientation_method(&self) -> OrientationMethod {
        self.orientation_method.into()
    }
    #[setter]
    fn set_orientation_method(&mut self, v: OrientationMethod) {
        self.orientation_method = v.into();
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
        d.set_item("threshold", self.threshold.borrow(py).to_dict(py)?)?;
        match &self.multiscale {
            Some(ms) => d.set_item("multiscale", ms.borrow(py).to_dict(py)?)?,
            None => d.set_item("multiscale", py.None())?,
        }
        d.set_item("refiner", self.refiner.borrow(py).to_dict(py)?)?;
        d.set_item(
            "orientation_method",
            orientation_method_str(self.orientation_method),
        )?;
        d.set_item("descriptor_mode", descriptor_mode_str(self.descriptor_mode))?;
        d.set_item("upscale", self.upscale.borrow(py).to_dict(py)?)?;
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
        reject_unknown_keys(
            &dict,
            &[
                "strategy",
                "threshold",
                "multiscale",
                "refiner",
                "orientation_method",
                "descriptor_mode",
                "upscale",
                "merge_radius",
            ],
            "config",
        )?;
        let mut cfg = Self::build(py)?;
        if let Some(value) = dict.get_item("strategy")? {
            let strategy_type = py.get_type::<DetectionStrategy>();
            cfg.strategy = Py::new(
                py,
                DetectionStrategy::from_dict(&strategy_type, py, &value)?,
            )?;
        }
        if let Some(value) = dict.get_item("threshold")? {
            let threshold_type = py.get_type::<Threshold>();
            cfg.threshold = Py::new(py, Threshold::from_dict(&threshold_type, &value)?)?;
        }
        if let Some(value) = dict.get_item("multiscale")? {
            if value.is_none() {
                cfg.multiscale = None;
            } else {
                let ms_type = py.get_type::<MultiscaleParams>();
                cfg.multiscale = Some(Py::new(py, MultiscaleParams::from_dict(&ms_type, &value)?)?);
            }
        }
        if let Some(value) = dict.get_item("refiner")? {
            let refiner_type = py.get_type::<RefinerConfig>();
            cfg.refiner = Py::new(py, RefinerConfig::from_dict(&refiner_type, py, &value)?)?;
        }
        if let Some(s) = extract_string(&dict, "orientation_method", "config")? {
            cfg.orientation_method = parse_orientation_method(&s, "config.orientation_method")?;
        }
        if let Some(s) = extract_string(&dict, "descriptor_mode", "config")? {
            cfg.descriptor_mode = parse_descriptor_mode(&s, "config.descriptor_mode")?;
        }
        if let Some(value) = dict.get_item("upscale")? {
            let upscale_type = py.get_type::<UpscaleConfig>();
            cfg.upscale = Py::new(py, UpscaleConfig::from_dict(&upscale_type, &value)?)?;
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
