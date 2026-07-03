//! PyO3 wrapper for the top-level `DetectorConfig`.

use chess_corners::{
    ChessConfig as RsChessConfig, DetectorConfig as RsDetectorConfig,
    OrientationMethod as RsOrientationMethod, RadonConfig as RsRadonConfig,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyType};

use super::multiscale::MultiscaleConfig;
use super::parse::{
    config_error, extract_float, json_dumps, json_loads, orientation_method_str,
    parse_orientation_method, reject_unknown_keys, require_dict, ChessRing, OrientationMethod,
    PeakFitMode,
};
use super::refiners::ChessRefiner;
use super::strategy::{ChessConfig, DetectionParams, DetectionStrategy, RadonConfig, StrategyKind};
use super::upscale::UpscaleConfig;

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
    /// Accepted kwargs: `ray_radius`, `image_upsample`,
    /// `response_blur_radius`, `peak_fit`. The shared NMS / clustering
    /// knobs moved to `with_detection`.
    #[pyo3(signature = (**kwargs))]
    fn with_radon(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        const RADON_FIELDS: &[&str] = &[
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
