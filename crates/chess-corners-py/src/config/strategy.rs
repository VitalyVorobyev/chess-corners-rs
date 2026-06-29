//! PyO3 wrappers for `ChessConfig`, `RadonConfig`, `DetectionParams`,
//! and `DetectionStrategy`.

use chess_corners::{
    ChessConfig as RsChessConfig, ChessRing as RsChessRing, DetectionParams as RsDetectionParams,
    DetectionStrategy as RsDetectionStrategy, PeakFitMode as RsPeakFitMode,
    RadonConfig as RsRadonConfig,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::parse::{
    chess_ring_str, config_error, extract_int, extract_string, json_dumps, json_loads,
    parse_chess_ring, parse_peak_fit_mode, peak_fit_mode_str, reject_unknown_keys, require_dict,
    ChessRing, PeakFitMode,
};
use super::refiners::ChessRefiner;

// ---------------------------------------------------------------------------
// ChessConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "chess_corners")]
pub struct ChessConfig {
    pub(crate) ring: RsChessRing,
    pub(crate) refiner: Py<ChessRefiner>,
}

impl ChessConfig {
    pub(crate) fn from_rs(py: Python<'_>, v: RsChessConfig) -> PyResult<Self> {
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
    pub(crate) fn from_rs(_py: Python<'_>, v: RsRadonConfig) -> PyResult<Self> {
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
    pub(crate) fn from_rs(v: RsDetectionParams) -> Self {
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

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("nms_radius", self.inner.nms_radius)?;
        d.set_item("min_cluster_size", self.inner.min_cluster_size)?;
        Ok(d.unbind())
    }

    #[classmethod]
    pub(crate) fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
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
pub(crate) enum StrategyKind {
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
    pub(crate) kind: StrategyKind,
    pub(crate) chess: Py<ChessConfig>,
    pub(crate) radon: Py<RadonConfig>,
}

impl DetectionStrategy {
    pub(crate) fn from_rs(py: Python<'_>, v: RsDetectionStrategy) -> PyResult<Self> {
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

    pub(crate) fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
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
    pub(crate) fn from_dict(
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
