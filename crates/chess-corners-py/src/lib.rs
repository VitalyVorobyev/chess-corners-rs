use ::chess_corners as chess_corners_rs;
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyModule, PyModuleMethods};

#[pyclass(name = "PyramidParams")]
#[derive(Clone)]
pub struct PyramidParamsPy {
    num_levels: u8,
    min_size: usize,
}

impl PyramidParamsPy {
    fn from_rust(params: &chess_corners_rs::PyramidParams) -> Self {
        Self {
            num_levels: params.num_levels,
            min_size: params.min_size,
        }
    }

    fn to_rust(&self) -> chess_corners_rs::PyramidParams {
        chess_corners_rs::PyramidParams {
            num_levels: self.num_levels,
            min_size: self.min_size,
        }
    }
}

#[pymethods]
impl PyramidParamsPy {
    #[new]
    fn new() -> Self {
        Self::from_rust(&chess_corners_rs::PyramidParams::default())
    }

    #[getter]
    fn num_levels(&self) -> u8 {
        self.num_levels
    }

    #[setter]
    fn set_num_levels(&mut self, value: u8) {
        self.num_levels = value;
    }

    #[getter]
    fn min_size(&self) -> usize {
        self.min_size
    }

    #[setter]
    fn set_min_size(&mut self, value: usize) {
        self.min_size = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("num_levels", self.num_levels())?;
        dict.set_item("min_size", self.min_size())?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "CoarseToFineParams")]
pub struct CoarseToFineParamsPy {
    pyramid: Py<PyramidParamsPy>,
    refinement_radius: u32,
    merge_radius: f32,
}

impl CoarseToFineParamsPy {
    fn from_rust(py: Python<'_>, params: &chess_corners_rs::CoarseToFineParams) -> PyResult<Self> {
        let pyramid = Py::new(py, PyramidParamsPy::from_rust(&params.pyramid))?;
        Ok(Self {
            pyramid,
            refinement_radius: params.refinement_radius,
            merge_radius: params.merge_radius,
        })
    }

    fn to_rust(&self, py: Python<'_>) -> PyResult<chess_corners_rs::CoarseToFineParams> {
        let pyramid = self.pyramid.bind(py).borrow();
        Ok(chess_corners_rs::CoarseToFineParams {
            pyramid: pyramid.to_rust(),
            refinement_radius: self.refinement_radius,
            merge_radius: self.merge_radius,
        })
    }
}

#[pymethods]
impl CoarseToFineParamsPy {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rust(py, &chess_corners_rs::CoarseToFineParams::default())
    }

    #[getter]
    fn pyramid(&self, py: Python<'_>) -> PyResult<Py<PyramidParamsPy>> {
        Ok(self.pyramid.clone_ref(py))
    }

    #[setter]
    fn set_pyramid(&mut self, value: Py<PyramidParamsPy>) {
        self.pyramid = value;
    }

    #[getter]
    fn refinement_radius(&self) -> u32 {
        self.refinement_radius
    }

    #[setter]
    fn set_refinement_radius(&mut self, value: u32) {
        self.refinement_radius = value;
    }

    #[getter]
    fn merge_radius(&self) -> f32 {
        self.merge_radius
    }

    #[setter]
    fn set_merge_radius(&mut self, value: f32) {
        self.merge_radius = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let pyramid = self.pyramid.bind(py).borrow();
        dict.set_item("pyramid", pyramid.to_dict(py)?)?;
        dict.set_item("refinement_radius", self.refinement_radius())?;
        dict.set_item("merge_radius", self.merge_radius())?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "CenterOfMassConfig")]
#[derive(Clone)]
pub struct CenterOfMassConfigPy {
    radius: i32,
}

impl CenterOfMassConfigPy {
    fn from_rust(cfg: &chess_corners_rs::CenterOfMassConfig) -> Self {
        Self { radius: cfg.radius }
    }

    fn to_rust(&self) -> chess_corners_rs::CenterOfMassConfig {
        chess_corners_rs::CenterOfMassConfig {
            radius: self.radius,
        }
    }
}

#[pymethods]
impl CenterOfMassConfigPy {
    #[new]
    fn new() -> Self {
        Self::from_rust(&chess_corners_rs::CenterOfMassConfig::default())
    }

    #[getter]
    fn radius(&self) -> i32 {
        self.radius
    }

    #[setter]
    fn set_radius(&mut self, value: i32) {
        self.radius = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("radius", self.radius())?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "ForstnerConfig")]
#[derive(Clone)]
pub struct ForstnerConfigPy {
    radius: i32,
    min_trace: f32,
    min_det: f32,
    max_condition_number: f32,
    max_offset: f32,
}

impl ForstnerConfigPy {
    fn from_rust(cfg: &chess_corners_rs::ForstnerConfig) -> Self {
        Self {
            radius: cfg.radius,
            min_trace: cfg.min_trace,
            min_det: cfg.min_det,
            max_condition_number: cfg.max_condition_number,
            max_offset: cfg.max_offset,
        }
    }

    fn to_rust(&self) -> chess_corners_rs::ForstnerConfig {
        chess_corners_rs::ForstnerConfig {
            radius: self.radius,
            min_trace: self.min_trace,
            min_det: self.min_det,
            max_condition_number: self.max_condition_number,
            max_offset: self.max_offset,
        }
    }
}

#[pymethods]
impl ForstnerConfigPy {
    #[new]
    fn new() -> Self {
        Self::from_rust(&chess_corners_rs::ForstnerConfig::default())
    }

    #[getter]
    fn radius(&self) -> i32 {
        self.radius
    }

    #[setter]
    fn set_radius(&mut self, value: i32) {
        self.radius = value;
    }

    #[getter]
    fn min_trace(&self) -> f32 {
        self.min_trace
    }

    #[setter]
    fn set_min_trace(&mut self, value: f32) {
        self.min_trace = value;
    }

    #[getter]
    fn min_det(&self) -> f32 {
        self.min_det
    }

    #[setter]
    fn set_min_det(&mut self, value: f32) {
        self.min_det = value;
    }

    #[getter]
    fn max_condition_number(&self) -> f32 {
        self.max_condition_number
    }

    #[setter]
    fn set_max_condition_number(&mut self, value: f32) {
        self.max_condition_number = value;
    }

    #[getter]
    fn max_offset(&self) -> f32 {
        self.max_offset
    }

    #[setter]
    fn set_max_offset(&mut self, value: f32) {
        self.max_offset = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("radius", self.radius())?;
        dict.set_item("min_trace", self.min_trace())?;
        dict.set_item("min_det", self.min_det())?;
        dict.set_item("max_condition_number", self.max_condition_number())?;
        dict.set_item("max_offset", self.max_offset())?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "SaddlePointConfig")]
#[derive(Clone)]
pub struct SaddlePointConfigPy {
    radius: i32,
    det_margin: f32,
    max_offset: f32,
    min_abs_det: f32,
}

impl SaddlePointConfigPy {
    fn from_rust(cfg: &chess_corners_rs::SaddlePointConfig) -> Self {
        Self {
            radius: cfg.radius,
            det_margin: cfg.det_margin,
            max_offset: cfg.max_offset,
            min_abs_det: cfg.min_abs_det,
        }
    }

    fn to_rust(&self) -> chess_corners_rs::SaddlePointConfig {
        chess_corners_rs::SaddlePointConfig {
            radius: self.radius,
            det_margin: self.det_margin,
            max_offset: self.max_offset,
            min_abs_det: self.min_abs_det,
        }
    }
}

#[pymethods]
impl SaddlePointConfigPy {
    #[new]
    fn new() -> Self {
        Self::from_rust(&chess_corners_rs::SaddlePointConfig::default())
    }

    #[getter]
    fn radius(&self) -> i32 {
        self.radius
    }

    #[setter]
    fn set_radius(&mut self, value: i32) {
        self.radius = value;
    }

    #[getter]
    fn det_margin(&self) -> f32 {
        self.det_margin
    }

    #[setter]
    fn set_det_margin(&mut self, value: f32) {
        self.det_margin = value;
    }

    #[getter]
    fn max_offset(&self) -> f32 {
        self.max_offset
    }

    #[setter]
    fn set_max_offset(&mut self, value: f32) {
        self.max_offset = value;
    }

    #[getter]
    fn min_abs_det(&self) -> f32 {
        self.min_abs_det
    }

    #[setter]
    fn set_min_abs_det(&mut self, value: f32) {
        self.min_abs_det = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("radius", self.radius())?;
        dict.set_item("det_margin", self.det_margin())?;
        dict.set_item("max_offset", self.max_offset())?;
        dict.set_item("min_abs_det", self.min_abs_det())?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "RefinerKind")]
#[derive(Clone)]
pub struct RefinerKindPy {
    inner: chess_corners_rs::RefinerKind,
}

impl RefinerKindPy {
    fn from_rust(kind: &chess_corners_rs::RefinerKind) -> Self {
        Self {
            inner: kind.clone(),
        }
    }

    fn to_rust(&self) -> chess_corners_rs::RefinerKind {
        self.inner.clone()
    }
}

#[pymethods]
impl RefinerKindPy {
    #[staticmethod]
    fn center_of_mass(cfg: Option<&CenterOfMassConfigPy>) -> Self {
        let cfg = cfg.map(|c| c.to_rust()).unwrap_or_default();
        Self {
            inner: chess_corners_rs::RefinerKind::CenterOfMass(cfg),
        }
    }

    #[staticmethod]
    fn forstner(cfg: Option<&ForstnerConfigPy>) -> Self {
        let cfg = cfg.map(|c| c.to_rust()).unwrap_or_default();
        Self {
            inner: chess_corners_rs::RefinerKind::Forstner(cfg),
        }
    }

    #[staticmethod]
    fn saddle_point(cfg: Option<&SaddlePointConfigPy>) -> Self {
        let cfg = cfg.map(|c| c.to_rust()).unwrap_or_default();
        Self {
            inner: chess_corners_rs::RefinerKind::SaddlePoint(cfg),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            chess_corners_rs::RefinerKind::CenterOfMass(_) => "center_of_mass",
            chess_corners_rs::RefinerKind::Forstner(_) => "forstner",
            chess_corners_rs::RefinerKind::SaddlePoint(_) => "saddle_point",
        }
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            chess_corners_rs::RefinerKind::CenterOfMass(cfg) => {
                Ok(Py::new(py, CenterOfMassConfigPy::from_rust(cfg))?.into_any())
            }
            chess_corners_rs::RefinerKind::Forstner(cfg) => {
                Ok(Py::new(py, ForstnerConfigPy::from_rust(cfg))?.into_any())
            }
            chess_corners_rs::RefinerKind::SaddlePoint(cfg) => {
                Ok(Py::new(py, SaddlePointConfigPy::from_rust(cfg))?.into_any())
            }
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("kind", self.kind())?;
        dict.set_item("config", self.config(py)?)?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass(name = "ChessParams")]
pub struct ChessParamsPy {
    use_radius10: bool,
    descriptor_use_radius10: Option<bool>,
    threshold_rel: f32,
    threshold_abs: Option<f32>,
    nms_radius: u32,
    min_cluster_size: u32,
    refiner: Py<RefinerKindPy>,
}

impl ChessParamsPy {
    fn from_rust(py: Python<'_>, params: &chess_corners_rs::ChessParams) -> PyResult<Self> {
        let refiner = Py::new(py, RefinerKindPy::from_rust(&params.refiner))?;
        Ok(Self {
            use_radius10: params.use_radius10,
            descriptor_use_radius10: params.descriptor_use_radius10,
            threshold_rel: params.threshold_rel,
            threshold_abs: params.threshold_abs,
            nms_radius: params.nms_radius,
            min_cluster_size: params.min_cluster_size,
            refiner,
        })
    }

    fn to_rust(&self, py: Python<'_>) -> PyResult<chess_corners_rs::ChessParams> {
        let refiner = self.refiner.bind(py).borrow();
        Ok(chess_corners_rs::ChessParams {
            use_radius10: self.use_radius10,
            descriptor_use_radius10: self.descriptor_use_radius10,
            threshold_rel: self.threshold_rel,
            threshold_abs: self.threshold_abs,
            nms_radius: self.nms_radius,
            min_cluster_size: self.min_cluster_size,
            refiner: refiner.to_rust(),
        })
    }
}

#[pymethods]
impl ChessParamsPy {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rust(py, &chess_corners_rs::ChessParams::default())
    }

    #[getter]
    fn use_radius10(&self) -> bool {
        self.use_radius10
    }

    #[setter]
    fn set_use_radius10(&mut self, value: bool) {
        self.use_radius10 = value;
    }

    #[getter]
    fn descriptor_use_radius10(&self) -> Option<bool> {
        self.descriptor_use_radius10
    }

    #[setter]
    fn set_descriptor_use_radius10(&mut self, value: Option<bool>) {
        self.descriptor_use_radius10 = value;
    }

    #[getter]
    fn threshold_rel(&self) -> f32 {
        self.threshold_rel
    }

    #[setter]
    fn set_threshold_rel(&mut self, value: f32) {
        self.threshold_rel = value;
    }

    #[getter]
    fn threshold_abs(&self) -> Option<f32> {
        self.threshold_abs
    }

    #[setter]
    fn set_threshold_abs(&mut self, value: Option<f32>) {
        self.threshold_abs = value;
    }

    #[getter]
    fn nms_radius(&self) -> u32 {
        self.nms_radius
    }

    #[setter]
    fn set_nms_radius(&mut self, value: u32) {
        self.nms_radius = value;
    }

    #[getter]
    fn min_cluster_size(&self) -> u32 {
        self.min_cluster_size
    }

    #[setter]
    fn set_min_cluster_size(&mut self, value: u32) {
        self.min_cluster_size = value;
    }

    #[getter]
    fn refiner(&self, py: Python<'_>) -> PyResult<Py<RefinerKindPy>> {
        Ok(self.refiner.clone_ref(py))
    }

    #[setter]
    fn set_refiner(&mut self, value: Py<RefinerKindPy>) {
        self.refiner = value;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("use_radius10", self.use_radius10())?;
        dict.set_item("descriptor_use_radius10", self.descriptor_use_radius10())?;
        dict.set_item("threshold_rel", self.threshold_rel())?;
        dict.set_item("threshold_abs", self.threshold_abs())?;
        dict.set_item("nms_radius", self.nms_radius())?;
        dict.set_item("min_cluster_size", self.min_cluster_size())?;
        let refiner = self.refiner.bind(py).borrow();
        dict.set_item("refiner", refiner.to_dict(py)?)?;
        Ok(dict.into_any().unbind())
    }
}

/// Python-facing configuration wrapper for `ChessConfig`.
#[pyclass(name = "ChessConfig")]
pub struct ChessConfigPy {
    params: Py<ChessParamsPy>,
    multiscale: Py<CoarseToFineParamsPy>,
}

impl ChessConfigPy {
    fn from_rust(py: Python<'_>, cfg: &chess_corners_rs::ChessConfig) -> PyResult<Self> {
        let params = Py::new(py, ChessParamsPy::from_rust(py, &cfg.params)?)?;
        let multiscale = Py::new(py, CoarseToFineParamsPy::from_rust(py, &cfg.multiscale)?)?;
        Ok(Self { params, multiscale })
    }

    fn to_rust(&self, py: Python<'_>) -> PyResult<chess_corners_rs::ChessConfig> {
        let params = self.params.bind(py).borrow().to_rust(py)?;
        let multiscale = self.multiscale.bind(py).borrow().to_rust(py)?;
        Ok(chess_corners_rs::ChessConfig { params, multiscale })
    }
}

#[pymethods]
impl ChessConfigPy {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Self::from_rust(py, &chess_corners_rs::ChessConfig::default())
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Py<ChessParamsPy>> {
        Ok(self.params.clone_ref(py))
    }

    #[setter]
    fn set_params(&mut self, value: Py<ChessParamsPy>) {
        self.params = value;
    }

    #[getter]
    fn multiscale(&self, py: Python<'_>) -> PyResult<Py<CoarseToFineParamsPy>> {
        Ok(self.multiscale.clone_ref(py))
    }

    #[setter]
    fn set_multiscale(&mut self, value: Py<CoarseToFineParamsPy>) {
        self.multiscale = value;
    }

    #[getter]
    fn use_radius10(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.params.bind(py).borrow().use_radius10)
    }

    #[setter]
    fn set_use_radius10(&mut self, py: Python<'_>, value: bool) -> PyResult<()> {
        self.params.bind(py).borrow_mut().use_radius10 = value;
        Ok(())
    }

    #[getter]
    fn descriptor_use_radius10(&self, py: Python<'_>) -> PyResult<Option<bool>> {
        Ok(self.params.bind(py).borrow().descriptor_use_radius10)
    }

    #[setter]
    fn set_descriptor_use_radius10(&mut self, py: Python<'_>, value: Option<bool>) -> PyResult<()> {
        self.params.bind(py).borrow_mut().descriptor_use_radius10 = value;
        Ok(())
    }

    #[getter]
    fn threshold_rel(&self, py: Python<'_>) -> PyResult<f32> {
        Ok(self.params.bind(py).borrow().threshold_rel)
    }

    #[setter]
    fn set_threshold_rel(&mut self, py: Python<'_>, value: f32) -> PyResult<()> {
        self.params.bind(py).borrow_mut().threshold_rel = value;
        Ok(())
    }

    #[getter]
    fn threshold_abs(&self, py: Python<'_>) -> PyResult<Option<f32>> {
        Ok(self.params.bind(py).borrow().threshold_abs)
    }

    #[setter]
    fn set_threshold_abs(&mut self, py: Python<'_>, value: Option<f32>) -> PyResult<()> {
        self.params.bind(py).borrow_mut().threshold_abs = value;
        Ok(())
    }

    #[getter]
    fn nms_radius(&self, py: Python<'_>) -> PyResult<u32> {
        Ok(self.params.bind(py).borrow().nms_radius)
    }

    #[setter]
    fn set_nms_radius(&mut self, py: Python<'_>, value: u32) -> PyResult<()> {
        self.params.bind(py).borrow_mut().nms_radius = value;
        Ok(())
    }

    #[getter]
    fn min_cluster_size(&self, py: Python<'_>) -> PyResult<u32> {
        Ok(self.params.bind(py).borrow().min_cluster_size)
    }

    #[setter]
    fn set_min_cluster_size(&mut self, py: Python<'_>, value: u32) -> PyResult<()> {
        self.params.bind(py).borrow_mut().min_cluster_size = value;
        Ok(())
    }

    #[getter]
    fn pyramid_num_levels(&self, py: Python<'_>) -> PyResult<u8> {
        let pyramid = self.multiscale.bind(py).borrow().pyramid.clone_ref(py);
        Ok(pyramid.bind(py).borrow().num_levels)
    }

    #[setter]
    fn set_pyramid_num_levels(&mut self, py: Python<'_>, value: u8) -> PyResult<()> {
        let pyramid = self.multiscale.bind(py).borrow().pyramid.clone_ref(py);
        pyramid.bind(py).borrow_mut().num_levels = value;
        Ok(())
    }

    #[getter]
    fn pyramid_min_size(&self, py: Python<'_>) -> PyResult<usize> {
        let pyramid = self.multiscale.bind(py).borrow().pyramid.clone_ref(py);
        Ok(pyramid.bind(py).borrow().min_size)
    }

    #[setter]
    fn set_pyramid_min_size(&mut self, py: Python<'_>, value: usize) -> PyResult<()> {
        let pyramid = self.multiscale.bind(py).borrow().pyramid.clone_ref(py);
        pyramid.bind(py).borrow_mut().min_size = value;
        Ok(())
    }

    #[getter]
    fn refinement_radius(&self, py: Python<'_>) -> PyResult<u32> {
        Ok(self.multiscale.bind(py).borrow().refinement_radius)
    }

    #[setter]
    fn set_refinement_radius(&mut self, py: Python<'_>, value: u32) -> PyResult<()> {
        self.multiscale.bind(py).borrow_mut().refinement_radius = value;
        Ok(())
    }

    #[getter]
    fn merge_radius(&self, py: Python<'_>) -> PyResult<f32> {
        Ok(self.multiscale.bind(py).borrow().merge_radius)
    }

    #[setter]
    fn set_merge_radius(&mut self, py: Python<'_>, value: f32) -> PyResult<()> {
        self.multiscale.bind(py).borrow_mut().merge_radius = value;
        Ok(())
    }

    /// Return a dictionary snapshot of the current configuration values.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let params = self.params.bind(py).borrow();
        let multiscale = self.multiscale.bind(py).borrow();
        let pyramid = multiscale.pyramid.bind(py).borrow();

        dict.set_item("use_radius10", params.use_radius10())?;
        dict.set_item("descriptor_use_radius10", params.descriptor_use_radius10())?;
        dict.set_item("threshold_rel", params.threshold_rel())?;
        dict.set_item("threshold_abs", params.threshold_abs())?;
        dict.set_item("nms_radius", params.nms_radius())?;
        dict.set_item("min_cluster_size", params.min_cluster_size())?;
        dict.set_item("pyramid_num_levels", pyramid.num_levels())?;
        dict.set_item("pyramid_min_size", pyramid.min_size())?;
        dict.set_item("refinement_radius", multiscale.refinement_radius())?;
        dict.set_item("merge_radius", multiscale.merge_radius())?;
        dict.set_item("params", params.to_dict(py)?)?;
        dict.set_item("multiscale", multiscale.to_dict(py)?)?;
        Ok(dict.into_any().unbind())
    }
}

fn extract_image<'py>(
    image: &Bound<'py, PyAny>,
) -> PyResult<(PyReadonlyArray2<'py, u8>, usize, usize)> {
    let array = image
        .extract::<PyReadonlyArray2<u8>>()
        .map_err(|_| PyTypeError::new_err("image must be a uint8 numpy array of shape (H, W)"))?;
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "image must be a C-contiguous uint8 array of shape (H, W)",
        ));
    }
    let (h, w) = view.dim();
    Ok((array, h, w))
}

fn corners_to_array(
    py: Python<'_>,
    mut corners: Vec<chess_corners_rs::CornerDescriptor>,
) -> PyResult<Py<PyAny>> {
    corners.sort_by(|a, b| {
        b.response
            .total_cmp(&a.response)
            .then_with(|| a.x.total_cmp(&b.x))
            .then_with(|| a.y.total_cmp(&b.y))
    });

    let mut data = Vec::with_capacity(corners.len() * 4);
    for corner in corners {
        data.push(corner.x);
        data.push(corner.y);
        data.push(corner.response);
        data.push(corner.orientation);
    }

    let out = Array2::from_shape_vec((data.len() / 4, 4), data)
        .map_err(|_| PyValueError::new_err("failed to build output array"))?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

fn resolve_cfg(
    py: Python<'_>,
    cfg: Option<&ChessConfigPy>,
) -> PyResult<chess_corners_rs::ChessConfig> {
    match cfg {
        Some(cfg) => cfg.to_rust(py),
        None => Ok(chess_corners_rs::ChessConfig::default()),
    }
}

/// Detect chessboard corners from a 2D uint8 NumPy array.
#[pyfunction]
fn find_chess_corners<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg: Option<&ChessConfigPy>,
) -> PyResult<Py<PyAny>> {
    let (array, h, w) = extract_image(image)?;
    let view = array.as_array();
    let slice = view.as_slice().ok_or_else(|| {
        PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
    })?;

    let width_u32 =
        u32::try_from(w).map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
    let height_u32 =
        u32::try_from(h).map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

    let cfg_rust = resolve_cfg(py, cfg)?;
    let corners = chess_corners_rs::find_chess_corners_u8(slice, width_u32, height_u32, &cfg_rust);
    corners_to_array(py, corners)
}

/// Detect chessboard corners from a 2D uint8 NumPy array using the ML refiner pipeline.
#[cfg(feature = "ml-refiner")]
#[pyfunction]
fn find_chess_corners_with_ml<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg: Option<&ChessConfigPy>,
) -> PyResult<Py<PyAny>> {
    let (array, h, w) = extract_image(image)?;
    let view = array.as_array();
    let slice = view.as_slice().ok_or_else(|| {
        PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
    })?;

    let width_u32 =
        u32::try_from(w).map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
    let height_u32 =
        u32::try_from(h).map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

    let cfg_rust = resolve_cfg(py, cfg)?;
    let corners =
        chess_corners_rs::find_chess_corners_u8_with_ml(slice, width_u32, height_u32, &cfg_rust);
    corners_to_array(py, corners)
}

#[pymodule]
fn chess_corners(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ChessConfigPy>()?;
    m.add_class::<ChessParamsPy>()?;
    m.add_class::<CoarseToFineParamsPy>()?;
    m.add_class::<PyramidParamsPy>()?;
    m.add_class::<CenterOfMassConfigPy>()?;
    m.add_class::<ForstnerConfigPy>()?;
    m.add_class::<SaddlePointConfigPy>()?;
    m.add_class::<RefinerKindPy>()?;
    #[cfg(feature = "ml-refiner")]
    {
        m.add_function(wrap_pyfunction!(find_chess_corners_with_ml, m)?)?;
    }
    m.add_function(wrap_pyfunction!(find_chess_corners, m)?)?;

    let mut exports = vec![
        "ChessConfig",
        "ChessParams",
        "CoarseToFineParams",
        "PyramidParams",
        "CenterOfMassConfig",
        "ForstnerConfig",
        "SaddlePointConfig",
        "RefinerKind",
        "find_chess_corners",
    ];
    #[cfg(feature = "ml-refiner")]
    {
        exports.push("find_chess_corners_with_ml");
    }
    m.add("__all__", exports)?;
    Ok(())
}
