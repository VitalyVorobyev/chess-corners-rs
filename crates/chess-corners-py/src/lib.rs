use ::chess_corners as chess_corners_rs;
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyModule, PyModuleMethods};

/// Python-facing configuration wrapper for `ChessConfig`.
#[pyclass(name = "ChessConfig")]
#[derive(Clone, Debug)]
pub struct ChessConfigPy {
    inner: chess_corners_rs::ChessConfig,
}

#[pymethods]
impl ChessConfigPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: chess_corners_rs::ChessConfig::default(),
        }
    }

    #[getter]
    fn use_radius10(&self) -> bool {
        self.inner.params.use_radius10
    }

    #[setter]
    fn set_use_radius10(&mut self, value: bool) {
        self.inner.params.use_radius10 = value;
    }

    #[getter]
    fn descriptor_use_radius10(&self) -> Option<bool> {
        self.inner.params.descriptor_use_radius10
    }

    #[setter]
    fn set_descriptor_use_radius10(&mut self, value: Option<bool>) {
        self.inner.params.descriptor_use_radius10 = value;
    }

    #[getter]
    fn threshold_rel(&self) -> f32 {
        self.inner.params.threshold_rel
    }

    #[setter]
    fn set_threshold_rel(&mut self, value: f32) {
        self.inner.params.threshold_rel = value;
    }

    #[getter]
    fn threshold_abs(&self) -> Option<f32> {
        self.inner.params.threshold_abs
    }

    #[setter]
    fn set_threshold_abs(&mut self, value: Option<f32>) {
        self.inner.params.threshold_abs = value;
    }

    #[getter]
    fn nms_radius(&self) -> u32 {
        self.inner.params.nms_radius
    }

    #[setter]
    fn set_nms_radius(&mut self, value: u32) {
        self.inner.params.nms_radius = value;
    }

    #[getter]
    fn min_cluster_size(&self) -> u32 {
        self.inner.params.min_cluster_size
    }

    #[setter]
    fn set_min_cluster_size(&mut self, value: u32) {
        self.inner.params.min_cluster_size = value;
    }

    #[getter]
    fn pyramid_num_levels(&self) -> u8 {
        self.inner.multiscale.pyramid.num_levels
    }

    #[setter]
    fn set_pyramid_num_levels(&mut self, value: u8) {
        self.inner.multiscale.pyramid.num_levels = value;
    }

    #[getter]
    fn pyramid_min_size(&self) -> usize {
        self.inner.multiscale.pyramid.min_size
    }

    #[setter]
    fn set_pyramid_min_size(&mut self, value: usize) {
        self.inner.multiscale.pyramid.min_size = value;
    }

    #[getter]
    fn refinement_radius(&self) -> u32 {
        self.inner.multiscale.refinement_radius
    }

    #[setter]
    fn set_refinement_radius(&mut self, value: u32) {
        self.inner.multiscale.refinement_radius = value;
    }

    #[getter]
    fn merge_radius(&self) -> f32 {
        self.inner.multiscale.merge_radius
    }

    #[setter]
    fn set_merge_radius(&mut self, value: f32) {
        self.inner.multiscale.merge_radius = value;
    }

    /// Return a dictionary snapshot of the current configuration values.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("use_radius10", self.use_radius10())?;
        dict.set_item("descriptor_use_radius10", self.descriptor_use_radius10())?;
        dict.set_item("threshold_rel", self.threshold_rel())?;
        dict.set_item("threshold_abs", self.threshold_abs())?;
        dict.set_item("nms_radius", self.nms_radius())?;
        dict.set_item("min_cluster_size", self.min_cluster_size())?;
        dict.set_item("pyramid_num_levels", self.pyramid_num_levels())?;
        dict.set_item("pyramid_min_size", self.pyramid_min_size())?;
        dict.set_item("refinement_radius", self.refinement_radius())?;
        dict.set_item("merge_radius", self.merge_radius())?;
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

    let cfg_default = chess_corners_rs::ChessConfig::default();
    let cfg_ref = cfg.map(|cfg| &cfg.inner).unwrap_or(&cfg_default);

    let mut corners =
        chess_corners_rs::find_chess_corners_u8(slice, width_u32, height_u32, cfg_ref);
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

#[pymodule]
fn chess_corners(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ChessConfigPy>()?;
    m.add_function(wrap_pyfunction!(find_chess_corners, m)?)?;
    m.add("__all__", vec!["ChessConfig", "find_chess_corners"])?;
    Ok(())
}
