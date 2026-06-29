//! Native Python bindings for the chess-corners detector.
//!
//! Exposes a typed [`config::DetectorConfig`] mirroring the Rust
//! facade tree (strategy-typed `ChessConfig` / `RadonConfig`,
//! per-detector refiners, tagged `MultiscaleConfig` / `UpscaleConfig`
//! classes) plus a [`Detector`] PyClass that wraps the facade's
//! reusable buffers.

mod config;

use ::chess_corners as chess_corners_rs;
use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyReadonlyArray2,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::config::{
    CenterOfMassConfig, ChessConfig, ChessRefiner, ChessRing, ConfigError, DetectionParams,
    DetectionStrategy, DetectorConfig, ForstnerConfig, MultiscaleConfig, OrientationMethod,
    PeakFitMode, RadonConfig, SaddlePointConfig, UpscaleConfig,
};

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
    let (height, width) = view.dim();
    Ok((array, height, width))
}

/// Detected chessboard corners with per-corner response and optional axis orientation.
///
/// Built by [`Detector::detect`]. Access coordinates via [`Self::xy`],
/// per-corner response via [`Self::response`], and axis orientations via
/// [`Self::angles`] and [`Self::sigmas`]. Both `angles` and `sigmas` are
/// `None` when orientation estimation was disabled for the run.
#[pyclass(module = "chess_corners")]
pub struct Detections {
    n: usize,
    orientation_on: bool,
    xy: Py<PyAny>,
    response: Py<PyAny>,
    angles: Option<Py<PyAny>>,
    sigmas: Option<Py<PyAny>>,
}

#[pymethods]
impl Detections {
    /// `(N, 2)` float32 array of corner coordinates `(x, y)` in image pixels.
    #[getter]
    fn xy(&self, py: Python<'_>) -> Py<PyAny> {
        self.xy.clone_ref(py)
    }

    /// `(N,)` float32 array of per-corner response values.
    #[getter]
    fn response(&self, py: Python<'_>) -> Py<PyAny> {
        self.response.clone_ref(py)
    }

    /// `(N, 2)` float32 array of axis angles (radians) when orientation
    /// estimation is enabled, `None` otherwise.
    #[getter]
    fn angles(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.angles.as_ref().map(|a| a.clone_ref(py))
    }

    /// `(N, 2)` float32 array of per-axis angle uncertainty (1σ, radians)
    /// when orientation estimation is enabled, `None` otherwise.
    #[getter]
    fn sigmas(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.sigmas.as_ref().map(|s| s.clone_ref(py))
    }

    fn __len__(&self) -> usize {
        self.n
    }

    fn __repr__(&self) -> String {
        let orient = if self.orientation_on { "on" } else { "off" };
        format!("Detections(n={}, orientation={})", self.n, orient)
    }
}

fn build_detections(
    py: Python<'_>,
    mut corners: Vec<chess_corners_rs::CornerDescriptor>,
) -> PyResult<Detections> {
    corners.sort_by(|a, b| {
        b.response
            .total_cmp(&a.response)
            .then_with(|| a.x.total_cmp(&b.x))
            .then_with(|| a.y.total_cmp(&b.y))
    });

    let n = corners.len();
    // Orientation is a global config: either all corners have axes or none do.
    // Empty result → treat as orientation-on with (0, 2) arrays.
    let orientation_on = n == 0 || corners[0].axes.is_some();

    let mut xy_data = Vec::with_capacity(n * 2);
    let mut resp_data = Vec::with_capacity(n);
    let mut angles_data = if orientation_on {
        Vec::with_capacity(n * 2)
    } else {
        Vec::new()
    };
    let mut sigmas_data = if orientation_on {
        Vec::with_capacity(n * 2)
    } else {
        Vec::new()
    };

    for corner in corners {
        xy_data.push(corner.x);
        xy_data.push(corner.y);
        resp_data.push(corner.response);
        if orientation_on {
            match corner.axes {
                Some(axes) => {
                    angles_data.push(axes[0].angle);
                    angles_data.push(axes[1].angle);
                    sigmas_data.push(axes[0].sigma);
                    sigmas_data.push(axes[1].sigma);
                }
                None => {
                    // Guard: orientation is global so this branch is unreachable
                    // in well-formed output, but NaN is safer than panicking.
                    angles_data.extend_from_slice(&[f32::NAN; 2]);
                    sigmas_data.extend_from_slice(&[f32::NAN; 2]);
                }
            }
        }
    }

    let xy = Array2::from_shape_vec((n, 2), xy_data)
        .map_err(|e| PyValueError::new_err(format!("xy array: {e}")))?
        .into_pyarray(py)
        .into_any()
        .unbind();

    let response = Array1::from_vec(resp_data)
        .into_pyarray(py)
        .into_any()
        .unbind();

    let (angles, sigmas) = if orientation_on {
        let a = Array2::from_shape_vec((n, 2), angles_data)
            .map_err(|e| PyValueError::new_err(format!("angles array: {e}")))?
            .into_pyarray(py)
            .into_any()
            .unbind();
        let s = Array2::from_shape_vec((n, 2), sigmas_data)
            .map_err(|e| PyValueError::new_err(format!("sigmas array: {e}")))?
            .into_pyarray(py)
            .into_any()
            .unbind();
        (Some(a), Some(s))
    } else {
        (None, None)
    };

    Ok(Detections {
        n,
        orientation_on,
        xy,
        response,
        angles,
        sigmas,
    })
}

/// Resolve the optional `cfg` argument into a Rust facade `DetectorConfig`.
/// Accepts a typed [`DetectorConfig`] or `None` (uses defaults). Any other
/// type raises `TypeError`.
fn resolve_config(
    py: Python<'_>,
    cfg: Option<&Bound<'_, PyAny>>,
) -> PyResult<chess_corners_rs::DetectorConfig> {
    let Some(cfg) = cfg else {
        return Ok(chess_corners_rs::DetectorConfig::default());
    };
    if cfg.is_none() {
        return Ok(chess_corners_rs::DetectorConfig::default());
    }
    if let Ok(typed) = cfg.cast::<DetectorConfig>() {
        return Ok(typed.borrow().to_inner(py));
    }
    Err(PyTypeError::new_err("cfg must be a DetectorConfig"))
}

/// Stateful chessboard-corner detector with reusable scratch buffers.
///
/// Mirrors the Rust [`chess_corners::Detector`]. Build once, then call
/// [`Self::detect`] in a loop to amortise pyramid / upscale buffer
/// allocations across frames.
#[pyclass]
pub struct Detector {
    inner: chess_corners_rs::Detector,
}

#[pymethods]
impl Detector {
    /// Build a detector. `cfg` may be `None` (use defaults) or a
    /// typed [`DetectorConfig`].
    #[new]
    #[pyo3(signature = (cfg=None))]
    fn new(py: Python<'_>, cfg: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let cfg = resolve_config(py, cfg)?;
        let inner = chess_corners_rs::Detector::new(cfg)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Detect chessboard corners in `image`. Returns a [`Detections`] object
    /// with named array fields. `image` must be a C-contiguous `uint8` array
    /// of shape `(H, W)`.
    fn detect<'py>(&mut self, py: Python<'py>, image: &Bound<'py, PyAny>) -> PyResult<Detections> {
        let (array, height, width) = extract_image(image)?;
        let view = array.as_array();
        let slice = view.as_slice().ok_or_else(|| {
            PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
        })?;

        let width_u32 = u32::try_from(width)
            .map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
        let height_u32 = u32::try_from(height)
            .map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

        let corners = py
            .detach(|| self.inner.detect_u8(slice, width_u32, height_u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        build_detections(py, corners)
    }

    /// Return a snapshot of the current detector configuration.
    ///
    /// The returned [`DetectorConfig`] is an independent copy — mutating it
    /// does not affect the live detector. Pass it back to
    /// [`Self::apply_config`] after editing.
    fn config(&self, py: Python<'_>) -> PyResult<DetectorConfig> {
        DetectorConfig::from_rs(py, *self.inner.config())
    }

    /// Replace the detector configuration.
    ///
    /// Equivalent to `Detector(cfg)` but reuses the existing pyramid and
    /// upscale scratch buffers (they are resized lazily on the next
    /// [`Self::detect`] call). Validates the upscale factor; raises
    /// `ValueError` if the config is invalid.
    fn apply_config(&mut self, py: Python<'_>, cfg: &Bound<'_, PyAny>) -> PyResult<()> {
        let cfg = resolve_config(py, Some(cfg))?;
        self.inner
            .set_config(cfg)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return the intermediate dense Radon-response heatmap for an image.
    ///
    /// This exposes the per-pixel Radon response that the detector
    /// computes internally as opt-in diagnostic evidence for debugging
    /// and visualization. It is not part of the normal detection result
    /// returned by `detect`.
    fn radon_heatmap<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (array, height, width) = extract_image(image)?;
        let view = array.as_array();
        let slice = view.as_slice().ok_or_else(|| {
            PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
        })?;

        let width_u32 = u32::try_from(width)
            .map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
        let height_u32 = u32::try_from(height)
            .map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

        let map = py
            .detach(|| {
                self.inner
                    .diagnostics()
                    .radon_heatmap_u8(slice, width_u32, height_u32)
            })
            .map_err(|e: chess_corners_rs::ChessError| PyValueError::new_err(e.to_string()))?;

        let arr = Array2::from_shape_vec((map.height(), map.width()), map.data().to_vec())
            .map_err(|_| PyValueError::new_err("failed to build heatmap array"))?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }
}

#[pymodule(name = "_native")]
fn native_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConfigError", py.get_type::<ConfigError>())?;

    m.add_class::<Detections>()?;
    m.add_class::<ChessRing>()?;
    m.add_class::<PeakFitMode>()?;
    m.add_class::<OrientationMethod>()?;

    m.add_class::<CenterOfMassConfig>()?;
    m.add_class::<ForstnerConfig>()?;
    m.add_class::<SaddlePointConfig>()?;
    m.add_class::<DetectionParams>()?;
    m.add_class::<MultiscaleConfig>()?;
    m.add_class::<UpscaleConfig>()?;
    m.add_class::<ChessRefiner>()?;
    m.add_class::<ChessConfig>()?;
    m.add_class::<RadonConfig>()?;
    m.add_class::<DetectionStrategy>()?;
    m.add_class::<DetectorConfig>()?;
    m.add_class::<Detector>()?;

    Ok(())
}
