//! Native Python bindings for the chess-corners detector.
//!
//! Exposes a typed [`config::DetectorConfig`] mirroring the Rust
//! facade tree (strategy-typed `ChessConfig` / `RadonConfig`,
//! per-detector refiners, tagged `MultiscaleConfig` / `UpscaleConfig`
//! / `Threshold` classes) plus a [`Detector`] PyClass that wraps
//! the facade's reusable buffers.

mod config;

use ::chess_corners as chess_corners_rs;
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::config::{
    CenterOfMassConfig, ChessConfig, ChessRefiner, ChessRing, ConfigError, DescriptorRing,
    DetectionStrategy, DetectorConfig, ForstnerConfig, MultiscaleConfig, OrientationMethod,
    PeakFitMode, RadonConfig, RadonPeakConfig, RadonRefiner, SaddlePointConfig, Threshold,
    UpscaleConfig,
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

/// Number of float32 columns in the returned corner array.
///
/// Columns: `x, y, response, contrast, fit_rms,
///           axis0_angle, axis0_sigma, axis1_angle, axis1_sigma`.
const CORNER_COLUMNS: usize = 9;

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

    let mut data = Vec::with_capacity(corners.len() * CORNER_COLUMNS);
    for corner in corners {
        data.push(corner.x);
        data.push(corner.y);
        data.push(corner.response);
        data.push(corner.contrast);
        data.push(corner.fit_rms);
        data.push(corner.axes[0].angle);
        data.push(corner.axes[0].sigma);
        data.push(corner.axes[1].angle);
        data.push(corner.axes[1].sigma);
    }

    let rows = data.len() / CORNER_COLUMNS;
    let out = Array2::from_shape_vec((rows, CORNER_COLUMNS), data)
        .map_err(|_| PyValueError::new_err("failed to build output array"))?;
    Ok(out.into_pyarray(py).into_any().unbind())
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

    /// Detect chessboard corners. `image` must be a C-contiguous
    /// `uint8` array of shape `(H, W)`. Returns an `(N, 9)` `float32`
    /// NumPy array — see module docs for column layout.
    fn detect<'py>(&mut self, py: Python<'py>, image: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
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
        corners_to_array(py, corners)
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

    /// Compute the dense Radon response heatmap.
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
            .detach(|| self.inner.radon_heatmap_u8(slice, width_u32, height_u32))
            .map_err(|e: chess_corners_rs::ChessError| PyValueError::new_err(e.to_string()))?;

        let arr = Array2::from_shape_vec((map.height(), map.width()), map.data().to_vec())
            .map_err(|_| PyValueError::new_err("failed to build heatmap array"))?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }
}

#[pymodule(name = "_native")]
fn native_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConfigError", py.get_type::<ConfigError>())?;

    m.add_class::<ChessRing>()?;
    m.add_class::<DescriptorRing>()?;
    m.add_class::<PeakFitMode>()?;
    m.add_class::<OrientationMethod>()?;

    m.add_class::<CenterOfMassConfig>()?;
    m.add_class::<ForstnerConfig>()?;
    m.add_class::<SaddlePointConfig>()?;
    m.add_class::<RadonPeakConfig>()?;
    m.add_class::<Threshold>()?;
    m.add_class::<MultiscaleConfig>()?;
    m.add_class::<UpscaleConfig>()?;
    m.add_class::<ChessRefiner>()?;
    m.add_class::<RadonRefiner>()?;
    m.add_class::<ChessConfig>()?;
    m.add_class::<RadonConfig>()?;
    m.add_class::<DetectionStrategy>()?;
    m.add_class::<DetectorConfig>()?;
    m.add_class::<Detector>()?;

    Ok(())
}
