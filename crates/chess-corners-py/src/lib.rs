//! Native Python bindings for the chess-corners detector.
//!
//! Exposes a typed [`config::ChessConfig`] (with nested
//! [`config::RefinerConfig`], [`config::RadonDetectorParams`], and
//! per-variant refiner configs) plus thin wrappers over the facade's
//! detection entry points. The FFI accepts the typed config directly
//! — no JSON serialization across the boundary — while preserving a
//! string-only fallback so callers built against the older Python
//! `to_json()`-then-pass-string path keep working for one release.

mod config;

use ::chess_corners as chess_corners_rs;
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::config::{
    CenterOfMassConfig, ChessConfig, ConfigError, DescriptorMode, DetectorMode, ForstnerConfig,
    PeakFitMode, RadonDetectorParams, RadonPeakConfig, RefinementMethod, RefinerConfig,
    SaddlePointConfig, ThresholdMode,
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

/// Resolve the optional `cfg` argument into a Rust facade
/// `ChessConfig`. Accepts either a typed [`ChessConfig`] (preferred,
/// no JSON across the boundary), or a JSON string (legacy path
/// retained for one release). Anything else raises `TypeError`.
fn resolve_config(
    py: Python<'_>,
    cfg: Option<&Bound<'_, PyAny>>,
) -> PyResult<chess_corners_rs::ChessConfig> {
    let Some(cfg) = cfg else {
        return Ok(chess_corners_rs::ChessConfig::default());
    };
    if cfg.is_none() {
        return Ok(chess_corners_rs::ChessConfig::default());
    }
    if let Ok(typed) = cfg.cast::<ChessConfig>() {
        return Ok(typed.borrow().to_inner(py));
    }
    if let Ok(json) = cfg.extract::<&str>() {
        return serde_json::from_str(json)
            .map_err(|err| PyValueError::new_err(format!("invalid config JSON: {err}")));
    }
    Err(PyTypeError::new_err(
        "cfg must be a ChessConfig or a JSON string",
    ))
}

#[pyfunction(signature = (image, cfg=None))]
fn find_chess_corners<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (array, height, width) = extract_image(image)?;
    let view = array.as_array();
    let slice = view.as_slice().ok_or_else(|| {
        PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
    })?;

    let width_u32 =
        u32::try_from(width).map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

    let cfg = resolve_config(py, cfg)?;
    let corners = chess_corners_rs::find_chess_corners_u8(slice, width_u32, height_u32, &cfg);
    corners_to_array(py, corners)
}

#[cfg(feature = "ml-refiner")]
#[pyfunction(signature = (image, cfg=None))]
fn find_chess_corners_with_ml<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (array, height, width) = extract_image(image)?;
    let view = array.as_array();
    let slice = view.as_slice().ok_or_else(|| {
        PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
    })?;

    let width_u32 =
        u32::try_from(width).map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

    let cfg = resolve_config(py, cfg)?;
    let corners =
        chess_corners_rs::find_chess_corners_u8_with_ml(slice, width_u32, height_u32, &cfg);
    corners_to_array(py, corners)
}

#[pyfunction(signature = (image, cfg=None))]
fn radon_heatmap<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (array, height, width) = extract_image(image)?;
    let view = array.as_array();
    let slice = view.as_slice().ok_or_else(|| {
        PyValueError::new_err("image must be a C-contiguous uint8 array of shape (H, W)")
    })?;

    let width_u32 =
        u32::try_from(width).map_err(|_| PyValueError::new_err("image width exceeds u32::MAX"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| PyValueError::new_err("image height exceeds u32::MAX"))?;

    let cfg = resolve_config(py, cfg)?;
    let map = chess_corners_rs::radon_heatmap_u8(slice, width_u32, height_u32, &cfg);

    let arr = Array2::from_shape_vec((map.height(), map.width()), map.data().to_vec())
        .map_err(|_| PyValueError::new_err("failed to build heatmap array"))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

#[pymodule(name = "_native")]
fn native_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConfigError", py.get_type::<ConfigError>())?;

    m.add_class::<DetectorMode>()?;
    m.add_class::<DescriptorMode>()?;
    m.add_class::<ThresholdMode>()?;
    m.add_class::<RefinementMethod>()?;
    m.add_class::<PeakFitMode>()?;

    m.add_class::<CenterOfMassConfig>()?;
    m.add_class::<ForstnerConfig>()?;
    m.add_class::<SaddlePointConfig>()?;
    m.add_class::<RadonPeakConfig>()?;
    m.add_class::<RadonDetectorParams>()?;
    m.add_class::<RefinerConfig>()?;
    m.add_class::<ChessConfig>()?;

    m.add_function(wrap_pyfunction!(find_chess_corners, m)?)?;
    m.add_function(wrap_pyfunction!(radon_heatmap, m)?)?;
    #[cfg(feature = "ml-refiner")]
    {
        m.add_function(wrap_pyfunction!(find_chess_corners_with_ml, m)?)?;
    }
    Ok(())
}
