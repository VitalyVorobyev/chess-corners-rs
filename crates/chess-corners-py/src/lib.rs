use ::chess_corners as chess_corners_rs;
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

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

fn resolve_cfg_json(cfg_json: Option<&str>) -> PyResult<chess_corners_rs::ChessConfig> {
    match cfg_json {
        Some(raw) => serde_json::from_str(raw)
            .map_err(|err| PyValueError::new_err(format!("invalid config JSON: {err}"))),
        None => Ok(chess_corners_rs::ChessConfig::default()),
    }
}

#[pyfunction(signature = (image, cfg_json=None))]
fn find_chess_corners<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg_json: Option<&str>,
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

    let cfg = resolve_cfg_json(cfg_json)?;
    let corners = chess_corners_rs::find_chess_corners_u8(slice, width_u32, height_u32, &cfg);
    corners_to_array(py, corners)
}

#[cfg(feature = "ml-refiner")]
#[pyfunction(signature = (image, cfg_json=None))]
fn find_chess_corners_with_ml<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    cfg_json: Option<&str>,
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

    let cfg = resolve_cfg_json(cfg_json)?;
    let corners =
        chess_corners_rs::find_chess_corners_u8_with_ml(slice, width_u32, height_u32, &cfg);
    corners_to_array(py, corners)
}

#[pymodule(name = "_native")]
fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_chess_corners, m)?)?;
    #[cfg(feature = "ml-refiner")]
    {
        m.add_function(wrap_pyfunction!(find_chess_corners_with_ml, m)?)?;
    }
    Ok(())
}
