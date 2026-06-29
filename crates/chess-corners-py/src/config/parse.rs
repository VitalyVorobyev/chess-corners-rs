//! Dict-deserialization helpers, shared enums, and JSON utilities.

use std::collections::BTreeSet;

use chess_corners::{
    ChessRing as RsChessRing, OrientationMethod as RsOrientationMethod,
    PeakFitMode as RsPeakFitMode,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};

use super::ConfigError;

// ---------------------------------------------------------------------------
// Helpers for dict → field deserialization with unknown-key rejection.
// ---------------------------------------------------------------------------

pub(crate) fn config_error(msg: impl Into<String>) -> PyErr {
    ConfigError::new_err(msg.into())
}

pub(crate) fn require_dict<'py>(
    value: &Bound<'py, PyAny>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    value
        .cast::<PyDict>()
        .cloned()
        .map_err(|_| config_error(format!("{path} must be an object")))
}

/// For an externally-tagged enum that has both unit variants (serialised by
/// serde as a bare string, e.g. `"single_scale"`) and payload variants
/// (serialised as `{"pyramid": {...}}`), accept either encoding on the
/// input side.
///
/// Returns:
/// - `Ok(None)` — `value` is a dict; the caller parses it normally.
/// - `Ok(Some(tag))` — `value` is the string `tag`, matched a unit-variant name.
/// - `Err(ConfigError)` — bare string that doesn't match any unit tag, or
///   a payload-only variant name used as a bare string.
pub(crate) fn accept_dict_or_bare_string<'py>(
    value: &Bound<'py, PyAny>,
    unit_tags: &[&str],
    payload_tags: &[&str],
    path: &str,
) -> PyResult<Option<String>> {
    if value.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    if value.is_instance_of::<PyString>() {
        let s: String = value.extract()?;
        if unit_tags.contains(&s.as_str()) {
            return Ok(Some(s));
        }
        if payload_tags.contains(&s.as_str()) {
            return Err(config_error(format!(
                "{path}: \"{s}\" requires a payload — use {{\"{}\":{{...}}}} instead",
                s
            )));
        }
        let all: Vec<&str> = unit_tags
            .iter()
            .chain(payload_tags.iter())
            .copied()
            .collect();
        return Err(config_error(format!(
            "{path}: got bare string \"{s}\", expected one of: {}",
            all.join(", ")
        )));
    }
    Err(config_error(format!("{path} must be an object or string")))
}

pub(crate) fn reject_unknown_keys(
    dict: &Bound<'_, PyDict>,
    allowed: &[&str],
    path: &str,
) -> PyResult<()> {
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

pub(crate) fn extract_int(
    dict: &Bound<'_, PyDict>,
    key: &str,
    path: &str,
) -> PyResult<Option<i64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(config_error(format!("{path}.{key} must be an integer")));
    }
    Ok(Some(value.extract::<i64>()?))
}

pub(crate) fn extract_float(
    dict: &Bound<'_, PyDict>,
    key: &str,
    path: &str,
) -> PyResult<Option<f64>> {
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

pub(crate) fn extract_string(
    dict: &Bound<'_, PyDict>,
    key: &str,
    path: &str,
) -> PyResult<Option<String>> {
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

pub(crate) fn parse_enum<E>(value: &str, path: &str, allowed: &[(&str, E)]) -> PyResult<E>
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

pub(crate) fn parse_chess_ring(value: &str, path: &str) -> PyResult<RsChessRing> {
    parse_enum(
        value,
        path,
        &[
            ("canonical", RsChessRing::Canonical),
            ("broad", RsChessRing::Broad),
        ],
    )
}

pub(crate) fn parse_peak_fit_mode(value: &str, path: &str) -> PyResult<RsPeakFitMode> {
    parse_enum(
        value,
        path,
        &[
            ("parabolic", RsPeakFitMode::Parabolic),
            ("gaussian", RsPeakFitMode::Gaussian),
        ],
    )
}

pub(crate) fn parse_orientation_method(value: &str, path: &str) -> PyResult<RsOrientationMethod> {
    parse_enum(
        value,
        path,
        &[
            ("ring_fit", RsOrientationMethod::RingFit),
            ("disk_fit", RsOrientationMethod::DiskFit),
        ],
    )
}

pub(crate) fn chess_ring_str(v: RsChessRing) -> &'static str {
    match v {
        RsChessRing::Canonical => "canonical",
        RsChessRing::Broad => "broad",
        _ => "canonical",
    }
}

pub(crate) fn peak_fit_mode_str(v: RsPeakFitMode) -> &'static str {
    match v {
        RsPeakFitMode::Parabolic => "parabolic",
        RsPeakFitMode::Gaussian => "gaussian",
        _ => "gaussian",
    }
}

pub(crate) fn orientation_method_str(v: RsOrientationMethod) -> &'static str {
    match v {
        RsOrientationMethod::RingFit => "ring_fit",
        RsOrientationMethod::DiskFit => "disk_fit",
        _ => "ring_fit",
    }
}

// ---------------------------------------------------------------------------
// Shared json helpers.
// ---------------------------------------------------------------------------

pub(crate) fn json_dumps(
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

pub(crate) fn json_loads<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    let json = py.import("json")?;
    json.call_method1("loads", (text,))
}
