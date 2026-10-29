//! PyO3 config module re-exports.
//!
//! Submodules hold the actual struct/impl blocks; this module re-exports
//! every public type so that `crate::config::TypeName` paths continue to
//! resolve unchanged.

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;

pub(crate) mod detector;
pub(crate) mod multiscale;
pub(crate) mod parse;
pub(crate) mod refiners;
pub(crate) mod strategy;
pub(crate) mod uncertainty;
pub(crate) mod upscale;

create_exception!(_native, ConfigError, PyValueError);

pub use detector::DetectorConfig;
pub use multiscale::MultiscaleConfig;
pub use parse::{ChessRing, OnRefinerReject, OrientationMethod, PeakFitMode};
pub use refiners::{CenterOfMassConfig, ImageRefiner, SaddlePointConfig};
pub use strategy::{ChessConfig, DetectionParams, DetectionStrategy, RadonConfig};
pub use uncertainty::Uncertainty;
pub use upscale::UpscaleConfig;
