//! Conversions between the flat C ABI types and the `chess-corners` facade
//! types.
//!
//! # Ownership
//!
//! Corner arrays returned through [`crate::cc_result`] are allocated by
//! [`crate::cc_detect_u8`] as a `Box<[cc_corner]>` and released by
//! [`crate::cc_result_free`], which reconstructs and drops that exact box.
//! The library both allocates and frees the array, so it never crosses an
//! allocator boundary.
//!
//! # Reentrancy
//!
//! Every function here is a pure transformation of its arguments and holds
//! no global state, so the C ABI built on top of them is fully reentrant.
//!
//! # Panic-safety
//!
//! Conversions never panic on adversarial C input: out-of-range enum tags
//! map to [`crate::cc_status::CC_ERR_INVALID_CONFIG`] instead of panicking.
//! Detection itself is wrapped in `catch_unwind` by the callers.

use chess_corners::{
    ChessError, ChessRefiner, CornerDescriptor, DetectionStrategy, DetectorConfig,
    MultiscaleConfig, OrientationMethod, RadonRefiner,
};

use crate::{
    cc_axis, cc_config, cc_corner, cc_refiner_t, cc_status, CC_ORIENTATION_DISK_FIT,
    CC_ORIENTATION_NONE, CC_ORIENTATION_RING_FIT, CC_REFINER_CENTER_OF_MASS, CC_REFINER_FORSTNER,
    CC_REFINER_RADON_PEAK, CC_REFINER_SADDLE_POINT, CC_STRATEGY_CHESS, CC_STRATEGY_RADON,
};

/// Convert a flat [`cc_config`] into a facade [`DetectorConfig`].
///
/// The flat config fully determines the exposed knobs (strategy, threshold,
/// NMS / clustering, refiner kind, orientation method, multiscale on/off).
/// Knobs it does not carry — per-strategy ring/ray geometry, refiner
/// tuning, upscaling, and the cross-level merge radius — are taken from the
/// selected strategy preset.
pub(crate) fn to_detector_config(cfg: &cc_config) -> Result<DetectorConfig, cc_status> {
    let threshold = cfg.threshold;

    let multiscale = if cfg.multiscale != 0 {
        MultiscaleConfig::pyramid_default()
    } else {
        MultiscaleConfig::SingleScale
    };

    let orientation_method: Option<OrientationMethod> = match cfg.orientation_method {
        CC_ORIENTATION_RING_FIT => Some(OrientationMethod::RingFit),
        CC_ORIENTATION_DISK_FIT => Some(OrientationMethod::DiskFit),
        CC_ORIENTATION_NONE => None,
        _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
    };

    let nms_radius = cfg.nms_radius;
    let min_cluster_size = cfg.min_cluster_size;

    // Start from the strategy preset (which carries the per-strategy nested
    // defaults the flat config does not expose), then set the chosen refiner.
    let base = match cfg.strategy {
        CC_STRATEGY_CHESS => {
            let refiner = match cfg.refiner {
                CC_REFINER_CENTER_OF_MASS => ChessRefiner::center_of_mass(),
                CC_REFINER_FORSTNER => ChessRefiner::forstner(),
                CC_REFINER_SADDLE_POINT => ChessRefiner::saddle_point(),
                _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
            };
            DetectorConfig::chess().with_chess(|c| c.refiner = refiner)
        }
        CC_STRATEGY_RADON => {
            let refiner = match cfg.refiner {
                CC_REFINER_CENTER_OF_MASS => RadonRefiner::center_of_mass(),
                CC_REFINER_RADON_PEAK => RadonRefiner::radon_peak(),
                _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
            };
            DetectorConfig::radon().with_radon(|r| r.refiner = refiner)
        }
        _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
    };

    let base = base
        .with_threshold(threshold)
        .with_multiscale(multiscale)
        .with_detection(|d| {
            d.nms_radius = nms_radius;
            d.min_cluster_size = min_cluster_size;
        });
    Ok(match orientation_method {
        Some(method) => base.with_orientation_method(method),
        None => base.without_orientation(),
    })
}

/// Flatten a facade [`DetectorConfig`] into a [`cc_config`].
///
/// Lossy by design: per-strategy geometry, refiner tuning, upscaling, and
/// the merge radius are dropped. For the library presets the flattened tags
/// round-trip back through [`to_detector_config`] to the original config.
pub(crate) fn flatten(config: &DetectorConfig) -> cc_config {
    let (strategy, refiner) = match config.strategy {
        DetectionStrategy::Chess(c) => (CC_STRATEGY_CHESS, chess_refiner_tag(c.refiner)),
        DetectionStrategy::Radon(r) => (CC_STRATEGY_RADON, radon_refiner_tag(r.refiner)),
        _ => (CC_STRATEGY_CHESS, CC_REFINER_CENTER_OF_MASS),
    };
    let multiscale = match config.multiscale {
        MultiscaleConfig::SingleScale => 0,
        _ => 1,
    };
    let orientation_method = match config.orientation_method {
        None => CC_ORIENTATION_NONE,
        Some(OrientationMethod::RingFit) => CC_ORIENTATION_RING_FIT,
        Some(OrientationMethod::DiskFit) => CC_ORIENTATION_DISK_FIT,
        Some(_) => CC_ORIENTATION_RING_FIT,
    };
    cc_config {
        strategy,
        threshold: config.threshold,
        nms_radius: config.detection.nms_radius,
        min_cluster_size: config.detection.min_cluster_size,
        refiner,
        orientation_method,
        multiscale,
    }
}

fn chess_refiner_tag(refiner: ChessRefiner) -> cc_refiner_t {
    match refiner {
        ChessRefiner::CenterOfMass(_) => CC_REFINER_CENTER_OF_MASS,
        ChessRefiner::Forstner(_) => CC_REFINER_FORSTNER,
        ChessRefiner::SaddlePoint(_) => CC_REFINER_SADDLE_POINT,
        // Future / feature-gated variants (e.g. the ML refiner) have no flat
        // tag; fall back to the universally valid center-of-mass tag.
        _ => CC_REFINER_CENTER_OF_MASS,
    }
}

fn radon_refiner_tag(refiner: RadonRefiner) -> cc_refiner_t {
    match refiner {
        RadonRefiner::RadonPeak(_) => CC_REFINER_RADON_PEAK,
        RadonRefiner::CenterOfMass(_) => CC_REFINER_CENTER_OF_MASS,
        _ => CC_REFINER_CENTER_OF_MASS,
    }
}

/// Convert a facade [`CornerDescriptor`] into a [`cc_corner`].
///
/// When the descriptor's orientation fit was skipped (`axes` is `None`),
/// `has_orientation` is `0` and `axes` is zeroed; otherwise `has_orientation`
/// is `1` and `axes` carries the fitted directions.
pub(crate) fn corner_to_ffi(corner: &CornerDescriptor) -> cc_corner {
    match corner.axes {
        Some(axes) => cc_corner {
            x: corner.x,
            y: corner.y,
            response: corner.response,
            axes: [
                cc_axis {
                    angle: axes[0].angle,
                    sigma: axes[0].sigma,
                },
                cc_axis {
                    angle: axes[1].angle,
                    sigma: axes[1].sigma,
                },
            ],
            has_orientation: 1,
        },
        None => cc_corner {
            x: corner.x,
            y: corner.y,
            response: corner.response,
            axes: [cc_axis {
                angle: 0.0,
                sigma: 0.0,
            }; 2],
            has_orientation: 0,
        },
    }
}

/// Map a facade [`ChessError`] onto a [`cc_status`].
pub(crate) fn map_error(err: &ChessError) -> cc_status {
    match err {
        ChessError::DimensionMismatch { .. } => cc_status::CC_ERR_DIMENSION_MISMATCH,
        ChessError::Upscale(_) => cc_status::CC_ERR_UPSCALE,
        _ => cc_status::CC_ERR_INVALID_CONFIG,
    }
}

/// Fallback config returned only if a preset constructor were to panic (it
/// cannot in practice). All-zero tags select single-scale ChESS with the
/// center-of-mass refiner and a zero threshold.
pub(crate) fn zeroed_config() -> cc_config {
    cc_config {
        strategy: CC_STRATEGY_CHESS,
        threshold: 0.0,
        nms_radius: 0,
        min_cluster_size: 0,
        refiner: CC_REFINER_CENTER_OF_MASS,
        orientation_method: CC_ORIENTATION_RING_FIT,
        multiscale: 0,
    }
}
