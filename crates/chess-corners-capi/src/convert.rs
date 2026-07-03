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
    ChessError, ChessRefiner, ChessRing, CornerDescriptor, DetectionStrategy, DetectorConfig,
    MultiscaleConfig, OrientationMethod, PeakFitMode, RadonConfig, UpscaleConfig,
};

use crate::{
    cc_axis, cc_chess_ring_t, cc_config, cc_corner, cc_peak_fit_t, cc_refiner_t, cc_status,
    CC_CHESS_RING_BROAD, CC_CHESS_RING_CANONICAL, CC_ORIENTATION_DISK_FIT, CC_ORIENTATION_NONE,
    CC_ORIENTATION_RING_FIT, CC_PEAK_FIT_GAUSSIAN, CC_PEAK_FIT_PARABOLIC,
    CC_REFINER_CENTER_OF_MASS, CC_REFINER_FORSTNER, CC_REFINER_SADDLE_POINT, CC_STRATEGY_CHESS,
    CC_STRATEGY_RADON,
};

/// Convert a flat [`cc_config`] into a facade [`DetectorConfig`].
///
/// The flat config carries every `DetectorConfig` knob except refiner
/// tuning (only the refiner *kind* is exposed) and the multiscale pyramid
/// detail behind the on/off switch. Strategy-specific fields that do not
/// apply to the active strategy (`chess_ring` under Radon; the Radon
/// geometry under ChESS) are ignored.
///
/// Returns [`cc_status::CC_ERR_INVALID_CONFIG`] on an unknown enum tag and
/// [`cc_status::CC_ERR_UPSCALE`] on an invalid `upscale_factor`.
pub(crate) fn to_detector_config(cfg: &cc_config) -> Result<DetectorConfig, cc_status> {
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

    // `0` is the explicit off-state; any other value is a fixed factor
    // validated by the facade's single source of truth for upscale
    // validity (invalid factors, including `1`, map to `CC_ERR_UPSCALE`).
    let upscale = match cfg.upscale_factor {
        0 => UpscaleConfig::disabled(),
        k => UpscaleConfig::fixed(k),
    };
    upscale.validate().map_err(|_| cc_status::CC_ERR_UPSCALE)?;

    // Build the active strategy and its strategy-specific knobs. The base
    // preset supplies the correct `DetectionStrategy` variant; every field
    // the flat config exposes is then set explicitly below.
    let base = match cfg.strategy {
        CC_STRATEGY_CHESS => {
            let refiner = match cfg.refiner {
                CC_REFINER_CENTER_OF_MASS => ChessRefiner::center_of_mass(),
                CC_REFINER_FORSTNER => ChessRefiner::forstner(),
                CC_REFINER_SADDLE_POINT => ChessRefiner::saddle_point(),
                _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
            };
            let ring = match cfg.chess_ring {
                CC_CHESS_RING_CANONICAL => ChessRing::Canonical,
                CC_CHESS_RING_BROAD => ChessRing::Broad,
                _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
            };
            DetectorConfig::chess().with_chess(|c| {
                c.ring = ring;
                c.refiner = refiner;
            })
        }
        // Radon has no pluggable refiner or ring (subpixel is its peak
        // fit); the `refiner` and `chess_ring` tags are ignored here.
        CC_STRATEGY_RADON => {
            let peak_fit = match cfg.peak_fit {
                CC_PEAK_FIT_PARABOLIC => PeakFitMode::Parabolic,
                CC_PEAK_FIT_GAUSSIAN => PeakFitMode::Gaussian,
                _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
            };
            DetectorConfig::radon().with_radon(|r| {
                r.ray_radius = cfg.ray_radius;
                r.image_upsample = cfg.image_upsample;
                r.response_blur_radius = cfg.response_blur_radius;
                r.peak_fit = peak_fit;
            })
        }
        _ => return Err(cc_status::CC_ERR_INVALID_CONFIG),
    };

    let base = base
        .with_threshold(cfg.threshold)
        .with_multiscale(multiscale)
        .with_upscale(upscale)
        .with_merge_radius(cfg.merge_radius)
        .with_detection(|d| {
            d.nms_radius = cfg.nms_radius;
            d.min_cluster_size = cfg.min_cluster_size;
        });
    Ok(match orientation_method {
        Some(method) => base.with_orientation_method(method),
        None => base.without_orientation(),
    })
}

/// Flatten a facade [`DetectorConfig`] into a [`cc_config`].
///
/// Every exposed knob is carried faithfully; only refiner tuning and the
/// multiscale pyramid detail are reduced (to the refiner kind and an
/// on/off switch respectively). Strategy-specific fields that do not apply
/// to the active strategy are filled with their strategy defaults so the
/// flat struct is always fully populated, and round-trip back through
/// [`to_detector_config`] to the original config.
pub(crate) fn flatten(config: &DetectorConfig) -> cc_config {
    let (strategy, refiner, chess_ring) = match config.strategy {
        DetectionStrategy::Chess(c) => (
            CC_STRATEGY_CHESS,
            chess_refiner_tag(c.refiner),
            chess_ring_tag(c.ring),
        ),
        // Radon has no pluggable refiner or ring; report the default tags
        // (ignored for Radon on the round-trip back).
        DetectionStrategy::Radon(_) => (
            CC_STRATEGY_RADON,
            CC_REFINER_CENTER_OF_MASS,
            CC_CHESS_RING_CANONICAL,
        ),
        _ => (
            CC_STRATEGY_CHESS,
            CC_REFINER_CENTER_OF_MASS,
            CC_CHESS_RING_CANONICAL,
        ),
    };
    // Radon geometry: from the active `RadonConfig`, else its defaults so a
    // ChESS config still flattens to a fully-populated struct.
    let radon = match config.strategy {
        DetectionStrategy::Radon(r) => r,
        _ => RadonConfig::default(),
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
    let upscale_factor = match config.upscale {
        UpscaleConfig::Disabled => 0,
        UpscaleConfig::Fixed(k) => k,
        // Future variants have no flat factor; report the off-state.
        _ => 0,
    };
    cc_config {
        strategy,
        threshold: config.threshold,
        nms_radius: config.detection.nms_radius,
        min_cluster_size: config.detection.min_cluster_size,
        refiner,
        orientation_method,
        multiscale,
        merge_radius: config.merge_radius,
        upscale_factor,
        chess_ring,
        ray_radius: radon.ray_radius,
        image_upsample: radon.image_upsample,
        response_blur_radius: radon.response_blur_radius,
        peak_fit: peak_fit_tag(radon.peak_fit),
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

fn chess_ring_tag(ring: ChessRing) -> cc_chess_ring_t {
    match ring {
        ChessRing::Canonical => CC_CHESS_RING_CANONICAL,
        ChessRing::Broad => CC_CHESS_RING_BROAD,
        // Future variants have no flat tag; fall back to the canonical ring.
        _ => CC_CHESS_RING_CANONICAL,
    }
}

fn peak_fit_tag(mode: PeakFitMode) -> cc_peak_fit_t {
    match mode {
        PeakFitMode::Parabolic => CC_PEAK_FIT_PARABOLIC,
        PeakFitMode::Gaussian => CC_PEAK_FIT_GAUSSIAN,
        // Future variants have no flat tag; fall back to the Radon default.
        _ => CC_PEAK_FIT_GAUSSIAN,
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
/// cannot in practice). Zero tags select single-scale ChESS with the
/// center-of-mass refiner, the canonical ring, no upscaling, and a zero
/// threshold. The Radon geometry fields carry harmless defaults (ignored
/// under the ChESS strategy).
pub(crate) fn zeroed_config() -> cc_config {
    let radon = RadonConfig::default();
    cc_config {
        strategy: CC_STRATEGY_CHESS,
        threshold: 0.0,
        nms_radius: 0,
        min_cluster_size: 0,
        refiner: CC_REFINER_CENTER_OF_MASS,
        orientation_method: CC_ORIENTATION_RING_FIT,
        multiscale: 0,
        merge_radius: 0.0,
        upscale_factor: 0,
        chess_ring: CC_CHESS_RING_CANONICAL,
        ray_radius: radon.ray_radius,
        image_upsample: radon.image_upsample,
        response_blur_radius: radon.response_blur_radius,
        peak_fit: peak_fit_tag(radon.peak_fit),
    }
}
