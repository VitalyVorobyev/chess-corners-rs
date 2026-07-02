use chess_corners_core::{
    CenterOfMassConfig, ForstnerConfig, OrientationMethod, PeakFitMode, RadonDetectorParams,
    RefinerKind, SaddlePointConfig,
};

use crate::upscale::UpscaleConfig;

use super::{
    ChessConfig, ChessRefiner, ChessRing, DetectionStrategy, DetectorConfig, MultiscaleConfig,
    RadonConfig,
};

fn assert_strategy_chess(cfg: &DetectorConfig) -> &ChessConfig {
    match &cfg.strategy {
        DetectionStrategy::Chess(c) => c,
        other => panic!("expected ChESS strategy, got {other:?}"),
    }
}

fn assert_strategy_radon(cfg: &DetectorConfig) -> &RadonConfig {
    match &cfg.strategy {
        DetectionStrategy::Radon(r) => r,
        other => panic!("expected Radon strategy, got {other:?}"),
    }
}

#[test]
fn default_is_single_scale_chess_with_denoise_floor() {
    let cfg = DetectorConfig::default();
    let chess = assert_strategy_chess(&cfg);
    assert_eq!(chess.ring, ChessRing::Canonical);
    assert_eq!(
        chess.refiner,
        ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
    );
    assert_eq!(cfg.detection.nms_radius, 2);
    assert_eq!(cfg.detection.min_cluster_size, 2);
    assert_eq!(cfg.multiscale, MultiscaleConfig::SingleScale);
    assert_eq!(cfg.upscale, UpscaleConfig::Disabled);
    assert_eq!(cfg.threshold, 30.0);
    assert_eq!(cfg.merge_radius, 3.0);
    assert!(cfg.coarse_to_fine_params().is_none());

    let params = cfg.chess_params();
    assert!(!params.use_radius10);
    assert_eq!(params.threshold, 30.0);
    assert_eq!(params.nms_radius, 2);
    assert_eq!(params.min_cluster_size, 2);
    assert_eq!(
        params.refiner,
        RefinerKind::CenterOfMass(CenterOfMassConfig::default())
    );
}

#[test]
fn chess_threshold_lowers_to_absolute_floor() {
    let cfg = DetectorConfig {
        threshold: 7.5,
        ..DetectorConfig::chess()
    };
    let params = cfg.chess_params();
    assert_eq!(params.threshold, 7.5);
}

#[test]
fn radon_threshold_lowers_to_relative_fraction() {
    let cfg = DetectorConfig {
        threshold: 0.15,
        ..DetectorConfig::radon()
    };
    let params = cfg.radon_detector_params();
    assert!((params.threshold_rel - 0.15).abs() < f32::EPSILON);
}

#[test]
fn chess_multiscale_preset_carries_pyramid_params() {
    let cfg = DetectorConfig::chess_multiscale();
    let MultiscaleConfig::Pyramid {
        levels,
        min_size,
        refinement_radius,
    } = cfg.multiscale
    else {
        panic!("chess_multiscale preset must carry Pyramid params");
    };
    assert_eq!(levels, 3);
    assert_eq!(min_size, 128);
    assert_eq!(refinement_radius, 3);

    let cf = cfg
        .coarse_to_fine_params()
        .expect("chess_multiscale config must produce CoarseToFineParams");
    assert_eq!(cf.pyramid.num_levels, 3);
    assert_eq!(cf.pyramid.min_size, 128);
    assert_eq!(cf.refinement_radius, 3);
    assert_eq!(cf.merge_radius, 3.0);
}

#[test]
fn radon_preset_uses_radon_config_and_relative_threshold() {
    let cfg = DetectorConfig::radon();
    let radon = assert_strategy_radon(&cfg);
    assert_eq!(radon.ray_radius, 4);
    assert_eq!(radon.image_upsample, 2);
    assert_eq!(radon.response_blur_radius, 1);
    assert_eq!(radon.peak_fit, PeakFitMode::Gaussian);
    assert_eq!(cfg.detection.nms_radius, 4);
    assert_eq!(cfg.detection.min_cluster_size, 2);
    assert_eq!(cfg.threshold, RadonDetectorParams::DEFAULT_THRESHOLD_REL);
    assert_eq!(cfg.multiscale, MultiscaleConfig::SingleScale);
    assert!(cfg.coarse_to_fine_params().is_none());

    let radon_params = cfg.radon_detector_params();
    assert_eq!(radon_params.ray_radius, 4);
    assert_eq!(radon_params.image_upsample, 2);
    assert_eq!(radon_params.nms_radius, 4);
    assert_eq!(radon_params.min_cluster_size, 2);
    assert!(
        (radon_params.threshold_rel - RadonDetectorParams::DEFAULT_THRESHOLD_REL).abs()
            < f32::EPSILON
    );
}

#[test]
fn radon_multiscale_preset_carries_pyramid_params() {
    let cfg = DetectorConfig::radon_multiscale();
    assert_strategy_radon(&cfg);
    assert_eq!(cfg.threshold, RadonDetectorParams::DEFAULT_THRESHOLD_REL);
    let MultiscaleConfig::Pyramid {
        levels,
        min_size,
        refinement_radius,
    } = cfg.multiscale
    else {
        panic!("radon_multiscale preset must carry Pyramid params");
    };
    assert_eq!(levels, 3);
    assert_eq!(min_size, 128);
    assert_eq!(refinement_radius, 3);

    let cf = cfg
        .coarse_to_fine_params()
        .expect("radon_multiscale config must produce CoarseToFineParams");
    assert_eq!(cf.pyramid.num_levels, 3);
    assert_eq!(cf.pyramid.min_size, 128);
    assert_eq!(cf.refinement_radius, 3);
    assert_eq!(cf.merge_radius, 3.0);
}

#[test]
fn broad_ring_and_forstner_refiner_propagate_to_params() {
    // `ForstnerConfig` is `#[non_exhaustive]`; build via `Default`
    // and set the field under test.
    let mut forstner = ForstnerConfig::default();
    forstner.max_offset = 2.0;
    let cfg = DetectorConfig {
        strategy: DetectionStrategy::Chess(ChessConfig {
            ring: ChessRing::Broad,
            refiner: ChessRefiner::Forstner(forstner),
            ..ChessConfig::default()
        }),
        ..DetectorConfig::chess()
    };

    let params = cfg.chess_params();
    assert!(params.use_radius10);
    assert_eq!(params.refiner, RefinerKind::Forstner(forstner));
}

#[test]
fn chess_preset_round_trips_through_serde() {
    let cfg = DetectorConfig::chess();
    let json = serde_json::to_string(&cfg).expect("serialize chess config");
    let decoded: DetectorConfig = serde_json::from_str(&json).expect("deserialize chess config");
    assert_eq!(decoded, cfg);
}

#[test]
fn chess_multiscale_preset_round_trips_through_serde() {
    let cfg = DetectorConfig::chess_multiscale();
    let json = serde_json::to_string(&cfg).expect("serialize chess_multiscale config");
    let decoded: DetectorConfig =
        serde_json::from_str(&json).expect("deserialize chess_multiscale config");
    assert_eq!(decoded, cfg);
}

#[test]
fn radon_preset_round_trips_through_serde() {
    let cfg = DetectorConfig::radon();
    let json = serde_json::to_string(&cfg).expect("serialize radon config");
    let decoded: DetectorConfig = serde_json::from_str(&json).expect("deserialize radon config");
    assert_eq!(decoded, cfg);
}

#[test]
fn radon_multiscale_preset_round_trips_through_serde() {
    let cfg = DetectorConfig::radon_multiscale();
    let json = serde_json::to_string(&cfg).expect("serialize radon_multiscale config");
    let decoded: DetectorConfig =
        serde_json::from_str(&json).expect("deserialize radon_multiscale config");
    assert_eq!(decoded, cfg);
}

#[test]
fn threshold_serializes_as_a_bare_number() {
    let cfg = DetectorConfig::chess().with_threshold(3.5);
    let json = serde_json::to_string(&cfg).expect("serialize config");
    assert!(json.contains("\"threshold\":3.5"));
    let decoded: DetectorConfig = serde_json::from_str(&json).expect("deserialize config");
    assert_eq!(decoded.threshold, 3.5);
}

#[test]
fn multiscale_config_round_trips_with_externally_tagged_payload() {
    let single = MultiscaleConfig::SingleScale;
    let single_json = serde_json::to_string(&single).expect("serialize single-scale");
    assert!(single_json.contains("single_scale"));
    let decoded: MultiscaleConfig =
        serde_json::from_str(&single_json).expect("deserialize single-scale");
    assert_eq!(decoded, single);

    let pyramid = MultiscaleConfig::Pyramid {
        levels: 3,
        min_size: 128,
        refinement_radius: 3,
    };
    let pyramid_json = serde_json::to_string(&pyramid).expect("serialize pyramid");
    assert!(pyramid_json.contains("pyramid"));
    let decoded: MultiscaleConfig =
        serde_json::from_str(&pyramid_json).expect("deserialize pyramid");
    assert_eq!(decoded, pyramid);
}

#[test]
fn chess_refiner_round_trips_each_variant() {
    let variants = [
        ChessRefiner::CenterOfMass(CenterOfMassConfig::default()),
        ChessRefiner::Forstner(ForstnerConfig::default()),
        ChessRefiner::SaddlePoint(SaddlePointConfig::default()),
    ];
    for v in variants {
        let json = serde_json::to_string(&v).expect("serialize chess refiner");
        let decoded: ChessRefiner = serde_json::from_str(&json).expect("deserialize chess refiner");
        assert_eq!(decoded, v);
    }
}

#[test]
fn unit_enum_variants_serialize_as_bare_strings() {
    // Codifies the externally-tagged serde encoding for unit variants.
    // The Python from_dict paths must accept these bare strings produced
    // by serde so that Rust→JSON→Python round-trips work end-to-end.
    let json = serde_json::to_string(&MultiscaleConfig::SingleScale).unwrap();
    assert_eq!(json, "\"single_scale\"");

    let json = serde_json::to_string(&UpscaleConfig::Disabled).unwrap();
    assert_eq!(json, "\"disabled\"");
}

#[test]
fn with_chess_mutates_in_place_when_strategy_is_chess() {
    let cfg = DetectorConfig::chess().with_chess(|c| c.ring = ChessRing::Broad);
    let chess = assert_strategy_chess(&cfg);
    assert_eq!(chess.ring, ChessRing::Broad);
    // Other chess fields untouched
    assert_eq!(
        chess.refiner,
        ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
    );
}

#[test]
fn with_chess_replaces_radon_preserves_threshold() {
    let cfg = DetectorConfig::radon()
        .with_threshold(5.0)
        .with_chess(|c| c.ring = ChessRing::Broad);
    // Strategy replaced with chess
    let chess = assert_strategy_chess(&cfg);
    assert_eq!(chess.ring, ChessRing::Broad);
    // Top-level threshold preserved
    assert_eq!(cfg.threshold, 5.0);
}

#[test]
fn with_radon_mutates_in_place_when_strategy_is_radon() {
    let cfg = DetectorConfig::radon().with_radon(|r| r.ray_radius = 9);
    let radon = assert_strategy_radon(&cfg);
    assert_eq!(radon.ray_radius, 9);
    // Other radon fields untouched
    assert_eq!(radon.image_upsample, 2);
}

#[test]
fn with_radon_replaces_chess_preserves_threshold() {
    let cfg = DetectorConfig::chess()
        .with_threshold(0.5)
        .with_radon(|r| r.ray_radius = 6);
    let radon = assert_strategy_radon(&cfg);
    assert_eq!(radon.ray_radius, 6);
    // Threshold preserved
    assert_eq!(cfg.threshold, 0.5);
}

#[test]
fn with_detection_sets_shared_params_for_both_strategies() {
    // ChESS strategy: shared detection params flow into ChessParams.
    let cfg = DetectorConfig::chess().with_detection(|d| {
        d.nms_radius = 7;
        d.min_cluster_size = 4;
    });
    assert_eq!(cfg.detection.nms_radius, 7);
    assert_eq!(cfg.detection.min_cluster_size, 4);
    let params = cfg.chess_params();
    assert_eq!(params.nms_radius, 7);
    assert_eq!(params.min_cluster_size, 4);

    // Radon strategy: same shared params flow into RadonDetectorParams.
    let cfg = DetectorConfig::radon().with_detection(|d| d.nms_radius = 9);
    assert_eq!(cfg.detection.nms_radius, 9);
    let params = cfg.radon_detector_params();
    assert_eq!(params.nms_radius, 9);
}

#[test]
fn chained_builder_produces_expected_state() {
    let cfg = DetectorConfig::chess()
        .with_threshold(0.15)
        .with_chess(|c| c.refiner = ChessRefiner::forstner());
    assert_eq!(cfg.threshold, 0.15);
    let chess = assert_strategy_chess(&cfg);
    assert_eq!(
        chess.refiner,
        ChessRefiner::Forstner(ForstnerConfig::default())
    );
}

#[test]
fn with_multiscale_sets_multiscale() {
    let cfg = DetectorConfig::chess().with_multiscale(MultiscaleConfig::pyramid_default());
    assert_eq!(
        cfg.multiscale,
        MultiscaleConfig::Pyramid {
            levels: 3,
            min_size: 128,
            refinement_radius: 3
        }
    );
}

#[test]
fn with_upscale_sets_upscale() {
    let cfg = DetectorConfig::chess().with_upscale(UpscaleConfig::Fixed(2));
    assert_eq!(cfg.upscale, UpscaleConfig::Fixed(2));
}

#[test]
fn with_orientation_method_sets_method() {
    let method = OrientationMethod::DiskFit;
    let cfg = DetectorConfig::chess().with_orientation_method(method);
    assert_eq!(cfg.orientation_method, Some(method));
}

#[test]
fn without_orientation_clears_method() {
    let cfg = DetectorConfig::chess().without_orientation();
    assert_eq!(cfg.orientation_method, None);
    // Lowering carries the `None` through to the low-level params.
    assert_eq!(cfg.chess_params().orientation_method, None);
}

#[test]
fn default_orientation_method_is_some_ring_fit() {
    let cfg = DetectorConfig::chess();
    assert_eq!(cfg.orientation_method, Some(OrientationMethod::RingFit));
}

#[test]
fn with_merge_radius_sets_radius() {
    let cfg = DetectorConfig::chess().with_merge_radius(5.0);
    assert!((cfg.merge_radius - 5.0).abs() < f32::EPSILON);
}

#[test]
fn chess_refiner_shortcuts_equal_full_constructors() {
    assert_eq!(
        ChessRefiner::center_of_mass(),
        ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
    );
    assert_eq!(
        ChessRefiner::forstner(),
        ChessRefiner::Forstner(ForstnerConfig::default())
    );
    assert_eq!(
        ChessRefiner::saddle_point(),
        ChessRefiner::SaddlePoint(SaddlePointConfig::default())
    );
}

#[test]
fn multiscale_config_pyramid_default_equals_literal() {
    assert_eq!(
        MultiscaleConfig::pyramid_default(),
        MultiscaleConfig::Pyramid {
            levels: 3,
            min_size: 128,
            refinement_radius: 3
        }
    );
}

#[cfg(feature = "ml-refiner")]
#[test]
fn chess_refiner_ml_serializes_as_bare_string() {
    let json = serde_json::to_string(&ChessRefiner::Ml).unwrap();
    assert_eq!(json, "\"ml\"");
    let decoded: ChessRefiner = serde_json::from_str(&json).expect("deserialize ml refiner");
    assert_eq!(decoded, ChessRefiner::Ml);
}
