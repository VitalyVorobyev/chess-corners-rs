use chess_corners_core::{
    CenterOfMassConfig, OrientationMethod, PeakFitMode, RadonDetectorParams, SaddlePointConfig,
};

use crate::upscale::UpscaleConfig;

use super::{
    ChessConfig, ChessRing, DetectionStrategy, DetectorConfig, ImageRefiner, MultiscaleConfig,
    NoiseLevel, RadonConfig, Uncertainty,
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
    assert_eq!(chess.center_of_mass, CenterOfMassConfig::default());
    assert_eq!(cfg.refiner, None);
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
    assert_eq!(params.center_of_mass, CenterOfMassConfig::default());
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
    assert_eq!(levels, 2);
    assert_eq!(min_size, 128);
    assert_eq!(refinement_radius, 3);

    let cf = cfg
        .coarse_to_fine_params()
        .expect("radon_multiscale config must produce CoarseToFineParams");
    assert_eq!(cf.pyramid.num_levels, 2);
    assert_eq!(cf.pyramid.min_size, 128);
    assert_eq!(cf.refinement_radius, 3);
    assert_eq!(cf.merge_radius, 3.0);
}

#[test]
fn broad_ring_and_center_of_mass_propagate_to_params() {
    // `CenterOfMassConfig` is `#[non_exhaustive]`; build via `Default`
    // and set the field under test.
    let mut com = CenterOfMassConfig::default();
    com.radius = 3;
    let cfg = DetectorConfig {
        strategy: DetectionStrategy::Chess(ChessConfig {
            ring: ChessRing::Broad,
            center_of_mass: com,
        }),
        ..DetectorConfig::chess()
    };

    let params = cfg.chess_params();
    assert!(params.use_radius10);
    assert_eq!(params.center_of_mass, com);
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
fn image_refiner_defaults_to_none_and_round_trips() {
    for preset in [
        DetectorConfig::chess(),
        DetectorConfig::chess_multiscale(),
        DetectorConfig::radon(),
        DetectorConfig::radon_multiscale(),
    ] {
        assert_eq!(preset.refiner, None);
        assert!(serde_json::to_value(preset).unwrap()["refiner"].is_null());
    }
    // Configs written without the field keep the detector's own localizer.
    let old: DetectorConfig = serde_json::from_str(r#"{"threshold": 30.0}"#).unwrap();
    assert_eq!(old.refiner, None);

    let mut saddle = SaddlePointConfig::default();
    saddle.max_offset = 0.75;
    for base in [DetectorConfig::chess(), DetectorConfig::radon()] {
        let cfg = base.with_refiner(ImageRefiner::SaddlePoint(saddle));
        let json = serde_json::to_value(cfg).unwrap();
        assert_eq!(json["refiner"]["saddle_point"]["max_offset"], 0.75);
        let back: DetectorConfig = serde_json::from_value(json).unwrap();
        assert_eq!(back, cfg);
        assert_eq!(back.without_refiner().refiner, None);
    }
}

#[test]
fn refiner_survives_a_strategy_switch() {
    let cfg = DetectorConfig::chess()
        .with_refiner(ImageRefiner::saddle_point())
        .with_radon(|r| r.ray_radius = 5);
    assert_eq!(cfg.refiner, Some(ImageRefiner::saddle_point()));
    let cfg = cfg.with_chess(|c| c.ring = ChessRing::Broad);
    assert_eq!(cfg.refiner, Some(ImageRefiner::saddle_point()));
}

#[test]
fn removed_refiner_shapes_are_rejected() {
    for json in [
        r#"{"strategy": {"chess": {"refiner": {"center_of_mass": {"radius": 2}}}}}"#,
        r#"{"strategy": {"radon": {"refiner": "peak_fit"}}}"#,
        r#"{"refiner": "center_of_mass"}"#,
        r#"{"refiner": "peak_fit"}"#,
    ] {
        assert!(
            serde_json::from_str::<DetectorConfig>(json).is_err(),
            "{json} must not parse"
        );
    }
}

#[test]
fn unknown_keys_are_rejected_and_named_at_every_level() {
    for (json, key) in [
        (r#"{"noise_sigma": 1.5}"#, "noise_sigma"),
        (
            r#"{"strategy": {"chess": {"refiner": "center_of_mass"}}}"#,
            "refiner",
        ),
        (r#"{"strategy": {"radon": {"ray_radus": 5}}}"#, "ray_radus"),
        (
            r#"{"strategy": {"chess": {"center_of_mass": {"radius": 2, "r": 1}}}}"#,
            "r",
        ),
        (r#"{"detection": {"nms": 3}}"#, "nms"),
        (
            r#"{"multiscale": {"pyramid": {"levels": 3, "min_size": 128, "refinement_radius": 3, "merge_radius": 3.0}}}"#,
            "merge_radius",
        ),
        (
            r#"{"refiner": {"saddle_point": {"max_offset": 1.0, "radius_px": 2}}}"#,
            "radius_px",
        ),
        (
            r#"{"uncertainty": {"propagated": {"noise": "estimate", "sigma": 1.0}}}"#,
            "sigma",
        ),
    ] {
        let err = serde_json::from_str::<DetectorConfig>(json)
            .expect_err(json)
            .to_string();
        assert!(
            err.contains(&format!("unknown field `{key}`")),
            "{json}: {err}"
        );
    }
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
fn image_refiner_round_trips_each_variant() {
    let variants = [(ImageRefiner::saddle_point(), r#"{"saddle_point":"#)];
    for (v, prefix) in variants {
        let json = serde_json::to_string(&v).expect("serialize image refiner");
        assert!(json.starts_with(prefix), "{json}");
        let decoded: ImageRefiner = serde_json::from_str(&json).expect("deserialize image refiner");
        assert_eq!(decoded, v);
    }
}

#[test]
fn uncertainty_json_shapes() {
    let shapes = [
        (Uncertainty::Off, r#""off""#),
        (
            Uncertainty::Propagated {
                noise: NoiseLevel::Estimate,
            },
            r#"{"propagated":{"noise":"estimate"}}"#,
        ),
        (
            Uncertainty::Propagated {
                noise: NoiseLevel::Known(1.25),
            },
            r#"{"propagated":{"noise":{"known":1.25}}}"#,
        ),
    ];
    for (value, json) in shapes {
        assert_eq!(serde_json::to_string(&value).unwrap(), json);
        assert_eq!(serde_json::from_str::<Uncertainty>(json).unwrap(), value);
    }
    // A bare "propagated" and a missing noise level are rejected, not
    // defaulted.
    for bad in [
        r#""propagated""#,
        r#"{"propagated":{}}"#,
        r#"{"propagated":{"noise":"known"}}"#,
    ] {
        assert!(serde_json::from_str::<Uncertainty>(bad).is_err(), "{bad}");
    }
    // The map form of the unit variant, as the Python `to_dict` writes it.
    assert_eq!(
        serde_json::from_str::<Uncertainty>(r#"{"off":null}"#).unwrap(),
        Uncertainty::Off
    );
    assert_eq!(DetectorConfig::default().uncertainty, Uncertainty::Off);
}

#[test]
fn known_noise_level_is_validated() {
    let known = |sigma| {
        DetectorConfig::chess().with_uncertainty(Uncertainty::Propagated {
            noise: NoiseLevel::Known(sigma),
        })
    };
    for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            known(bad).validate(),
            Err(crate::ChessError::InvalidNoiseSigma(_))
        ));
    }
    assert!(known(0.0).validate().is_ok());
    assert!(DetectorConfig::chess().validate().is_ok());
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
    assert_eq!(chess.center_of_mass, CenterOfMassConfig::default());
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
        .with_refiner(ImageRefiner::saddle_point());
    assert_eq!(cfg.threshold, 0.15);
    assert_eq!(
        cfg.refiner,
        Some(ImageRefiner::SaddlePoint(SaddlePointConfig::default()))
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
fn image_refiner_shortcut_equals_full_constructor() {
    assert_eq!(
        ImageRefiner::saddle_point(),
        ImageRefiner::SaddlePoint(SaddlePointConfig::default())
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
