//! `ChessConfig::on_refiner_reject`: what a ChESS detection does with the
//! corners the image refiner rejects.
//!
//! Single-scale, `KeepCenterOfMass` must return the `Drop` output unchanged,
//! bit for bit, plus every rejected corner at exactly the position the center
//! of mass alone gives it. Under a pyramid the coarse seeds follow the same
//! rule, so a kept seed opens a base-image window the image refiner may then
//! refine; there the `Drop` corners must survive and nothing may be lost.

use chess_corners::{
    CornerDescriptor, Detector, DetectorConfig, ImageRefiner, MultiscaleConfig, NoiseLevel,
    OnRefinerReject, Uncertainty,
};

fn load(name: &str) -> image::GrayImage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testimages")
        .join(name);
    image::open(&path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .to_luma8()
}

fn detect(img: &image::GrayImage, cfg: DetectorConfig) -> Vec<CornerDescriptor> {
    Detector::new(cfg)
        .expect("valid config")
        .detect(img)
        .expect("detection runs")
}

fn key(c: &CornerDescriptor) -> [u32; 2] {
    [c.x.to_bits(), c.y.to_bits()]
}

fn configs(base: DetectorConfig) -> [DetectorConfig; 3] {
    let saddle = base.with_refiner(ImageRefiner::saddle_point());
    let keep = saddle.with_chess(|c| c.on_refiner_reject = OnRefinerReject::KeepCenterOfMass);
    [base, saddle, keep]
}

fn check(img: &image::GrayImage, base: DetectorConfig, name: &str) -> usize {
    let [com_cfg, drop_cfg, keep_cfg] = configs(base);
    let com = detect(img, com_cfg);
    let dropped = detect(img, drop_cfg);
    let kept = detect(img, keep_cfg);

    // Every corner the image refiner accepted is unchanged.
    let kept_keys: std::collections::HashSet<[u32; 2]> = kept.iter().map(key).collect();
    for c in &dropped {
        assert!(kept_keys.contains(&key(c)), "{name}: refined corner moved");
    }
    // Every other corner sits exactly where the center of mass puts it.
    let dropped_keys: std::collections::HashSet<[u32; 2]> = dropped.iter().map(key).collect();
    let com_keys: std::collections::HashSet<[u32; 2]> = com.iter().map(key).collect();
    let extra: Vec<&CornerDescriptor> = kept
        .iter()
        .filter(|c| !dropped_keys.contains(&key(c)))
        .collect();
    for c in &extra {
        assert!(
            com_keys.contains(&key(c)),
            "{name}: kept corner ({}, {}) is not a center-of-mass position",
            c.x,
            c.y
        );
    }
    assert_eq!(kept.len(), dropped.len() + extra.len(), "{name}");
    extra.len()
}

#[test]
fn keep_center_of_mass_adds_the_rejected_corners_and_changes_nothing_else() {
    let mut recovered = 0;
    for name in ["small.png", "mid.png", "large.png"] {
        let img = load(name);
        recovered += check(&img, DetectorConfig::chess(), name);
    }
    // The test images include peaks the saddle-point refiner rejects;
    // otherwise this test would not exercise the fallback at all.
    assert!(recovered > 0, "no rejected corner was exercised");
}

#[test]
fn under_a_pyramid_keep_center_of_mass_loses_no_refined_corner() {
    let pyramid = DetectorConfig::chess().with_multiscale(MultiscaleConfig::pyramid(3, 128, 3));
    for name in ["small.png", "mid.png", "large.png"] {
        let img = load(name);
        let [_, drop_cfg, keep_cfg] = configs(pyramid);
        let dropped = detect(&img, drop_cfg);
        let kept = detect(&img, keep_cfg);
        let kept_keys: std::collections::HashSet<[u32; 2]> = kept.iter().map(key).collect();
        for c in &dropped {
            assert!(kept_keys.contains(&key(c)), "{name}: refined corner lost");
        }
        assert!(kept.len() >= dropped.len(), "{name}");
    }
}

#[test]
fn the_default_serializes_as_an_absent_key() {
    let json = serde_json::to_value(DetectorConfig::chess()).expect("serialize");
    assert!(json["strategy"]["chess"].get("on_refiner_reject").is_none());
    let keep = DetectorConfig::chess()
        .with_chess(|c| c.on_refiner_reject = OnRefinerReject::KeepCenterOfMass);
    let json = serde_json::to_value(keep).expect("serialize");
    assert_eq!(
        json["strategy"]["chess"]["on_refiner_reject"],
        "keep_center_of_mass"
    );
    let back: DetectorConfig = serde_json::from_value(json).expect("deserialize");
    assert_eq!(back, keep);
}

#[test]
fn drop_is_the_default_and_the_option_is_ignored_without_an_image_refiner() {
    assert_eq!(
        chess_corners::ChessConfig::default().on_refiner_reject,
        OnRefinerReject::Drop
    );
    let img = load("mid.png");
    let plain = detect(&img, DetectorConfig::chess());
    let keep_without_refiner = detect(
        &img,
        DetectorConfig::chess()
            .with_chess(|c| c.on_refiner_reject = OnRefinerReject::KeepCenterOfMass),
    );
    assert_eq!(
        plain.iter().map(key).collect::<Vec<_>>(),
        keep_without_refiner.iter().map(key).collect::<Vec<_>>()
    );
}

#[test]
fn kept_corners_carry_the_center_of_mass_covariance() {
    let img = load("mid.png");
    let propagated = DetectorConfig::chess().with_uncertainty(Uncertainty::Propagated {
        noise: NoiseLevel::Known(1.0),
    });
    let [com_cfg, drop_cfg, keep_cfg] = configs(propagated);
    let com = detect(&img, com_cfg);
    let dropped = detect(&img, drop_cfg);
    let kept = detect(&img, keep_cfg);
    let dropped_keys: std::collections::HashSet<[u32; 2]> = dropped.iter().map(key).collect();
    for c in kept.iter().filter(|c| !dropped_keys.contains(&key(c))) {
        let twin = com
            .iter()
            .find(|m| key(m) == key(c))
            .expect("a center-of-mass twin");
        assert_eq!(c.position_covariance, twin.position_covariance);
        assert!(c.position_covariance.is_some());
    }
}
