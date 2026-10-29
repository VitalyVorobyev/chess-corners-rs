//! Application-level helpers.
//!
//! These functions wire up I/O (load image, JSON/PNG output) around the
//! detector APIs so both the CLI and examples can share the same behavior.

use anyhow::{Context, Result};
use chess_corners::{
    AxisEstimate, ChessConfig, ChessRing, CornerDescriptor, DetectionStrategy, Detector,
    DetectorConfig, ImageRefiner, MultiscaleConfig, UpscaleConfig,
};
use image::{ImageBuffer, ImageReader, Luma};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path, path::PathBuf};

/// Image-refiner selector exposed by the CLI (`--refiner`), with default
/// tuning. `none` keeps the detector's own subpixel localizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinerSel {
    None,
    SaddlePoint,
}

impl RefinerSel {
    fn image_refiner(self) -> Option<ImageRefiner> {
        match self {
            Self::None => None,
            Self::SaddlePoint => Some(ImageRefiner::saddle_point()),
        }
    }
}

/// The CLI's JSON config file: the run's I/O settings plus the detector
/// configuration nested under `detector`. Unknown keys are rejected at
/// every level.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectionConfig {
    pub image: PathBuf,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub log_level: Option<String>,
    /// Omitted: the library default (`DetectorConfig::default()`).
    #[serde(default)]
    pub detector: DetectorConfig,
}

/// CLI overrides applied on top of the JSON config. Each `Option`
/// represents a flag the user passed on the command line; `None` means
/// "leave the JSON-loaded value alone".
#[derive(Debug, Default)]
pub struct DetectionOverrides {
    pub pyramid_levels: Option<u8>,
    pub pyramid_min_size: Option<u32>,
    pub refinement_radius: Option<u32>,
    pub merge_radius: Option<f32>,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub threshold: Option<f32>,
    pub chess_ring: Option<ChessRing>,
    pub nms_radius: Option<u32>,
    pub min_cluster_size: Option<u32>,
    /// Override the image refiner (either strategy).
    pub refiner: Option<RefinerSel>,
    /// Integer upscale factor override. `Some(0)` means no override;
    /// `Some(k)` for k >= 2 sets `upscale = Fixed(k)`.
    pub upscale_factor: Option<u32>,
}

#[derive(Serialize)]
pub struct AxisOut {
    pub angle: f32,
    pub fit_std_error: f32,
}

#[derive(Serialize)]
pub struct CovarianceOut {
    pub xx: f32,
    pub xy: f32,
    pub yy: f32,
}

#[derive(Serialize)]
pub struct CornerOut {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    /// `null` when the orientation fit was skipped.
    pub axes: Option<[AxisOut; 2]>,
    /// Position covariance in px²; `null` when the localizer reports none.
    pub position_covariance: Option<CovarianceOut>,
}

impl From<&AxisEstimate> for AxisOut {
    fn from(a: &AxisEstimate) -> Self {
        Self {
            angle: a.angle,
            fit_std_error: a.fit_std_error,
        }
    }
}

#[derive(Serialize)]
pub struct DetectionDump {
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub detector: DetectorConfig,
    /// Image noise σ (gray levels) the position covariances were computed
    /// with; `null` when no corner carries one.
    pub noise_sigma: Option<f32>,
    pub corners: Vec<CornerOut>,
}

pub fn run_detection(cfg: DetectionConfig) -> Result<()> {
    validate_detector_config(&cfg.detector)?;

    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();

    log_active_refiner(&cfg.detector);
    let mut detector = Detector::new(cfg.detector).map_err(|e| anyhow::anyhow!(e))?;
    let corners = detector.detect(&img).map_err(|e| anyhow::anyhow!(e))?;

    let multiscale_active = matches!(cfg.detector.multiscale, MultiscaleConfig::Pyramid { .. });
    let json_out = cfg.output_json.clone().unwrap_or_else(|| {
        if multiscale_active {
            cfg.image.with_extension("multiscale.corners.json")
        } else {
            cfg.image.with_extension("corners.json")
        }
    });
    let dump = DetectionDump {
        image: cfg.image.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        detector: cfg.detector,
        noise_sigma: detector.last_noise_sigma(),
        corners: corners.iter().map(CornerOut::from).collect(),
    };
    write_json(&json_out, &dump)?;

    let png_out = cfg.output_png.clone().unwrap_or_else(|| {
        if multiscale_active {
            cfg.image.with_extension("multiscale.corners.png")
        } else {
            cfg.image.with_extension("corners.png")
        }
    });
    let mut vis: ImageBuffer<Luma<u8>, _> = img.clone();
    draw_corners(&mut vis, dump.corners.iter().map(|c| (c.x, c.y)))?;
    vis.save(&png_out)?;

    Ok(())
}

impl From<&CornerDescriptor> for CornerOut {
    fn from(c: &CornerDescriptor) -> Self {
        Self {
            x: c.x,
            y: c.y,
            response: c.response,
            axes: c
                .axes
                .map(|axes| [AxisOut::from(&axes[0]), AxisOut::from(&axes[1])]),
            position_covariance: c.position_covariance.map(|p| CovarianceOut {
                xx: p.xx,
                xy: p.xy,
                yy: p.yy,
            }),
        }
    }
}

pub fn validate_detector_config(cfg: &DetectorConfig) -> Result<()> {
    if let MultiscaleConfig::Pyramid {
        levels,
        min_size,
        refinement_radius,
    } = cfg.multiscale
    {
        if levels == 0 {
            anyhow::bail!("multiscale.pyramid.levels must be >= 1");
        }
        if min_size == 0 {
            anyhow::bail!("multiscale.pyramid.min_size must be >= 1");
        }
        if refinement_radius == 0 {
            anyhow::bail!("multiscale.pyramid.refinement_radius must be >= 1");
        }
    }
    if cfg.merge_radius <= 0.0 {
        anyhow::bail!("merge_radius must be > 0");
    }
    if cfg.threshold < 0.0 {
        anyhow::bail!("threshold must be >= 0");
    }
    // Radon reads the threshold as a fraction of the per-frame maximum.
    if matches!(cfg.strategy, DetectionStrategy::Radon(_)) && cfg.threshold > 1.0 {
        anyhow::bail!("threshold for the Radon detector is a fraction in [0, 1]");
    }
    cfg.upscale
        .validate()
        .map_err(|err| anyhow::anyhow!("invalid upscale config: {err}"))?;
    Ok(())
}

fn log_active_refiner(cfg: &DetectorConfig) {
    let localizer = match &cfg.strategy {
        DetectionStrategy::Chess(chess) => format!("{:?}", chess.center_of_mass),
        DetectionStrategy::Radon(radon) => format!("{:?} peak fit", radon.peak_fit),
        _ => "<unknown strategy>".to_string(),
    };
    info!("localizer: {localizer}; image refiner: {:?}", cfg.refiner);
}

fn chess_strategy_mut(cfg: &mut DetectorConfig) -> Option<&mut ChessConfig> {
    match &mut cfg.strategy {
        DetectionStrategy::Chess(c) => Some(c),
        _ => None,
    }
}

pub fn apply_overrides(cfg: &mut DetectionConfig, overrides: DetectionOverrides) {
    let DetectionOverrides {
        pyramid_levels,
        pyramid_min_size,
        refinement_radius,
        merge_radius,
        output_json,
        output_png,
        threshold,
        chess_ring,
        nms_radius,
        min_cluster_size,
        refiner,
        upscale_factor,
    } = overrides;

    // Multiscale overrides apply to the top-level multiscale field,
    // which is honoured by both ChESS and Radon strategies. Any single
    // multiscale override coerces `multiscale` from `SingleScale` →
    // `Pyramid { defaults }` so subsequent fields land somewhere visible.
    if pyramid_levels.is_some() || pyramid_min_size.is_some() || refinement_radius.is_some() {
        let (mut cur_levels, mut cur_min_size, mut cur_refinement_radius) =
            match cfg.detector.multiscale {
                MultiscaleConfig::Pyramid {
                    levels,
                    min_size,
                    refinement_radius,
                } => (levels, min_size, refinement_radius),
                MultiscaleConfig::SingleScale => (3, 128usize, 3),
                _ => (3, 128usize, 3),
            };
        if let Some(v) = pyramid_levels {
            cur_levels = v;
        }
        if let Some(v) = pyramid_min_size {
            cur_min_size = v as usize;
        }
        if let Some(v) = refinement_radius {
            cur_refinement_radius = v;
        }
        cfg.detector.multiscale = MultiscaleConfig::Pyramid {
            levels: cur_levels,
            min_size: cur_min_size,
            refinement_radius: cur_refinement_radius,
        };
    }
    if let Some(v) = merge_radius {
        cfg.detector.merge_radius = v;
    }
    if let Some(v) = output_json {
        cfg.output_json = Some(v);
    }
    if let Some(v) = output_png {
        cfg.output_png = Some(v);
    }
    if let Some(v) = threshold {
        cfg.detector.threshold = v;
    }
    if let Some(v) = chess_ring {
        if let Some(chess) = chess_strategy_mut(&mut cfg.detector) {
            chess.ring = v;
        }
    }
    // NMS / clustering are shared detection knobs honoured by both
    // strategies; they live on `DetectorConfig::detection`.
    if let Some(v) = nms_radius {
        cfg.detector.detection.nms_radius = v;
    }
    if let Some(v) = min_cluster_size {
        cfg.detector.detection.min_cluster_size = v;
    }
    if let Some(v) = refiner {
        cfg.detector.refiner = v.image_refiner();
    }
    // `Some(0)` is the documented sentinel for "no override — keep the value
    // parsed from the JSON config". Any other value is forwarded to
    // `UpscaleConfig::Fixed`; `UpscaleConfig::validate` below rejects factors
    // outside `{2, 3, 4}` with a clear error.
    if let Some(factor) = upscale_factor {
        if factor != 0 {
            cfg.detector.upscale = UpscaleConfig::Fixed(factor);
        }
    }
}

fn draw_corners(
    vis: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    corners: impl Iterator<Item = (f32, f32)>,
) -> Result<()> {
    for (x_f, y_f) in corners {
        let x = x_f.round() as i32;
        let y = y_f.round() as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let xx = x + dx;
                let yy = y + dy;
                if xx >= 0 && yy >= 0 && xx < vis.width() as i32 && yy < vis.height() as i32 {
                    vis.put_pixel(xx as u32, yy as u32, Luma([255u8]));
                }
            }
        }
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut json_file = File::create(path)?;
    serde_json::to_writer_pretty(&mut json_file, value)?;
    json_file.write_all(b"\n")?;
    Ok(())
}

pub fn load_config(path: &Path) -> Result<DetectionConfig> {
    let file = File::open(path).with_context(|| format!("opening config {}", path.display()))?;
    let cfg: DetectionConfig = serde_json::from_reader(file)
        .with_context(|| format!("parsing config {}", path.display()))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn empty_overrides() -> DetectionOverrides {
        DetectionOverrides::default()
    }

    fn radon_detection_config() -> DetectionConfig {
        let mut detector = DetectorConfig::radon();
        // Sanity: confirm `DetectorConfig::radon()` actually selects the
        // Radon strategy, so the regression below tests the intended
        // dispatch path.
        assert!(matches!(detector.strategy, DetectionStrategy::Radon(_)));
        // Pin the shared NMS / cluster fields away from defaults so the
        // override path is observable.
        detector.detection.nms_radius = 7;
        detector.detection.min_cluster_size = 3;
        DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            detector,
        }
    }

    #[test]
    fn nms_radius_override_applies_to_active_radon_strategy() {
        let mut cfg = radon_detection_config();
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                nms_radius: Some(11),
                ..empty_overrides()
            },
        );
        assert!(matches!(cfg.detector.strategy, DetectionStrategy::Radon(_)));
        assert_eq!(cfg.detector.detection.nms_radius, 11);
    }

    #[test]
    fn min_cluster_size_override_applies_to_active_radon_strategy() {
        let mut cfg = radon_detection_config();
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                min_cluster_size: Some(5),
                ..empty_overrides()
            },
        );
        assert!(matches!(cfg.detector.strategy, DetectionStrategy::Radon(_)));
        assert_eq!(cfg.detector.detection.min_cluster_size, 5);
    }

    #[test]
    fn nms_overrides_still_apply_to_chess_strategy() {
        let mut cfg = DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            detector: DetectorConfig::chess(),
        };
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                nms_radius: Some(9),
                min_cluster_size: Some(4),
                ..empty_overrides()
            },
        );
        assert!(matches!(cfg.detector.strategy, DetectionStrategy::Chess(_)));
        assert_eq!(cfg.detector.detection.nms_radius, 9);
        assert_eq!(cfg.detector.detection.min_cluster_size, 4);
    }

    // The shared detection knobs must reach both detectors' low-level
    // params. If a future change drops the threading for one strategy,
    // this fails fast — better than a silent CLI-override no-op.
    #[test]
    fn shared_detection_params_reach_both_strategies() {
        let cfg = DetectorConfig::chess().with_detection(|d| {
            d.nms_radius = 5;
            d.min_cluster_size = 3;
        });
        let chess = cfg.chess_params();
        assert_eq!(chess.nms_radius, 5);
        assert_eq!(chess.min_cluster_size, 3);

        let cfg = DetectorConfig::radon().with_detection(|d| {
            d.nms_radius = 6;
            d.min_cluster_size = 2;
        });
        let radon = cfg.radon_detector_params();
        assert_eq!(radon.nms_radius, 6);
        assert_eq!(radon.min_cluster_size, 2);
    }

    fn detection_config_with_upscale(upscale: UpscaleConfig) -> DetectionConfig {
        let mut detector = DetectorConfig::chess();
        detector.upscale = upscale;
        DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            detector,
        }
    }

    #[test]
    fn upscale_factor_zero_preserves_json_config_value() {
        let mut cfg = detection_config_with_upscale(UpscaleConfig::Fixed(3));
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                upscale_factor: Some(0),
                ..empty_overrides()
            },
        );
        assert_eq!(cfg.detector.upscale, UpscaleConfig::Fixed(3));
    }

    #[test]
    fn upscale_factor_nonzero_overrides_json_config_value() {
        let mut cfg = detection_config_with_upscale(UpscaleConfig::Disabled);
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                upscale_factor: Some(2),
                ..empty_overrides()
            },
        );
        assert_eq!(cfg.detector.upscale, UpscaleConfig::Fixed(2));
    }

    fn workspace_file(rel: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel)
    }

    #[test]
    fn example_config_loads_with_its_detector_block() {
        let cfg = load_config(&workspace_file("config/chess_cli_config_example.json")).unwrap();
        assert_eq!(cfg.image, PathBuf::from("testimages/mid.png"));
        assert_eq!(cfg.log_level.as_deref(), Some("info"));
        let algorithm_only: DetectorConfig = serde_json::from_reader(
            File::open(workspace_file("config/chess_algorithm_config_example.json")).unwrap(),
        )
        .unwrap();
        assert_ne!(algorithm_only, DetectorConfig::default());
    }

    #[test]
    fn detector_block_is_optional() {
        let cfg: DetectionConfig = serde_json::from_str(r#"{"image": "a.png"}"#).unwrap();
        assert_eq!(cfg.detector, DetectorConfig::default());
        assert_eq!(cfg.output_json, None);
    }

    #[test]
    fn unknown_keys_are_rejected_and_named_at_every_level() {
        for (json, key) in [
            // A 1.x flat file: detector keys beside the CLI keys.
            (r#"{"image": "a.png", "threshold": 30.0}"#, "threshold"),
            (
                r#"{"image": "a.png", "output_jsn": "o.json"}"#,
                "output_jsn",
            ),
            (
                r#"{"image": "a.png", "detector": {"noise_sigma": 1.5}}"#,
                "noise_sigma",
            ),
            (
                r#"{"image": "a.png", "detector": {"strategy": {"chess": {"refiner": "center_of_mass"}}}}"#,
                "refiner",
            ),
        ] {
            let err = serde_json::from_str::<DetectionConfig>(json)
                .expect_err(json)
                .to_string();
            assert!(
                err.contains(&format!("unknown field `{key}`")),
                "{json}: {err}"
            );
        }
    }
}
