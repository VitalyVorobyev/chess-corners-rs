//! Application-level helpers.
//!
//! These functions wire up I/O (load image, JSON/PNG output) around the
//! detector APIs so both the CLI and examples can share the same behavior.

use anyhow::{Context, Result};
use chess_corners::{
    AxisEstimate, CenterOfMassConfig, ChessConfig, ChessRefiner, ChessRing, CornerDescriptor,
    DescriptorRing, DetectionStrategy, Detector, DetectorConfig, ForstnerConfig, MultiscaleConfig,
    RadonPeakConfig, RadonRefiner, SaddlePointConfig, Threshold, UpscaleConfig,
};
use image::{ImageBuffer, ImageReader, Luma};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path, path::PathBuf};

/// Refiner selector for the ChESS strategy, exposed by the CLI.
///
/// Accepts only the variants valid for ChESS; passing a Radon-only
/// refiner via `--chess-refiner` is rejected at clap parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChessRefinerSel {
    CenterOfMass,
    Forstner,
    SaddlePoint,
}

/// Refiner selector for the Radon strategy, exposed by the CLI.
///
/// Accepts only the variants valid for Radon; passing a ChESS-only
/// refiner via `--radon-refiner` is rejected at clap parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RadonRefinerSel {
    RadonPeak,
    CenterOfMass,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
    pub image: PathBuf,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub log_level: Option<String>,
    /// Enable the ML refiner pipeline (requires the `ml-refiner` feature).
    pub ml: Option<bool>,
    #[serde(flatten, default)]
    pub algorithm: DetectorConfig,
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
    pub threshold: Option<Threshold>,
    pub chess_ring: Option<ChessRing>,
    pub descriptor_ring: Option<DescriptorRing>,
    pub nms_radius: Option<u32>,
    pub min_cluster_size: Option<u32>,
    /// Override the ChESS subpixel refiner. Ignored when the active
    /// strategy is not ChESS.
    pub chess_refiner: Option<ChessRefinerSel>,
    /// Override the Radon subpixel refiner. Ignored when the active
    /// strategy is not Radon.
    pub radon_refiner: Option<RadonRefinerSel>,
    /// Integer upscale factor override. `Some(0)` means no override;
    /// `Some(k)` for k >= 2 sets `upscale = Fixed(k)`.
    pub upscale_factor: Option<u32>,
}

#[derive(Serialize)]
pub struct AxisOut {
    pub angle: f32,
    pub sigma: f32,
}

#[derive(Serialize)]
pub struct CornerOut {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub contrast: f32,
    pub fit_rms: f32,
    pub axes: [AxisOut; 2],
}

impl From<&AxisEstimate> for AxisOut {
    fn from(a: &AxisEstimate) -> Self {
        Self {
            angle: a.angle,
            sigma: a.sigma,
        }
    }
}

#[derive(Serialize)]
pub struct DetectionDump {
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub algorithm: DetectorConfig,
    pub corners: Vec<CornerOut>,
}

pub fn run_detection(cfg: DetectionConfig) -> Result<()> {
    validate_algorithm_config(&cfg.algorithm)?;

    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();

    #[cfg_attr(not(feature = "ml-refiner"), allow(unused_mut))]
    let mut algorithm = cfg.algorithm;
    if cfg.ml.unwrap_or(false) {
        #[cfg(feature = "ml-refiner")]
        {
            info!("ml refiner: enabled");
            if let DetectionStrategy::Chess(ref mut chess) = algorithm.strategy {
                chess.refiner = ChessRefiner::Ml;
            } else {
                anyhow::bail!(
                    "ml refiner is only valid with the ChESS strategy; the active config selects Radon"
                );
            }
        }
        #[cfg(not(feature = "ml-refiner"))]
        {
            anyhow::bail!("ml refiner requires the \"ml-refiner\" feature")
        }
    }
    log_active_refiner(&algorithm);
    let mut detector = Detector::new(algorithm).map_err(|e| anyhow::anyhow!(e))?;
    let corners = detector.detect(&img).map_err(|e| anyhow::anyhow!(e))?;

    let multiscale_active = matches!(cfg.algorithm.multiscale, MultiscaleConfig::Pyramid { .. });
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
        algorithm: cfg.algorithm,
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
            contrast: c.contrast,
            fit_rms: c.fit_rms,
            axes: [AxisOut::from(&c.axes[0]), AxisOut::from(&c.axes[1])],
        }
    }
}

pub fn validate_algorithm_config(cfg: &DetectorConfig) -> Result<()> {
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
    match cfg.threshold {
        Threshold::Absolute(v) if v < 0.0 => {
            anyhow::bail!("threshold.absolute must be >= 0")
        }
        Threshold::Relative(f) if !(0.0..=1.0).contains(&f) => {
            anyhow::bail!("threshold.relative must be in [0, 1]")
        }
        _ => {}
    }
    cfg.upscale
        .validate()
        .map_err(|err| anyhow::anyhow!("invalid upscale config: {err}"))?;
    Ok(())
}

fn log_active_refiner(cfg: &DetectorConfig) {
    match &cfg.strategy {
        DetectionStrategy::Chess(chess) => info!("refiner: {:?}", chess.refiner),
        DetectionStrategy::Radon(radon) => info!("refiner: {:?}", radon.refiner),
        _ => info!("refiner: <unknown strategy>"),
    }
}

fn chess_strategy_mut(cfg: &mut DetectorConfig) -> Option<&mut ChessConfig> {
    match &mut cfg.strategy {
        DetectionStrategy::Chess(c) => Some(c),
        _ => None,
    }
}

/// Apply an NMS-window override to whichever strategy variant is
/// active. `nms_radius` and `min_cluster_size` are advertised by the
/// CLI as generic detection overrides; both ChESS and Radon expose
/// the same field names in their respective strategy payloads.
fn apply_nms_radius(cfg: &mut DetectorConfig, v: u32) {
    match &mut cfg.strategy {
        DetectionStrategy::Chess(chess) => chess.nms_radius = v,
        DetectionStrategy::Radon(radon) => radon.nms_radius = v,
        _ => {}
    }
}

fn apply_min_cluster_size(cfg: &mut DetectorConfig, v: u32) {
    match &mut cfg.strategy {
        DetectionStrategy::Chess(chess) => chess.min_cluster_size = v,
        DetectionStrategy::Radon(radon) => radon.min_cluster_size = v,
        _ => {}
    }
}

fn apply_chess_refiner(cfg: &mut DetectorConfig, sel: ChessRefinerSel) {
    if let DetectionStrategy::Chess(chess) = &mut cfg.strategy {
        chess.refiner = match sel {
            ChessRefinerSel::CenterOfMass => {
                ChessRefiner::CenterOfMass(CenterOfMassConfig::default())
            }
            ChessRefinerSel::Forstner => ChessRefiner::Forstner(ForstnerConfig::default()),
            ChessRefinerSel::SaddlePoint => ChessRefiner::SaddlePoint(SaddlePointConfig::default()),
        };
    }
}

fn apply_radon_refiner(cfg: &mut DetectorConfig, sel: RadonRefinerSel) {
    if let DetectionStrategy::Radon(radon) = &mut cfg.strategy {
        radon.refiner = match sel {
            RadonRefinerSel::RadonPeak => RadonRefiner::RadonPeak(RadonPeakConfig::default()),
            RadonRefinerSel::CenterOfMass => {
                RadonRefiner::CenterOfMass(CenterOfMassConfig::default())
            }
        };
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
        descriptor_ring,
        nms_radius,
        min_cluster_size,
        chess_refiner,
        radon_refiner,
        upscale_factor,
    } = overrides;

    // Multiscale overrides apply to the top-level multiscale field,
    // which is honoured by both ChESS and Radon strategies. Any single
    // multiscale override coerces `multiscale` from `SingleScale` →
    // `Pyramid { defaults }` so subsequent fields land somewhere visible.
    if pyramid_levels.is_some() || pyramid_min_size.is_some() || refinement_radius.is_some() {
        let (mut cur_levels, mut cur_min_size, mut cur_refinement_radius) =
            match cfg.algorithm.multiscale {
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
        cfg.algorithm.multiscale = MultiscaleConfig::Pyramid {
            levels: cur_levels,
            min_size: cur_min_size,
            refinement_radius: cur_refinement_radius,
        };
    }
    if let Some(v) = merge_radius {
        cfg.algorithm.merge_radius = v;
    }
    if let Some(v) = output_json {
        cfg.output_json = Some(v);
    }
    if let Some(v) = output_png {
        cfg.output_png = Some(v);
    }
    if let Some(v) = threshold {
        cfg.algorithm.threshold = v;
    }
    if let Some(v) = chess_ring {
        if let Some(chess) = chess_strategy_mut(&mut cfg.algorithm) {
            chess.ring = v;
        }
    }
    if let Some(v) = descriptor_ring {
        if let Some(chess) = chess_strategy_mut(&mut cfg.algorithm) {
            chess.descriptor_ring = v;
        }
    }
    if let Some(v) = nms_radius {
        apply_nms_radius(&mut cfg.algorithm, v);
    }
    if let Some(v) = min_cluster_size {
        apply_min_cluster_size(&mut cfg.algorithm, v);
    }
    if let Some(v) = chess_refiner {
        apply_chess_refiner(&mut cfg.algorithm, v);
    }
    if let Some(v) = radon_refiner {
        apply_radon_refiner(&mut cfg.algorithm, v);
    }
    // `Some(0)` is the documented sentinel for "no override — keep the value
    // parsed from the JSON config". Any other value is forwarded to
    // `UpscaleConfig::Fixed`; `UpscaleConfig::validate` below rejects factors
    // outside `{2, 3, 4}` with a clear error.
    if let Some(factor) = upscale_factor {
        if factor != 0 {
            cfg.algorithm.upscale = UpscaleConfig::Fixed(factor);
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
    use chess_corners::low_level;
    use chess_corners::RadonConfig;
    use std::path::PathBuf;

    fn empty_overrides() -> DetectionOverrides {
        DetectionOverrides::default()
    }

    fn radon_detection_config() -> DetectionConfig {
        let mut algorithm = DetectorConfig::radon();
        // Sanity: confirm `DetectorConfig::radon()` actually selects the
        // Radon strategy, so the regression below tests the intended
        // dispatch path.
        assert!(matches!(algorithm.strategy, DetectionStrategy::Radon(_)));
        // Pin the NMS / cluster fields away from defaults so the
        // override path is observable.
        if let DetectionStrategy::Radon(ref mut radon) = algorithm.strategy {
            radon.nms_radius = 7;
            radon.min_cluster_size = 3;
        }
        DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            ml: None,
            algorithm,
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
        match cfg.algorithm.strategy {
            DetectionStrategy::Radon(radon) => assert_eq!(radon.nms_radius, 11),
            other => panic!("expected Radon strategy, got {other:?}"),
        }
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
        match cfg.algorithm.strategy {
            DetectionStrategy::Radon(radon) => assert_eq!(radon.min_cluster_size, 5),
            other => panic!("expected Radon strategy, got {other:?}"),
        }
    }

    #[test]
    fn nms_overrides_still_apply_to_chess_strategy() {
        let mut cfg = DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            ml: None,
            algorithm: DetectorConfig::chess(),
        };
        apply_overrides(
            &mut cfg,
            DetectionOverrides {
                nms_radius: Some(9),
                min_cluster_size: Some(4),
                ..empty_overrides()
            },
        );
        match cfg.algorithm.strategy {
            DetectionStrategy::Chess(chess) => {
                assert_eq!(chess.nms_radius, 9);
                assert_eq!(chess.min_cluster_size, 4);
            }
            other => panic!("expected ChESS strategy, got {other:?}"),
        }
    }

    // The Radon detector exposes the same nms/cluster field names as
    // the ChESS strategy. If a future Radon-side rename happens, this
    // test will fail fast — better than a silent regression where the
    // CLI override returns to being a no-op for one strategy variant.
    #[test]
    fn nms_and_cluster_fields_exist_on_both_strategies() {
        let _chess_nms: u32 = low_level::to_chess_params(&DetectorConfig::chess()).nms_radius;
        let _radon_nms: u32 =
            low_level::to_radon_detector_params(&DetectorConfig::radon()).nms_radius;
        let _radon_cluster: u32 = RadonConfig::default().min_cluster_size;
    }

    fn detection_config_with_upscale(upscale: UpscaleConfig) -> DetectionConfig {
        let mut algorithm = DetectorConfig::chess();
        algorithm.upscale = upscale;
        DetectionConfig {
            image: PathBuf::from("ignored.png"),
            output_json: None,
            output_png: None,
            log_level: None,
            ml: None,
            algorithm,
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
        assert_eq!(cfg.algorithm.upscale, UpscaleConfig::Fixed(3));
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
        assert_eq!(cfg.algorithm.upscale, UpscaleConfig::Fixed(2));
    }
}
