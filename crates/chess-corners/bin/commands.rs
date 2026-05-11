//! Application-level helpers.
//!
//! These functions wire up I/O (load image, JSON/PNG output) around the
//! detector APIs so both the CLI and examples can share the same behavior.

use anyhow::{Context, Result};
use chess_corners::{
    AxisEstimate, ChessConfig, ChessRing, ChessStrategy, CornerDescriptor, DescriptorMode,
    DetectionStrategy, Detector, MultiscaleParams, RefinementMethod, Threshold,
};
use image::{ImageBuffer, ImageReader, Luma};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path, path::PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
    pub image: PathBuf,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub log_level: Option<String>,
    /// Enable the ML refiner pipeline (requires the `ml-refiner` feature).
    pub ml: Option<bool>,
    #[serde(flatten, default)]
    pub algorithm: ChessConfig,
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
    pub descriptor_mode: Option<DescriptorMode>,
    pub nms_radius: Option<u32>,
    pub min_cluster_size: Option<u32>,
    pub refiner_kind: Option<RefinementMethod>,
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
    pub algorithm: ChessConfig,
    pub corners: Vec<CornerOut>,
}

pub fn run_detection(cfg: DetectionConfig) -> Result<()> {
    validate_algorithm_config(&cfg.algorithm)?;

    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();
    info!("refiner: {:?}", cfg.algorithm.refiner.kind);

    #[cfg_attr(not(feature = "ml-refiner"), allow(unused_mut))]
    let mut algorithm = cfg.algorithm.clone();
    if cfg.ml.unwrap_or(false) {
        #[cfg(feature = "ml-refiner")]
        {
            info!("ml refiner: enabled");
            algorithm.refiner.kind = RefinementMethod::Ml;
        }
        #[cfg(not(feature = "ml-refiner"))]
        {
            anyhow::bail!("ml refiner requires the \"ml-refiner\" feature")
        }
    }
    let mut detector = Detector::new(algorithm).map_err(|e| anyhow::anyhow!(e))?;
    let corners = detector.detect(&img).map_err(|e| anyhow::anyhow!(e))?;

    let multiscale_active = chess_multiscale(&cfg.algorithm).is_some();
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
        algorithm: cfg.algorithm.clone(),
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

pub fn validate_algorithm_config(cfg: &ChessConfig) -> Result<()> {
    if let Some(ms) = chess_multiscale(cfg) {
        if ms.pyramid_levels == 0 {
            anyhow::bail!("strategy.chess.multiscale.pyramid_levels must be >= 1");
        }
        if ms.pyramid_min_size == 0 {
            anyhow::bail!("strategy.chess.multiscale.pyramid_min_size must be >= 1");
        }
        if ms.refinement_radius == 0 {
            anyhow::bail!("strategy.chess.multiscale.refinement_radius must be >= 1");
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

fn chess_multiscale(cfg: &ChessConfig) -> Option<MultiscaleParams> {
    match &cfg.strategy {
        DetectionStrategy::Chess(c) => c.multiscale,
        _ => None,
    }
}

fn chess_strategy_mut(cfg: &mut ChessConfig) -> Option<&mut ChessStrategy> {
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
        descriptor_mode,
        nms_radius,
        min_cluster_size,
        refiner_kind,
    } = overrides;

    // Multiscale overrides apply only to the ChESS strategy. If the
    // config is currently Radon, multiscale-shaped flags are ignored
    // (Radon is single-scale today). For ChESS, any single multiscale
    // override coerces `multiscale` from `None` → `Some(default)` so
    // subsequent fields land somewhere visible.
    if pyramid_levels.is_some() || pyramid_min_size.is_some() || refinement_radius.is_some() {
        if let Some(chess) = chess_strategy_mut(&mut cfg.algorithm) {
            let ms = chess
                .multiscale
                .get_or_insert_with(MultiscaleParams::default);
            if let Some(v) = pyramid_levels {
                ms.pyramid_levels = v;
            }
            if let Some(v) = pyramid_min_size {
                ms.pyramid_min_size = v as usize;
            }
            if let Some(v) = refinement_radius {
                ms.refinement_radius = v;
            }
        }
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
    if let Some(v) = descriptor_mode {
        cfg.algorithm.descriptor_mode = v;
    }
    if let Some(v) = nms_radius {
        if let Some(chess) = chess_strategy_mut(&mut cfg.algorithm) {
            chess.nms_radius = v;
        }
    }
    if let Some(v) = min_cluster_size {
        if let Some(chess) = chess_strategy_mut(&mut cfg.algorithm) {
            chess.min_cluster_size = v;
        }
    }
    if let Some(v) = refiner_kind {
        cfg.algorithm.refiner.kind = v;
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
