//! Application-level helpers.
//!
//! These functions wire up I/O (load image, JSON/PNG output) around the
//! detector APIs so both the CLI and examples can share the same behavior.

use anyhow::{Context, Result};
#[cfg(feature = "ml-refiner")]
use chess_corners::find_chess_corners_image_with_ml;
use chess_corners::{
    find_chess_corners_image, AxisEstimate, ChessConfig, CornerDescriptor, DescriptorMode,
    DetectorMode, RefinementMethod, ThresholdMode,
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

#[derive(Debug, Default)]
pub struct DetectionOverrides {
    pub pyramid_levels: Option<u8>,
    pub pyramid_min_size: Option<u32>,
    pub refinement_radius: Option<u32>,
    pub merge_radius: Option<f32>,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub threshold_mode: Option<ThresholdMode>,
    pub threshold_value: Option<f32>,
    pub detector_mode: Option<DetectorMode>,
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

    let use_ml = cfg.ml.unwrap_or(false);
    let corners = if use_ml {
        #[cfg(feature = "ml-refiner")]
        {
            info!("ml refiner: enabled");
            find_chess_corners_image_with_ml(&img, &cfg.algorithm)
        }
        #[cfg(not(feature = "ml-refiner"))]
        {
            anyhow::bail!("ml refiner requires the \"ml-refiner\" feature")
        }
    } else {
        find_chess_corners_image(&img, &cfg.algorithm)
    };

    let json_out = cfg.output_json.clone().unwrap_or_else(|| {
        if cfg.algorithm.pyramid_levels <= 1 {
            cfg.image.with_extension("corners.json")
        } else {
            cfg.image.with_extension("multiscale.corners.json")
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
        if cfg.algorithm.pyramid_levels <= 1 {
            cfg.image.with_extension("corners.png")
        } else {
            cfg.image.with_extension("multiscale.corners.png")
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
    if cfg.pyramid_levels == 0 {
        anyhow::bail!("pyramid_levels must be >= 1");
    }
    if cfg.pyramid_min_size == 0 {
        anyhow::bail!("pyramid_min_size must be >= 1");
    }
    if cfg.refinement_radius == 0 {
        anyhow::bail!("refinement_radius must be >= 1");
    }
    if cfg.merge_radius <= 0.0 {
        anyhow::bail!("merge_radius must be > 0");
    }
    if cfg.threshold_value < 0.0 {
        anyhow::bail!("threshold_value must be >= 0");
    }
    cfg.upscale
        .validate()
        .map_err(|err| anyhow::anyhow!("invalid upscale config: {err}"))?;
    Ok(())
}

pub fn apply_overrides(cfg: &mut DetectionConfig, overrides: DetectionOverrides) {
    let DetectionOverrides {
        pyramid_levels,
        pyramid_min_size,
        refinement_radius,
        merge_radius,
        output_json,
        output_png,
        threshold_mode,
        threshold_value,
        detector_mode,
        descriptor_mode,
        nms_radius,
        min_cluster_size,
        refiner_kind,
    } = overrides;

    if let Some(v) = pyramid_levels {
        cfg.algorithm.pyramid_levels = v;
    }
    if let Some(v) = pyramid_min_size {
        cfg.algorithm.pyramid_min_size = v as usize;
    }
    if let Some(v) = refinement_radius {
        cfg.algorithm.refinement_radius = v;
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
    if let Some(v) = threshold_mode {
        cfg.algorithm.threshold_mode = v;
    }
    if let Some(v) = threshold_value {
        cfg.algorithm.threshold_value = v;
    }
    if let Some(v) = detector_mode {
        cfg.algorithm.detector_mode = v;
    }
    if let Some(v) = descriptor_mode {
        cfg.algorithm.descriptor_mode = v;
    }
    if let Some(v) = nms_radius {
        cfg.algorithm.nms_radius = v;
    }
    if let Some(v) = min_cluster_size {
        cfg.algorithm.min_cluster_size = v;
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
