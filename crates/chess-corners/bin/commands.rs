//! Application-level helpers.
//!
//! These functions wire up I/O (load image, JSON/PNG
//! output) around the `chess` detection APIs so both the CLI and examples can
//! share the same behavior.

use anyhow::{Context, Result};
use chess_corners::{find_chess_corners_image, ChessConfig, ChessParams, CoarseToFineParams};
use image::{ImageBuffer, ImageReader, Luma};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path, path::PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
    pub image: PathBuf,
    pub pyramid_levels: Option<u8>,
    pub min_size: Option<u32>,
    pub roi_radius: Option<u32>,
    pub merge_radius: Option<f32>,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub threshold_rel: Option<f32>,
    pub threshold_abs: Option<f32>,
    pub radius10: Option<bool>,
    pub descriptor_radius10: Option<bool>,
    pub nms_radius: Option<u32>,
    pub min_cluster_size: Option<u32>,
    pub log_level: Option<String>,
}

#[derive(Serialize)]
pub struct CornerOut {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub orientation: f32,
    pub phase: u8,
    pub anisotropy: f32,
}

#[derive(Serialize)]
pub struct DetectionDump {
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub pyramid_levels: u8,
    pub min_size: u32,
    pub roi_radius: u32,
    pub merge_radius: f32,
    pub corners: Vec<CornerOut>,
}

pub fn run_detection(cfg: DetectionConfig) -> Result<()> {
    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();

    // Configure detector: `pyramid_levels` and `min_size` together
    // determine whether the run is effectively single-scale
    // (levels <= 1) or multiscale (levels > 1 with a valid pyramid).
    let mut config = ChessConfig::default();
    apply_params_overrides(&mut config.params, &cfg);
    apply_multiscale_overrides(&mut config.multiscale, &cfg)?;

    let corners = find_chess_corners_image(&img, &config);

    let levels = config.multiscale.pyramid.num_levels;
    let min_size = config.multiscale.pyramid.min_size;
    let roi_radius = config.multiscale.roi_radius;
    let merge_radius = config.multiscale.merge_radius;

    let json_out = cfg.output_json.unwrap_or_else(|| {
        if levels <= 1 {
            cfg.image.with_extension("corners.json")
        } else {
            cfg.image.with_extension("multiscale.corners.json")
        }
    });
    let dump = DetectionDump {
        image: cfg.image.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        pyramid_levels: levels,
        min_size,
        roi_radius,
        merge_radius,
        corners: corners
            .iter()
            .map(|c| CornerOut {
                x: c.x,
                y: c.y,
                response: c.response,
                orientation: c.orientation,
                phase: c.phase,
                anisotropy: c.anisotropy,
            })
            .collect(),
    };
    write_json(&json_out, &dump)?;

    let png_out = cfg.output_png.unwrap_or_else(|| {
        if levels <= 1 {
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

fn apply_params_overrides(params: &mut ChessParams, cfg: &DetectionConfig) {
    if let Some(r) = cfg.radius10 {
        params.use_radius10 = r;
    }
    if let Some(r) = cfg.descriptor_radius10 {
        params.descriptor_use_radius10 = Some(r);
    }
    if let Some(t) = cfg.threshold_rel {
        params.threshold_rel = t;
    }
    if let Some(t) = cfg.threshold_abs {
        params.threshold_abs = Some(t);
    }
    if let Some(n) = cfg.nms_radius {
        params.nms_radius = n;
    }
    if let Some(m) = cfg.min_cluster_size {
        params.min_cluster_size = m;
    }
}

fn apply_multiscale_overrides(cf: &mut CoarseToFineParams, cfg: &DetectionConfig) -> Result<()> {
    if let Some(v) = cfg.pyramid_levels {
        if v == 0 {
            anyhow::bail!("levels must be >= 1");
        }
        cf.pyramid.num_levels = v;
    }

    if let Some(v) = cfg.min_size {
        if v == 0 {
            anyhow::bail!("min-size must be >= 1");
        }
        cf.pyramid.min_size = v;
    }
    if let Some(v) = cfg.roi_radius {
        if v == 0 {
            anyhow::bail!("roi radius must be >= 1");
        }
        cf.roi_radius = v;
    }
    if let Some(v) = cfg.merge_radius {
        if v <= 0.0 {
            anyhow::bail!("merge radius must be > 0");
        }
        cf.merge_radius = v;
    }

    Ok(())
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
