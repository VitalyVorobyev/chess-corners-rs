//! Application-level helpers.
//!
//! These functions wire up I/O (load image, optional downsampling, JSON/PNG
//! output) around the `chess` detection APIs so both the CLI and examples can
//! share the same behavior.

use anyhow::{Context, Result};
use chess::image::find_corners_image;
use chess::multiscale::find_corners_coarse_to_fine_image;
use chess::{ChessParams, CoarseToFineParams, PyramidBuffers};
use image::{
    imageops::{resize, FilterType},
    ImageBuffer, ImageReader, Luma,
};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path, path::PathBuf, str::FromStr};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DetectionMode {
    Single,
    Multiscale,
}

impl FromStr for DetectionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "single" => Ok(DetectionMode::Single),
            "multiscale" | "multi" => Ok(DetectionMode::Multiscale),
            other => Err(format!(
                "invalid mode '{other}', expected single|multiscale"
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
    pub image: PathBuf,
    pub mode: Option<DetectionMode>,
    pub downsample: Option<u32>,
    pub pyramid_levels: Option<u8>,
    pub min_size: Option<u32>,
    pub roi_radius: Option<u32>,
    pub merge_radius: Option<f32>,
    pub output_json: Option<PathBuf>,
    pub output_png: Option<PathBuf>,
    pub threshold_rel: Option<f32>,
    pub threshold_abs: Option<f32>,
    pub radius: Option<u32>,
    pub descriptor_radius: Option<u32>,
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
    pub mode: String,
    pub downsample: Option<u32>,
    pub pyramid_levels: Option<u8>,
    pub min_size: Option<u32>,
    pub roi_radius: Option<u32>,
    pub merge_radius: Option<f32>,
    pub corners: Vec<CornerOut>,
}

pub fn run_detection(cfg: DetectionConfig) -> Result<()> {
    let mode = cfg.mode.unwrap_or(DetectionMode::Single);
    match mode {
        DetectionMode::Single => run_single(cfg),
        DetectionMode::Multiscale => run_multiscale(cfg),
    }
}

fn run_single(cfg: DetectionConfig) -> Result<()> {
    let downsample = cfg.downsample.unwrap_or(1);
    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();
    let work_img = if downsample > 1 {
        let w = img.width().div_ceil(downsample);
        let h = img.height().div_ceil(downsample);
        resize(&img, w, h, FilterType::Triangle)
    } else {
        img.clone()
    };

    let mut params = ChessParams::default();
    apply_params_overrides(&mut params, &cfg);

    let mut corners = find_corners_image(&work_img, &params);

    if downsample > 1 {
        let s = downsample as f32;
        for c in &mut corners {
            c.x *= s;
            c.y *= s;
        }
    }

    let json_out = cfg
        .output_json
        .unwrap_or_else(|| cfg.image.with_extension("corners.json"));
    let dump = DetectionDump {
        image: cfg.image.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        mode: "single".to_string(),
        downsample: Some(downsample),
        pyramid_levels: None,
        min_size: None,
        roi_radius: None,
        merge_radius: None,
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

    let png_out = cfg
        .output_png
        .unwrap_or_else(|| cfg.image.with_extension("corners.png"));
    let mut vis: ImageBuffer<Luma<u8>, _> = img.clone();
    draw_corners(&mut vis, dump.corners.iter().map(|c| (c.x, c.y)))?;
    vis.save(&png_out)?;

    Ok(())
}

fn run_multiscale(cfg: DetectionConfig) -> Result<()> {
    let mut cf = CoarseToFineParams::default();
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

    let mut params = ChessParams::default();
    apply_params_overrides(&mut params, &cfg);

    let img = ImageReader::open(&cfg.image)?.decode()?.to_luma8();
    let mut buffers = PyramidBuffers::with_capacity(cf.pyramid.num_levels);
    buffers.prepare_for_image(&img, &cf.pyramid);

    let res = find_corners_coarse_to_fine_image(&img, &params, &cf, &mut buffers);

    let json_out = cfg
        .output_json
        .unwrap_or_else(|| cfg.image.with_extension("multiscale.corners.json"));
    let dump = DetectionDump {
        image: cfg.image.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        mode: "multiscale".to_string(),
        downsample: None,
        pyramid_levels: Some(cf.pyramid.num_levels),
        min_size: Some(cf.pyramid.min_size),
        roi_radius: Some(cf.roi_radius),
        merge_radius: Some(cf.merge_radius),
        corners: res
            .corners
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

    let png_out = cfg
        .output_png
        .unwrap_or_else(|| cfg.image.with_extension("multiscale.corners.png"));
    let mut vis: ImageBuffer<Luma<u8>, _> = img.clone();
    draw_corners(&mut vis, dump.corners.iter().map(|c| (c.x, c.y)))?;
    vis.save(&png_out)?;

    Ok(())
}

fn apply_params_overrides(params: &mut ChessParams, cfg: &DetectionConfig) {
    if let Some(r) = cfg.radius {
        params.radius = r;
    }
    if let Some(r) = cfg.descriptor_radius {
        params.descriptor_radius = Some(r);
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
