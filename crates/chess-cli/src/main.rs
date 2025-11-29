use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use chess_cli::commands::{load_config, run_detection, DetectionMode};
use std::str::FromStr;

#[derive(Parser)]
#[command(author, version, about = "ChESS detector CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run detection from a config JSON (single or multiscale).
    Run {
        /// Path to config JSON.
        config: PathBuf,
        /// Override mode (single or multiscale).
        #[arg(long, value_parser = DetectionMode::from_str)]
        mode: Option<DetectionMode>,
        /// Override downsample factor.
        #[arg(long)]
        downsample: Option<u32>,
        /// Override pyramid levels.
        #[arg(long)]
        levels: Option<u8>,
        /// Override pyramid min size.
        #[arg(long)]
        min_size: Option<u32>,
        /// Override ROI radius (coarse pixels).
        #[arg(long)]
        roi_radius: Option<u32>,
        /// Override merge radius.
        #[arg(long)]
        merge_radius: Option<f32>,
        /// Output JSON path override.
        #[arg(long)]
        output_json: Option<PathBuf>,
        /// Output overlay PNG path override.
        #[arg(long)]
        output_png: Option<PathBuf>,
        /// Relative threshold override.
        #[arg(long)]
        threshold_rel: Option<f32>,
        /// Absolute threshold override.
        #[arg(long)]
        threshold_abs: Option<f32>,
        /// Ring radius override.
        #[arg(long)]
        radius: Option<u32>,
        /// NMS radius override.
        #[arg(long)]
        nms_radius: Option<u32>,
        /// Min cluster size override.
        #[arg(long)]
        min_cluster_size: Option<u32>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            mode,
            downsample,
            levels,
            min_size,
            roi_radius,
            merge_radius,
            output_json,
            output_png,
            threshold_rel,
            threshold_abs,
            radius,
            nms_radius,
            min_cluster_size,
        } => {
            let mut cfg = load_config(&config)?;
            if let Some(m) = mode {
                cfg.mode = Some(m);
            }
            if let Some(v) = downsample {
                cfg.downsample = Some(v);
            }
            if let Some(v) = levels {
                cfg.pyramid_levels = Some(v);
            }
            if let Some(v) = min_size {
                cfg.min_size = Some(v);
            }
            if let Some(v) = roi_radius {
                cfg.roi_radius = Some(v);
            }
            if let Some(v) = merge_radius {
                cfg.merge_radius = Some(v);
            }
            if let Some(v) = output_json {
                cfg.output_json = Some(v);
            }
            if let Some(v) = output_png {
                cfg.output_png = Some(v);
            }
            if let Some(v) = threshold_rel {
                cfg.threshold_rel = Some(v);
            }
            if let Some(v) = threshold_abs {
                cfg.threshold_abs = Some(v);
            }
            if let Some(v) = radius {
                cfg.radius = Some(v);
            }
            if let Some(v) = nms_radius {
                cfg.nms_radius = Some(v);
            }
            if let Some(v) = min_cluster_size {
                cfg.min_cluster_size = Some(v);
            }
            run_detection(cfg)
        }
    }
}
