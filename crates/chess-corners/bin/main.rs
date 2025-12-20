use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;

#[cfg(not(feature = "tracing"))]
mod logger;
#[cfg(not(feature = "tracing"))]
use log::LevelFilter;
#[cfg(not(feature = "tracing"))]
use std::str::FromStr;

use commands::{load_config, run_detection};

#[cfg(feature = "tracing")]
use tracing_subscriber::fmt::format::FmtSpan;
#[cfg(feature = "tracing")]
use tracing_subscriber::util::SubscriberInitExt;
#[cfg(feature = "tracing")]
use tracing_subscriber::{fmt, EnvFilter};

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
        /// Override pyramid levels (1 => single scale, >=2 => multiscale).
        #[arg(long)]
        levels: Option<u8>,
        /// Override pyramid min size.
        #[arg(long)]
        min_size: Option<u32>,
        /// Override refinement radius (coarse pixels).
        #[arg(long)]
        refinement_radius: Option<u32>,
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
        /// Use large (r=10) ring instead of default r=5.
        #[arg(long)]
        radius10: bool,
        /// Use large (r=10) ring for descriptors instead of default.
        #[arg(long)]
        descriptor_radius10: bool,
        /// NMS radius override.
        #[arg(long)]
        nms_radius: Option<u32>,
        /// Min cluster size override.
        #[arg(long)]
        min_cluster_size: Option<u32>,
        /// Emit tracing in JSON format.
        #[cfg(feature = "tracing")]
        #[arg(long)]
        json_trace: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            levels,
            min_size,
            refinement_radius,
            merge_radius,
            output_json,
            output_png,
            threshold_rel,
            threshold_abs,
            radius10,
            descriptor_radius10,
            nms_radius,
            min_cluster_size,
            #[cfg(feature = "tracing")]
            json_trace,
        } => {
            #[cfg(feature = "tracing")]
            init_tracing(json_trace);
            let mut cfg = load_config(&config)?;
            if let Some(v) = levels {
                cfg.pyramid_levels = Some(v);
            }
            if let Some(v) = min_size {
                cfg.min_size = Some(v);
            }
            if let Some(v) = refinement_radius {
                cfg.refinement_radius = Some(v);
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
            if radius10 {
                cfg.radius10 = Some(true);
            }
            if descriptor_radius10 {
                cfg.descriptor_radius10 = Some(true);
            }
            if let Some(v) = nms_radius {
                cfg.nms_radius = Some(v);
            }
            if let Some(v) = min_cluster_size {
                cfg.min_cluster_size = Some(v);
            }

            #[cfg(not(feature = "tracing"))]
            {
                let log_level = cfg
                    .log_level
                    .as_deref()
                    .map(LevelFilter::from_str)
                    .transpose()?
                    .unwrap_or(LevelFilter::Info);
                logger::init_with_level(log_level)?;
            }
            run_detection(cfg)
        }
    }
}

#[cfg(feature = "tracing")]
fn init_tracing(json: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if json {
        let _ = fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .json()
            .flatten_event(true)
            .finish()
            .try_init();
    } else {
        let _ = fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_timer(fmt::time::Uptime::default())
            .finish()
            .try_init();
    }
}
