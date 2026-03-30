use anyhow::Result;
use chess_corners::{DescriptorMode, DetectorMode, RefinementMethod, ThresholdMode};
use clap::{Parser, Subcommand};
use serde::de::DeserializeOwned;
use std::path::PathBuf;

mod commands;

#[cfg(not(feature = "tracing"))]
mod logger;
#[cfg(not(feature = "tracing"))]
use log::LevelFilter;
#[cfg(not(feature = "tracing"))]
use std::str::FromStr;

use commands::{apply_overrides, load_config, run_detection, DetectionOverrides};

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
        pyramid_levels: Option<u8>,
        /// Override pyramid min size.
        #[arg(long)]
        pyramid_min_size: Option<u32>,
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
        /// Override threshold mode (`relative` or `absolute`).
        #[arg(long)]
        threshold_mode: Option<String>,
        /// Override threshold value.
        #[arg(long)]
        threshold_value: Option<f32>,
        /// Override detector mode (`canonical` or `broad`).
        #[arg(long)]
        detector_mode: Option<String>,
        /// Override descriptor mode (`follow_detector`, `canonical`, `broad`).
        #[arg(long)]
        descriptor_mode: Option<String>,
        /// NMS radius override.
        #[arg(long)]
        nms_radius: Option<u32>,
        /// Min cluster size override.
        #[arg(long)]
        min_cluster_size: Option<u32>,
        /// Override the active refiner kind (`center_of_mass`, `forstner`, `saddle_point`).
        #[arg(long)]
        refiner_kind: Option<String>,
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
            #[cfg(feature = "tracing")]
            json_trace,
        } => {
            #[cfg(feature = "tracing")]
            init_tracing(json_trace);
            let mut cfg = load_config(&config)?;
            let overrides = DetectionOverrides {
                pyramid_levels,
                pyramid_min_size,
                refinement_radius,
                merge_radius,
                output_json,
                output_png,
                threshold_mode: parse_flag_enum::<ThresholdMode>(threshold_mode.as_deref())?,
                threshold_value,
                detector_mode: parse_flag_enum::<DetectorMode>(detector_mode.as_deref())?,
                descriptor_mode: parse_flag_enum::<DescriptorMode>(descriptor_mode.as_deref())?,
                nms_radius,
                min_cluster_size,
                refiner_kind: parse_flag_enum::<RefinementMethod>(refiner_kind.as_deref())?,
            };
            apply_overrides(&mut cfg, overrides);

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

fn parse_flag_enum<T>(value: Option<&str>) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    match value {
        Some(raw) => {
            let json = format!("\"{raw}\"");
            let parsed = serde_json::from_str(&json)?;
            Ok(Some(parsed))
        }
        None => Ok(None),
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
