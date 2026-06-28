use anyhow::Result;
use chess_corners::{ChessRing, Threshold};
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

use commands::{
    apply_overrides, load_config, run_detection, ChessRefinerSel, DetectionOverrides,
    RadonRefinerSel,
};

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
        /// Absolute threshold override. Mutually exclusive with
        /// `--threshold-relative`. Accepted values are non-negative
        /// floats in the detector's native score units.
        #[arg(long)]
        threshold_absolute: Option<f32>,
        /// Relative threshold override (fraction in `[0, 1]` of the
        /// per-frame response maximum). Mutually exclusive with
        /// `--threshold-absolute`.
        #[arg(long)]
        threshold_relative: Option<f32>,
        /// Override the ChESS ring (`canonical` or `broad`). Has no
        /// effect on the Radon strategy.
        #[arg(long)]
        chess_ring: Option<String>,
        /// NMS radius override (applied to whichever strategy is active).
        #[arg(long)]
        nms_radius: Option<u32>,
        /// Min cluster size override (applied to whichever strategy is active).
        #[arg(long)]
        min_cluster_size: Option<u32>,
        /// Override the ChESS subpixel refiner (`center_of_mass`,
        /// `forstner`, `saddle_point`). Has no effect when the active
        /// strategy is Radon.
        #[arg(long)]
        chess_refiner: Option<String>,
        /// Override the Radon subpixel refiner (`radon_peak`,
        /// `center_of_mass`). Has no effect when the active strategy
        /// is ChESS.
        #[arg(long)]
        radon_refiner: Option<String>,
        /// Integer upscale factor (2, 3, or 4). If set, enables
        /// pre-pipeline bilinear upscaling. Pass `0` or omit to leave
        /// the JSON config value unchanged.
        #[arg(long)]
        upscale_factor: Option<u32>,
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
            threshold_absolute,
            threshold_relative,
            chess_ring,
            nms_radius,
            min_cluster_size,
            chess_refiner,
            radon_refiner,
            upscale_factor,
            #[cfg(feature = "tracing")]
            json_trace,
        } => {
            #[cfg(feature = "tracing")]
            init_tracing(json_trace);
            let mut cfg = load_config(&config)?;
            let threshold = match (threshold_absolute, threshold_relative) {
                (Some(_), Some(_)) => anyhow::bail!(
                    "--threshold-absolute and --threshold-relative are mutually exclusive",
                ),
                (Some(v), None) => Some(Threshold::Absolute(v)),
                (None, Some(v)) => Some(Threshold::Relative(v)),
                (None, None) => None,
            };
            let overrides = DetectionOverrides {
                pyramid_levels,
                pyramid_min_size,
                refinement_radius,
                merge_radius,
                output_json,
                output_png,
                threshold,
                chess_ring: parse_flag_enum::<ChessRing>(chess_ring.as_deref())?,
                nms_radius,
                min_cluster_size,
                chess_refiner: parse_flag_enum::<ChessRefinerSel>(chess_refiner.as_deref())?,
                radon_refiner: parse_flag_enum::<RadonRefinerSel>(radon_refiner.as_deref())?,
                upscale_factor,
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
