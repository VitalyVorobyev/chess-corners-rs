//! Hot-loop binary for sampling profilers (`samply`, `cargo-flamegraph`).
//!
//! The example deliberately keeps argument parsing minimal so the
//! profile target itself contributes negligible overhead. Compile with
//! release-profile debug-info to get readable stack frames:
//!
//! ```sh
//! CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
//!   cargo build --release --example profile_target -p chess-corners
//! ```
//!
//! Usage:
//!
//! ```sh
//! profile_target --mode <chess|radon> \
//!                [--refiner <com|forstner|saddle|radon>] \
//!                --image <path> \
//!                [--iters N]
//! ```

use chess_corners::{
    find_chess_corners_u8, find_chess_corners_u8_with_refiner, CenterOfMassConfig, ChessConfig,
    ForstnerConfig, RadonPeakConfig, RefinerKind, SaddlePointConfig,
};
use image::ImageReader;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Chess,
    Radon,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RefinerSel {
    CenterOfMass,
    Forstner,
    Saddle,
    Radon,
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    match s {
        "chess" => Ok(Mode::Chess),
        "radon" => Ok(Mode::Radon),
        other => Err(format!("unknown mode '{other}' (expected chess|radon)")),
    }
}

fn parse_refiner(s: &str) -> Result<RefinerSel, String> {
    match s {
        "com" | "center_of_mass" | "centerofmass" => Ok(RefinerSel::CenterOfMass),
        "forstner" => Ok(RefinerSel::Forstner),
        "saddle" | "saddle_point" | "saddlepoint" => Ok(RefinerSel::Saddle),
        "radon" | "radon_peak" | "radonpeak" => Ok(RefinerSel::Radon),
        other => Err(format!(
            "unknown refiner '{other}' (expected com|forstner|saddle|radon)"
        )),
    }
}

fn refiner_kind(sel: RefinerSel) -> RefinerKind {
    match sel {
        RefinerSel::CenterOfMass => RefinerKind::CenterOfMass(CenterOfMassConfig::default()),
        RefinerSel::Forstner => RefinerKind::Forstner(ForstnerConfig::default()),
        RefinerSel::Saddle => RefinerKind::SaddlePoint(SaddlePointConfig::default()),
        RefinerSel::Radon => RefinerKind::RadonPeak(RadonPeakConfig::default()),
    }
}

struct Args {
    mode: Mode,
    refiner: RefinerSel,
    image: PathBuf,
    iters: u32,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut mode = None;
    let mut refiner = RefinerSel::CenterOfMass;
    let mut image = None;
    let mut iters: u32 = 200;

    let mut argv = env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--mode" => {
                let v = argv.next().ok_or("--mode requires a value")?;
                mode = Some(parse_mode(&v)?);
            }
            "--refiner" => {
                let v = argv.next().ok_or("--refiner requires a value")?;
                refiner = parse_refiner(&v)?;
            }
            "--image" => {
                let v = argv.next().ok_or("--image requires a path")?;
                image = Some(PathBuf::from(v));
            }
            "--iters" => {
                let v = argv.next().ok_or("--iters requires a value")?;
                iters = v.parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "profile_target --mode <chess|radon> [--refiner <com|forstner|saddle|radon>] --image <path> [--iters N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag '{other}'").into()),
        }
    }

    let mode = mode.ok_or("--mode is required")?;
    let image = image.ok_or("--image is required")?;
    Ok(Args {
        mode,
        refiner,
        image,
        iters,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let img = ImageReader::open(&args.image)?.decode()?.to_luma8();
    let (w, h) = (img.width(), img.height());
    let data = img.into_raw();

    let cfg = match args.mode {
        Mode::Chess => ChessConfig::multiscale(),
        Mode::Radon => ChessConfig::radon(),
    };
    let kind = refiner_kind(args.refiner);

    // Warm up: pages, caches, and any one-time allocations.
    let warm = match args.mode {
        Mode::Chess => find_chess_corners_u8_with_refiner(&data, w, h, &cfg, &kind).unwrap(),
        Mode::Radon => find_chess_corners_u8(&data, w, h, &cfg).unwrap(),
    };
    println!(
        "warm: {} corners in {}x{} ({:?} mode, refiner={:?})",
        warm.len(),
        w,
        h,
        as_str_mode(args.mode),
        as_str_refiner(args.refiner),
    );

    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..args.iters {
        let corners = match args.mode {
            Mode::Chess => find_chess_corners_u8_with_refiner(&data, w, h, &cfg, &kind).unwrap(),
            Mode::Radon => find_chess_corners_u8(&data, w, h, &cfg).unwrap(),
        };
        total = total.wrapping_add(corners.len());
    }
    let elapsed = start.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / args.iters as f64;
    println!(
        "{} iters, {:.3} ms/iter, total corners (sink) = {}",
        args.iters, per_iter_ms, total
    );
    Ok(())
}

fn as_str_mode(m: Mode) -> &'static str {
    match m {
        Mode::Chess => "chess",
        Mode::Radon => "radon",
    }
}

fn as_str_refiner(r: RefinerSel) -> &'static str {
    match r {
        RefinerSel::CenterOfMass => "com",
        RefinerSel::Forstner => "forstner",
        RefinerSel::Saddle => "saddle",
        RefinerSel::Radon => "radon",
    }
}
