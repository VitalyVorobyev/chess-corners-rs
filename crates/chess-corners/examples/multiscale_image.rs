//! Multiscale ChESS detection with a 3-level pyramid.
//!
//! Usage:
//!   cargo run -p chess-corners --example multiscale_image -- path/to/image.png

use chess_corners::ChessConfig;
use image::ImageReader;
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let img_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("testimages/large.png"));

    let img = ImageReader::open(&img_path)?.decode()?.to_luma8();

    let mut cfg = ChessConfig::multiscale();
    cfg.pyramid_min_size = 64;

    let corners = chess_corners::find_chess_corners_image(&img, &cfg)?;
    println!("image: {}", img_path.display());
    println!(
        "multiscale: levels={}, min_size={}, refinement_radius={}, merge_radius={}",
        cfg.pyramid_levels, cfg.pyramid_min_size, cfg.refinement_radius, cfg.merge_radius
    );
    println!("found {} corners", corners.len());

    Ok(())
}
