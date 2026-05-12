//! Multiscale ChESS detection with a 3-level pyramid.
//!
//! Usage:
//!   cargo run -p chess-corners --example multiscale_image -- path/to/image.png

use chess_corners::{Detector, DetectorConfig, MultiscaleConfig};
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

    let mut cfg = DetectorConfig::multiscale();
    if let MultiscaleConfig::Pyramid {
        ref mut min_size, ..
    } = cfg.multiscale
    {
        *min_size = 64;
    }

    let mut detector = Detector::new(cfg)?;
    let corners = detector.detect(&img)?;
    println!("image: {}", img_path.display());
    if let MultiscaleConfig::Pyramid {
        levels,
        min_size,
        refinement_radius,
    } = cfg.multiscale
    {
        println!(
            "multiscale: levels={levels}, min_size={min_size}, \
             refinement_radius={refinement_radius}, merge_radius={}",
            cfg.merge_radius
        );
    }
    println!("found {} corners", corners.len());

    Ok(())
}
