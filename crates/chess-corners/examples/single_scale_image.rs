//! Single-scale ChESS detection on a grayscale image.
//!
//! Usage:
//!   cargo run -p chess-corners --example single_scale_image -- path/to/image.png

use chess_corners::{ChessConfig, CornerDescriptor};
use image::ImageReader;
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let img_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("testimages/mid.png"));

    let img = ImageReader::open(&img_path)?.decode()?.to_luma8();

    let cfg = ChessConfig::single_scale();

    let corners = chess_corners::find_chess_corners_image(&img, &cfg);
    println!("image: {}", img_path.display());
    println!("found {} corners", corners.len());

    if let Some(best) = corners
        .iter()
        .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
    {
        print_corner("strongest corner", best);
    }

    Ok(())
}

fn print_corner(label: &str, c: &CornerDescriptor) {
    println!(
        "{label}: ({:.2}, {:.2}), response {:.1}, theta {:.2} rad",
        c.x, c.y, c.response, c.orientation
    );
}
