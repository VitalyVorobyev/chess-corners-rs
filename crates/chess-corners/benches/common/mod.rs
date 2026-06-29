//! Shared helpers for the `chess-corners` facade benches.
//!
//! Lives in a `benches/common/` subdirectory (not a top-level `benches/*.rs`)
//! so Cargo does not treat it as its own bench target; each bench binary
//! pulls it in with `mod common;`. The synthetic boards come from the
//! workspace `chess-corners-testutil` crate; only the real-image loader —
//! which needs the `image` dev-dependency — lives here.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use image::ImageReader;

/// Load a grayscale test image from `testimages/<name>`, relative to the
/// crate manifest. Returns `None` (so the caller can skip the case) when the
/// file is missing or cannot be decoded.
pub fn load_test_image(name: &str) -> Option<(Vec<u8>, u32, u32)> {
    let mut path: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    path.push("..");
    path.push("..");
    path.push("testimages");
    path.push(name);
    let img = ImageReader::open(&path).ok()?.decode().ok()?.to_luma8();
    let (w, h) = (img.width(), img.height());
    Some((img.into_raw(), w, h))
}
