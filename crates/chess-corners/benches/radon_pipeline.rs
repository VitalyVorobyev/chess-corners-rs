//! Whole-image Radon detector pipeline benchmark.
//!
//! Measures end-to-end `find_chess_corners_u8` latency under
//! `ChessConfig::radon()` — i.e. SAT build + dense response +
//! box-blur + threshold/NMS + 3-point peak fit + descriptor sampling.
//! Complements `chess-corners-core/benches/radon_response.rs`, which
//! benches only the response stage.
//!
//! Runs on:
//!
//! - **Synthetic chessboards** at 640×480, 1280×720, 1920×1080. These
//!   are deterministic across machines, so absolute speedups are
//!   directly comparable run-to-run.
//! - **Real test images** (`testimages/{small,mid,large}.png`) when
//!   the `image` dev-dependency loads them successfully. Skipped
//!   silently (with a stderr note) if the file is missing.

use chess_corners::{find_chess_corners_u8, ChessConfig};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use image::ImageReader;
use std::hint::black_box;
use std::path::{Path, PathBuf};

fn synth_chessboard(w: usize, h: usize) -> Vec<u8> {
    let cell = (h.min(w) / 25).max(8) as i32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = (x as i32) / cell;
            let cy = (y as i32) / cell;
            out[y * w + x] = if (cx + cy) & 1 == 0 { 40 } else { 215 };
        }
    }
    out
}

fn load_test_image(name: &str) -> Option<(Vec<u8>, u32, u32)> {
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

fn bench_radon_pipeline_synth(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let cfg = ChessConfig::radon();
    let mut group = c.benchmark_group("radon_pipeline_synth");
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(format!("{}x{}", w, h), |b| {
            b.iter(|| {
                let corners = find_chess_corners_u8(&img, w as u32, h as u32, &cfg).unwrap();
                black_box(corners.len())
            });
        });
    }
    group.finish();
}

fn bench_radon_pipeline_real(c: &mut Criterion) {
    let cfg = ChessConfig::radon();
    let mut group = c.benchmark_group("radon_pipeline_real");
    for name in ["small.png", "mid.png", "large.png"] {
        let Some((data, w, h)) = load_test_image(name) else {
            eprintln!("skipping radon_pipeline_real/{name}: testimage not found");
            continue;
        };
        group.throughput(Throughput::Elements((w as u64) * (h as u64)));
        group.bench_function(name, |b| {
            b.iter(|| {
                let corners = find_chess_corners_u8(&data, w, h, &cfg).unwrap();
                black_box(corners.len())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_radon_pipeline_synth,
    bench_radon_pipeline_real
);
criterion_main!(benches);
