//! Full ChESS multiscale pipeline benchmark.
//!
//! Measures end-to-end `find_chess_corners_u8` latency under
//! `ChessConfig::multiscale()` and `ChessConfig::single_scale()` —
//! i.e. the multi-pyramid coarse-to-fine path and the single-scale
//! reference. Complements `chess-corners-core/benches/radon_response.rs`,
//! which benches only the dense response stage.
//!
//! Runs on:
//!
//! - **Synthetic chessboards** at 640×480, 1280×720, 1920×1080.
//! - **Real test images** (`testimages/{small,mid,large}.png`).
//!   Skipped with a stderr note if the file is missing.

use chess_corners::{find_chess_corners_u8, ChessConfig};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::ImageReader;
use std::hint::black_box;
use std::path::{Path, PathBuf};

type ConfigCtor = fn() -> ChessConfig;

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

fn bench_chess_pipeline_synth(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let presets: &[(&str, ConfigCtor)] = &[
        ("multiscale", ChessConfig::multiscale),
        ("single", ChessConfig::single_scale),
    ];
    let mut group = c.benchmark_group("chess_pipeline_synth");
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        for &(label, mk) in presets {
            let cfg = mk();
            group.bench_with_input(
                BenchmarkId::new(label, format!("{w}x{h}")),
                &cfg,
                |b, cfg| {
                    b.iter(|| {
                        let corners = find_chess_corners_u8(&img, w as u32, h as u32, cfg).unwrap();
                        black_box(corners.len())
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_chess_pipeline_real(c: &mut Criterion) {
    let presets: &[(&str, ConfigCtor)] = &[
        ("multiscale", ChessConfig::multiscale),
        ("single", ChessConfig::single_scale),
    ];
    let mut group = c.benchmark_group("chess_pipeline_real");
    for name in ["small.png", "mid.png", "large.png"] {
        let Some((data, w, h)) = load_test_image(name) else {
            eprintln!("skipping chess_pipeline_real/{name}: testimage not found");
            continue;
        };
        group.throughput(Throughput::Elements((w as u64) * (h as u64)));
        for &(label, mk) in presets {
            let cfg = mk();
            group.bench_with_input(BenchmarkId::new(label, name), &cfg, |b, cfg| {
                b.iter(|| {
                    let corners = find_chess_corners_u8(&data, w, h, cfg).unwrap();
                    black_box(corners.len())
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chess_pipeline_synth,
    bench_chess_pipeline_real
);
criterion_main!(benches);
