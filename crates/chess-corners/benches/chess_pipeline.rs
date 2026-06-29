//! Full ChESS multiscale pipeline benchmark.
//!
//! Measures end-to-end `Detector::detect_u8` latency under
//! `DetectorConfig::chess_multiscale()` and `DetectorConfig::chess()` —
//! i.e. the multi-pyramid coarse-to-fine path and the single-scale
//! reference. Complements `chess-corners-core/benches/radon_response.rs`,
//! which benches only the dense response stage.
//!
//! Runs on:
//!
//! - **Synthetic chessboards** at 640×480, 1280×720, 1920×1080.
//! - **Real test images** (`testimages/{small,mid,large}.png`).
//!   Skipped with a stderr note if the file is missing.

use chess_corners::{Detector, DetectorConfig};
use chess_corners_testutil::synth_chessboard;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

mod common;
use common::load_test_image;

type ConfigCtor = fn() -> DetectorConfig;

fn bench_chess_pipeline_synth(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let presets: &[(&str, ConfigCtor)] = &[
        ("multiscale", DetectorConfig::chess_multiscale),
        ("single", DetectorConfig::chess),
    ];
    let mut group = c.benchmark_group("chess_pipeline_synth");
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        for &(label, mk) in presets {
            let cfg = mk();
            let mut detector = Detector::new(cfg).unwrap();
            group.bench_with_input(
                BenchmarkId::new(label, format!("{w}x{h}")),
                &cfg,
                |b, _cfg| {
                    b.iter(|| {
                        let corners = detector.detect_u8(&img, w as u32, h as u32).unwrap();
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
        ("multiscale", DetectorConfig::chess_multiscale),
        ("single", DetectorConfig::chess),
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
            let mut detector = Detector::new(cfg).unwrap();
            group.bench_with_input(BenchmarkId::new(label, name), &cfg, |b, _cfg| {
                b.iter(|| {
                    let corners = detector.detect_u8(&data, w, h).unwrap();
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
