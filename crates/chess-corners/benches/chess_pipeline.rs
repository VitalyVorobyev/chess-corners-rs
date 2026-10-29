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
//! - **Synthetic chessboards** at 640×480, 1280×720, 1920×1080, with the
//!   presets as they are (the default uncertainty tier).
//! - **Dense board** (2048×1536, ~1.7k corners, 48 px cells) under both
//!   uncertainty tiers: the regime where per-corner covariance work (and
//!   its `rayon` scheduling) is visible next to the dense response.
//! - **Real test images** (`testimages/{small,mid,large}.png`) under both
//!   uncertainty tiers, named in the id: `single/off`, `single/propagated`,
//!   `multiscale/off`, `multiscale/propagated`. Skipped with a stderr note
//!   if the file is missing.

use chess_corners::{Detector, DetectorConfig, NoiseLevel, Uncertainty};
use chess_corners_testutil::{aa_chessboard, synth_chessboard};
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
    let modes: &[(&str, ConfigCtor)] = &[
        ("multiscale", DetectorConfig::chess_multiscale),
        ("single", DetectorConfig::chess),
    ];
    let tiers = [
        ("off", Uncertainty::Off),
        (
            "propagated",
            Uncertainty::Propagated {
                noise: NoiseLevel::Estimate,
            },
        ),
    ];
    let mut group = c.benchmark_group("chess_pipeline_real");
    for name in ["small.png", "mid.png", "large.png"] {
        let Some((data, w, h)) = load_test_image(name) else {
            eprintln!("skipping chess_pipeline_real/{name}: testimage not found");
            continue;
        };
        group.throughput(Throughput::Elements((w as u64) * (h as u64)));
        for &(mode, mk) in modes {
            for (tier, uncertainty) in tiers {
                let cfg = mk().with_uncertainty(uncertainty);
                let mut detector = Detector::new(cfg).unwrap();
                let id = BenchmarkId::new(format!("{mode}/{tier}"), name);
                group.bench_with_input(id, &cfg, |b, _cfg| {
                    b.iter(|| {
                        let corners = detector.detect_u8(&data, w, h).unwrap();
                        black_box(corners.len())
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_chess_pipeline_dense(c: &mut Criterion) {
    const W: usize = 2048;
    const H: usize = 1536;
    // Row-major square board truncated to its first `H` rows.
    let mut img = aa_chessboard(W, 48, (20.3, 17.7), 40, 200);
    img.truncate(W * H);
    let tiers = [
        ("off", Uncertainty::Off),
        (
            "propagated",
            Uncertainty::Propagated {
                noise: NoiseLevel::Estimate,
            },
        ),
    ];
    let mut group = c.benchmark_group("chess_pipeline_dense");
    group.sample_size(20);
    group.throughput(Throughput::Elements((W * H) as u64));
    for (tier, uncertainty) in tiers {
        let cfg = DetectorConfig::chess().with_uncertainty(uncertainty);
        let mut detector = Detector::new(cfg).unwrap();
        let n = detector.detect_u8(&img, W as u32, H as u32).unwrap().len();
        eprintln!("chess_pipeline_dense/single/{tier}: {n} corners");
        group.bench_function(
            BenchmarkId::new(format!("single/{tier}"), "2048x1536"),
            |b| {
                b.iter(|| {
                    let corners = detector.detect_u8(&img, W as u32, H as u32).unwrap();
                    black_box(corners.len())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chess_pipeline_synth,
    bench_chess_pipeline_real,
    bench_chess_pipeline_dense
);
criterion_main!(benches);
