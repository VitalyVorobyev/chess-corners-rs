//! Whole-image Radon detector pipeline benchmark.
//!
//! Measures end-to-end `Detector::detect_u8` latency under
//! `DetectorConfig::radon()` — i.e. SAT build + dense response +
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

use chess_corners::{Detector, DetectorConfig};
use chess_corners_testutil::synth_chessboard;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

mod common;
use common::load_test_image;

fn bench_radon_pipeline_synth(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let cfg = DetectorConfig::radon();
    let mut detector = Detector::new(cfg).unwrap();
    let mut group = c.benchmark_group("radon_pipeline_synth");
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(format!("{}x{}", w, h), |b| {
            b.iter(|| {
                let corners = detector.detect_u8(&img, w as u32, h as u32).unwrap();
                black_box(corners.len())
            });
        });
    }
    group.finish();
}

fn bench_radon_pipeline_real(c: &mut Criterion) {
    let cfg = DetectorConfig::radon();
    let mut detector = Detector::new(cfg).unwrap();
    let mut group = c.benchmark_group("radon_pipeline_real");
    for name in ["small.png", "mid.png", "large.png"] {
        let Some((data, w, h)) = load_test_image(name) else {
            eprintln!("skipping radon_pipeline_real/{name}: testimage not found");
            continue;
        };
        group.throughput(Throughput::Elements((w as u64) * (h as u64)));
        group.bench_function(name, |b| {
            b.iter(|| {
                let corners = detector.detect_u8(&data, w, h).unwrap();
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
