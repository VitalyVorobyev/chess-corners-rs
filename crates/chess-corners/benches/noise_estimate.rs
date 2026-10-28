//! Per-frame cost of the noise estimate (`estimate_noise_sigma_u8`) on the
//! real test images. Skipped with a stderr note if an image is missing.

use chess_corners::estimate_noise_sigma_u8;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

mod common;
use common::load_test_image;

fn bench_noise_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_estimate");
    for name in ["small.png", "mid.png", "large.png"] {
        let Some((data, w, h)) = load_test_image(name) else {
            eprintln!("skipping noise_estimate/{name}: testimage not found");
            continue;
        };
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| black_box(estimate_noise_sigma_u8(black_box(data), w, h)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_noise_estimate);
criterion_main!(benches);
