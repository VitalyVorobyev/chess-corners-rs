//! Benchmark the per-corner descriptor fit (two-axis intensity model
//! via Gauss-Newton, plus 1σ uncertainty extraction).

use chess_corners_core::descriptor::{corners_to_descriptors, Corner};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

/// Generate a synthetic chessboard-like image (two orthogonal stripe
/// patterns multiplied). Corner descriptors stay in the image interior.
fn synth_image(w: usize, h: usize) -> Vec<u8> {
    let period = 24i32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let rx = ((x as i32) / period) & 1;
            let ry = ((y as i32) / period) & 1;
            out[y * w + x] = if rx ^ ry == 0 { 40 } else { 215 };
        }
    }
    out
}

fn synth_corners(n: usize, w: usize, h: usize) -> Vec<Corner> {
    let mut v = Vec::with_capacity(n);
    let margin = 12;
    let cols = ((n as f32).sqrt() as usize).max(1);
    let rows = n.div_ceil(cols);
    let x_stride = ((w - 2 * margin) / cols).max(1);
    let y_stride = ((h - 2 * margin) / rows).max(1);
    for j in 0..rows {
        for i in 0..cols {
            if v.len() >= n {
                break;
            }
            v.push(Corner::new(
                (margin + i * x_stride) as f32 + 0.25,
                (margin + j * y_stride) as f32 + 0.25,
                1.0,
            ));
        }
    }
    v
}

fn bench_descriptor_fit(c: &mut Criterion) {
    const W: usize = 640;
    const H: usize = 480;
    let img = synth_image(W, H);
    let mut group = c.benchmark_group("descriptor_fit");
    for n in [64usize, 256, 1024] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("corners_{n}"), |b| {
            b.iter_batched(
                || synth_corners(n, W, H),
                |corners| black_box(corners_to_descriptors(&img, W, H, 5, corners)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_descriptor_fit);
criterion_main!(benches);
