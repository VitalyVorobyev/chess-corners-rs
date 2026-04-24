//! Benchmark the whole-image Radon response against the ChESS
//! response kernel on representative frame sizes.
//!
//! Measures:
//!
//! - `radon_response_u8` at `image_upsample ∈ {1, 2}` — the SAT-based
//!   4-angle `(max − min)²` response that drives the Duda-Frese
//!   detector.
//! - `chess_response_u8` with the canonical r=5 ring kernel, for
//!   relative scale.
//!
//! Both kernels produce a dense response map; the two costs are not
//! directly interchangeable (Radon does subpixel refinement inside
//! the detector pass while ChESS delegates it to a refiner), but this
//! bench gives the reader a single number to reason about whole-image
//! latency on 640×480 / 1280×720 / 1920×1080 frames.

use chess_corners_core::response::chess_response_u8;
use chess_corners_core::{radon_response_u8, ChessParams, RadonBuffers, RadonDetectorParams};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

/// Deterministic chessboard-like synthetic image so runs are
/// comparable across machines. Corner density is roughly 25 cells
/// across the shorter image dimension, which matches a typical
/// calibration target in frame.
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

fn bench_radon_response(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let mut group = c.benchmark_group("radon_response");
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        for up in [1u32, 2u32] {
            let params = RadonDetectorParams {
                image_upsample: up,
                ..RadonDetectorParams::default()
            };
            group.bench_function(format!("{}x{}_up{}", w, h, up), |b| {
                b.iter_batched(
                    RadonBuffers::new,
                    |mut buffers| {
                        let resp = radon_response_u8(&img, w, h, &params, &mut buffers);
                        black_box(resp.data().len())
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_chess_response(c: &mut Criterion) {
    let dims: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
    let mut group = c.benchmark_group("chess_response");
    let params = ChessParams::default();
    for &(w, h) in dims {
        let img = synth_chessboard(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(format!("{}x{}", w, h), |b| {
            b.iter(|| {
                let resp = chess_response_u8(&img, w, h, &params);
                black_box(resp.data().len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_radon_response, bench_chess_response);
criterion_main!(benches);
