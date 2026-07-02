//! Microbenchmark for the dense ChESS response kernel (PERF-01).
//!
//! Isolates [`chess_response_u8`] — the per-pixel 16-sample ring gather
//! plus SR/DR/MR accumulation — on synthetic chessboards at 256², 512²,
//! and 1024². Throughput is reported in pixels/second so the scalar and
//! `simd` builds can be compared directly:
//!
//! ```text
//! # scalar inner loop (auto-vectorized only):
//! cargo bench -p chess-corners-core --bench chess_response
//! # portable-SIMD inner loop (nightly):
//! cargo bench -p chess-corners-core --bench chess_response --features simd
//! ```
//!
//! Both builds compute the same response values and differ only in the
//! inner-loop implementation, so the ratio of the two throughputs is the
//! SIMD speedup for this kernel. (The `rayon` feature would also affect
//! this entry point; leave it off to measure single-thread kernel cost.)

use chess_corners_core::{chess_response_u8, ChessParams};
use chess_corners_testutil::synth_chessboard;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn bench_chess_response_kernel(c: &mut Criterion) {
    let sides: &[usize] = &[256, 512, 1024];
    let params = ChessParams::default();

    let mut group = c.benchmark_group("chess_response_kernel");
    for &side in sides {
        let img = synth_chessboard(side, side);
        // Throughput in pixels lets criterion print Mpix/s directly.
        group.throughput(Throughput::Elements((side * side) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{side}x{side}")),
            &img,
            |b, img| {
                b.iter(|| {
                    let resp = chess_response_u8(black_box(img), side, side, &params);
                    black_box(resp.data().len())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chess_response_kernel);
criterion_main!(benches);
