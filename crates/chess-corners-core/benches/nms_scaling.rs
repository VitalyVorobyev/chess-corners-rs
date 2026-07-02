//! Threshold -> NMS -> cluster-filter peak extraction scaling (PERF-05).
//!
//! Isolates stage 1 of ChESS detection
//! ([`detect_peaks_from_response_with_refine_radius`]) on a fixed dense
//! response map, sweeping the NMS radius over `{1, 2, 4, 8}`. The
//! response map is computed once (PERF-01 territory) and reused across
//! all radii, so the measured deltas are purely the cost of the
//! `(2r+1)²` local-max window and the positive-neighbour cluster count.
//!
//! ```text
//! cargo bench -p chess-corners-core --bench nms_scaling
//! ```
//!
//! `detect_peaks_from_response_with_refine_radius` is a stage-level entry
//! point re-exported from the `chess-corners-core` crate root.

use chess_corners_core::{
    chess_response_u8, detect_peaks_from_response_with_refine_radius, ChessParams,
};
use chess_corners_testutil::synth_chessboard;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn bench_nms_scaling(c: &mut Criterion) {
    const W: usize = 1024;
    const H: usize = 1024;

    // One dense response map shared across every radius. Default params
    // use `threshold = 0.0`, i.e. accept every strictly
    // positive response, which maximises the number of pixels that reach
    // the NMS window and therefore stresses the radius sweep.
    let img = synth_chessboard(W, H);
    let base = ChessParams::default();
    let resp = chess_response_u8(&img, W, H, &base);

    let mut group = c.benchmark_group("nms_scaling");
    // Pixels processed per call is constant (the whole response map), so
    // criterion prints a directly comparable Melem/s for every radius.
    group.throughput(Throughput::Elements((W * H) as u64));
    for &r in &[1u32, 2, 4, 8] {
        // `ChessParams` is `#[non_exhaustive]`: build via `Default` and
        // assign the public field rather than a struct literal.
        let mut params = ChessParams::default();
        params.nms_radius = r;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("radius_{r}")),
            &params,
            |b, params| {
                b.iter(|| {
                    let peaks =
                        detect_peaks_from_response_with_refine_radius(black_box(&resp), params, 0);
                    black_box(peaks.len())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_nms_scaling);
criterion_main!(benches);
