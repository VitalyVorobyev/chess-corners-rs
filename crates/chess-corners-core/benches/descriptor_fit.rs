//! Per-corner orientation/descriptor microbenchmarks.
//!
//! All inputs come from the shared synthetic board
//! ([`common::synth_chessboard`], SOLID-01) so timings are comparable
//! with the other core benches. Two groups:
//!
//! - `descriptor_fit` — batched [`describe_corners`] throughput with the
//!   default RingFit method, lifting N raw corners to descriptors in one
//!   call.
//! - `orientation_fit` — single-corner [`fit_axes_at_point`] cost for
//!   both [`OrientationMethod`] variants (PERF-03 RingFit, PERF-04
//!   DiskFit), measured at a clean 90° checkerboard corner and at a
//!   single straight-edge point.
//!
//!   NOTE: the shared board has hard 40/215 steps. The soft two-axis
//!   `tanh` model fits a hard step with a large relative RMS — ~0.47
//!   even at a perfect corner (measured) — which is scale-invariant in
//!   cell size. That residual sits above both RingFit's robust-fallback
//!   trigger (0.12) and DiskFit's lazy-gate threshold (0.04), so on this
//!   board RingFit always takes its deterministic grid-search path and
//!   DiskFit always pays the full disk cost (the lazy gate never
//!   short-circuits). The DiskFit/RingFit ratio here is therefore the
//!   cost of the disk estimator when it runs. RingFit's fast
//!   2nd-harmonic seed path and DiskFit's gate short-circuit require
//!   soft (real/blurred) edges and are not reachable from the hard
//!   synthetic board.
//!
//! ```text
//! cargo bench -p chess-corners-core --bench descriptor_fit
//! ```

mod common;

use chess_corners_core::{describe_corners, fit_axes_at_point, Corner, OrientationMethod};
use common::synth_chessboard;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

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
    let img = synth_chessboard(W, H);
    let mut group = c.benchmark_group("descriptor_fit");
    for n in [64usize, 256, 1024] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("corners_{n}"), |b| {
            b.iter_batched(
                || synth_corners(n, W, H),
                |corners| {
                    black_box(describe_corners(
                        &img,
                        W,
                        H,
                        5,
                        corners,
                        OrientationMethod::RingFit,
                    ))
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// PERF-03 (RingFit) + PERF-04 (DiskFit): single-corner two-axis fit.
fn bench_orientation_fit(c: &mut Criterion) {
    const W: usize = 256;
    const H: usize = 256;
    const RADIUS: u32 = 5;

    let img = synth_chessboard(W, H);

    // `synth_chessboard` uses `cell = min(w, h) / 25 = 10` here, so cell
    // boundaries fall on multiples of 10. (130, 130) is a 4-quadrant 90°
    // checkerboard corner near the image centre. (130, 135) sits on the
    // same vertical boundary at a cell mid-height: a single straight edge
    // with no crossing line, a degenerate (near-collapsed-axes) input.
    // Both points have rel_rms well above the robust/lazy-gate
    // thresholds (hard step vs soft model), so RingFit runs its robust
    // grid path and DiskFit runs the full disk at both.
    let corner = (130.0f32, 130.0f32);
    let edge = (130.0f32, 135.0f32);

    let mut group = c.benchmark_group("orientation_fit");
    group.throughput(Throughput::Elements(1));
    for (label, (x, y), method) in [
        ("ringfit_corner", corner, OrientationMethod::RingFit),
        ("ringfit_edge", edge, OrientationMethod::RingFit),
        ("diskfit_corner", corner, OrientationMethod::DiskFit),
        ("diskfit_edge", edge, OrientationMethod::DiskFit),
    ] {
        group.bench_function(label, |b| {
            b.iter(|| {
                black_box(fit_axes_at_point(
                    black_box(&img),
                    W,
                    H,
                    black_box(x),
                    black_box(y),
                    RADIUS,
                    method,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_descriptor_fit, bench_orientation_fit);
criterion_main!(benches);
