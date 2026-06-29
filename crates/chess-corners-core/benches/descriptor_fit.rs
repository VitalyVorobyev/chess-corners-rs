//! Per-corner orientation/descriptor microbenchmarks.
//!
//! Inputs come from the shared synthetic boards (SOLID-01) so timings
//! are comparable with the other core benches. Two groups:
//!
//! - `descriptor_fit` — batched [`describe_corners`] throughput with the
//!   default RingFit method, lifting N raw corners to descriptors in one
//!   call (hard board).
//! - `orientation_fit` — single-corner [`fit_axes_at_point`] cost for
//!   both [`OrientationMethod`] variants (PERF-03 RingFit, PERF-04
//!   DiskFit) across three fixtures that exercise each method's fast and
//!   slow paths.
//!
//!   The hard 40/215 board (`synth_chessboard`) is the worst
//!   case: the two-axis `tanh` model cannot represent an instantaneous
//!   step, so even a perfect corner fits with a large relative residual
//!   (`rel_rms ≈ 0.47`, scale-invariant in cell size). That residual
//!   sits above RingFit's robust-fallback trigger (0.12) and DiskFit's
//!   lazy-gate threshold (0.04), so RingFit always takes its
//!   deterministic grid-search path and DiskFit always pays the full
//!   disk cost — the `ringfit_corner`/`diskfit_corner` (90° crossing)
//!   and `*_edge` (single straight edge) cases.
//!
//!   The anti-aliased board (`synth_chessboard_soft`) matches
//!   the model: a clean corner settles at `rel_rms ≈ 0.01`, so
//!   `ringfit_soft_corner` reaches RingFit's fast 2nd-harmonic seed path
//!   and `diskfit_soft_corner` short-circuits through the lazy gate to
//!   that same ring fit.
//!
//!   The warped board (`synth_chessboard_warped`) skews the
//!   corner to ~61° axis separation — outside the lazy-gate band — so
//!   `diskfit_warped` runs the full disk estimator on its intended
//!   projective-skew input.
//!
//! ```text
//! cargo bench -p chess-corners-core --bench descriptor_fit
//! ```

use chess_corners_core::{describe_corners, fit_axes_at_point, Corner, OrientationMethod};
use chess_corners_testutil::{
    synth_chessboard, synth_chessboard_soft, synth_chessboard_warped, ORIENT_CORNER, ORIENT_DIM,
};
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
                        Some(OrientationMethod::RingFit),
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
    const RADIUS: u32 = 5;
    let dim = ORIENT_DIM;
    let (cx, cy) = ORIENT_CORNER;

    // All three boards share cell geometry (`cell = dim / 25 = 10`), so
    // the corner falls on a real cell crossing in each. On the hard and
    // soft boards it is a 4-quadrant 90° crossing; on the warped board
    // the axes are sheared to ~61° separation. The edge point sits on the
    // same vertical boundary at a cell mid-height: a single straight edge
    // with no crossing line (a degenerate, near-collapsed-axes input).
    let edge = (cx, cy + 5.0);
    let hard = synth_chessboard(dim, dim);
    let soft = synth_chessboard_soft(dim, dim);
    let warped = synth_chessboard_warped(dim, dim);

    // (label, image, point, method). Hard cases force the slow paths
    // (rel_rms ≈ 0.47); the soft corner reaches RingFit's fast seed path
    // and DiskFit's lazy-gate short-circuit (rel_rms ≈ 0.01); the warped
    // corner drives DiskFit's full disk estimator (sep ≈ 61°). `as_slice`
    // keeps the element type `&[u8]` without an annotation clippy flags
    // as over-complex.
    use OrientationMethod::{DiskFit, RingFit};
    let cases = [
        ("ringfit_corner", hard.as_slice(), (cx, cy), RingFit),
        ("ringfit_edge", hard.as_slice(), edge, RingFit),
        ("diskfit_corner", hard.as_slice(), (cx, cy), DiskFit),
        ("diskfit_edge", hard.as_slice(), edge, DiskFit),
        ("ringfit_soft_corner", soft.as_slice(), (cx, cy), RingFit),
        ("diskfit_soft_corner", soft.as_slice(), (cx, cy), DiskFit),
        ("diskfit_warped", warped.as_slice(), (cx, cy), DiskFit),
    ];

    let mut group = c.benchmark_group("orientation_fit");
    group.throughput(Throughput::Elements(1));
    for (label, img, (x, y), method) in cases {
        group.bench_function(label, |b| {
            b.iter(|| {
                black_box(fit_axes_at_point(
                    black_box(img),
                    dim,
                    dim,
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
