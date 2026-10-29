//! Per-refiner microbenchmark: cost of `refine()` per corner.
//!
//! Builds a synthetic anti-aliased chessboard, computes a ChESS
//! response, harvests integer-rounded NMS seeds, and times each of the
//! two non-ML refiners (CenterOfMass, SaddlePoint)
//! over a fixed seed list.
//!
//! The harness keeps the seed list constant across refiners so the
//! resulting per-refiner numbers are directly comparable. Throughput
//! is reported in seeds/second. Each refiner runs with the position
//! covariance on (`propagated`) and off (`off`); the Radon group times the
//! peak-fit covariance alone (the peak fit itself is part of detection).

use chess_corners_core::{
    chess_response_u8, detect_corners_from_response, detect_peaks_from_radon, radon_response_u8,
    CenterOfMassConfig, CenterOfMassRefiner, ChessParams, CornerRefiner, DenseDetector, ImageView,
    RadonBuffers, RadonDetector, RadonDetectorParams, RefineContext, Refiner, RefinerKind,
    ResponseMap, SaddlePointConfig,
};
use chess_corners_testutil::aa_chessboard;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

/// Holds the frozen scene used by every refiner: image, response, seeds.
struct Scene {
    img: Vec<u8>,
    side: usize,
    resp: ResponseMap,
    seeds: Vec<[f32; 2]>,
}

fn build_scene(side: usize, cell: usize) -> Scene {
    let img = aa_chessboard(
        side,
        cell,
        (cell as f32 / 2.0 + 0.31, cell as f32 / 2.0 + 0.47),
        30,
        230,
    );
    let params = ChessParams::default();
    let resp = chess_response_u8(&img, side, side, &params);
    // Use the existing default detector to harvest a stable seed list.
    let corners = detect_corners_from_response(&resp, &params);
    let seeds: Vec<[f32; 2]> = corners.iter().map(|c| [c.x.round(), c.y.round()]).collect();
    Scene {
        img,
        side,
        resp,
        seeds,
    }
}

fn ctx<'a>(scene: &'a Scene) -> RefineContext<'a> {
    RefineContext::new(
        ImageView::from_u8_slice(scene.side, scene.side, &scene.img),
        Some(&scene.resp),
    )
}

fn bench_refiners(c: &mut Criterion) {
    // 256, 512, 1024 pixel sides give roughly 30 / 120 / 480 seeds at
    // cell = 12 — covers the small / medium / large regimes the
    // descriptor_fit bench targets, but with realistic image data
    // instead of synthetic seed lists.
    let sides = [256usize, 512, 1024];
    let cell = 12usize;

    let mut group = c.benchmark_group("refiners");
    for &side in &sides {
        let scene = build_scene(side, cell);
        if scene.seeds.is_empty() {
            eprintln!("skipping refiners/{side}: scene produced no seeds");
            continue;
        }
        group.throughput(Throughput::Elements(scene.seeds.len() as u64));

        type Build = fn() -> Refiner;
        let cases: [(&str, Build); 2] = [
            ("center_of_mass", || {
                Refiner::CenterOfMass(CenterOfMassRefiner::new(CenterOfMassConfig::default()))
            }),
            ("saddle_point", || {
                Refiner::from_kind(RefinerKind::SaddlePoint(SaddlePointConfig::default()))
            }),
        ];

        for (label, build) in cases {
            for (tier, covariance) in [("propagated", true), ("off", false)] {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{label}/{tier}"),
                        format!("{side}x{side}_n{}", scene.seeds.len()),
                    ),
                    &build,
                    |b, build| {
                        let mut refiner = build().with_covariance(covariance);
                        b.iter(|| {
                            let mut acc = 0.0f32;
                            for seed in &scene.seeds {
                                let r = refiner.refine(*seed, ctx(&scene));
                                acc += r.x + r.y;
                            }
                            black_box(acc)
                        });
                    },
                );
            }
        }
    }
    group.finish();

    let mut group = c.benchmark_group("radon_peak_covariance");
    for &side in &sides {
        let scene = build_scene(side, cell);
        let params = RadonDetectorParams::default();
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&scene.img, side, side, &params, &mut buffers);
        let peaks = detect_peaks_from_radon(&resp, &params);
        let view = ImageView::from_u8_slice(side, side, &scene.img).unwrap();
        group.throughput(Throughput::Elements(peaks.len() as u64));
        group.bench_function(format!("{side}x{side}_n{}", peaks.len()), |b| {
            b.iter(|| {
                let mut corners = peaks.clone();
                RadonDetector.attach_peak_covariance(&mut corners, view, &params, &mut buffers);
                black_box(corners.len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_refiners);
criterion_main!(benches);
