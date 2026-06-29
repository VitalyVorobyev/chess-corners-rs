//! Per-refiner microbenchmark: cost of `refine()` per corner.
//!
//! Builds a synthetic anti-aliased chessboard, computes a ChESS
//! response, harvests integer-rounded NMS seeds, and times each of the
//! three non-ML refiners (CenterOfMass, Förstner, SaddlePoint)
//! over a fixed seed list.
//!
//! The harness keeps the seed list constant across refiners so the
//! resulting per-refiner numbers are directly comparable. Throughput
//! is reported in seeds/second.

use chess_corners_core::unstable::{ChessParams, RefinerKind};
use chess_corners_core::{
    chess_response_u8, detect_corners_from_response, CenterOfMassConfig, CornerRefiner,
    ForstnerConfig, ImageView, RefineContext, Refiner, ResponseMap, SaddlePointConfig,
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

        let cases: &[(&str, RefinerKind)] = &[
            (
                "center_of_mass",
                RefinerKind::CenterOfMass(CenterOfMassConfig::default()),
            ),
            ("forstner", RefinerKind::Forstner(ForstnerConfig::default())),
            (
                "saddle_point",
                RefinerKind::SaddlePoint(SaddlePointConfig::default()),
            ),
        ];

        for (label, kind) in cases {
            group.bench_with_input(
                BenchmarkId::new(*label, format!("{side}x{side}_n{}", scene.seeds.len())),
                kind,
                |b, kind| {
                    let mut refiner = Refiner::from_kind(kind.clone());
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
    group.finish();
}

criterion_group!(benches, bench_refiners);
criterion_main!(benches);
