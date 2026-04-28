//! Per-refiner microbenchmark: cost of `refine()` per corner.
//!
//! Builds a synthetic anti-aliased chessboard, computes a ChESS
//! response, harvests integer-rounded NMS seeds, and times each of the
//! four non-ML refiners (CenterOfMass, Förstner, SaddlePoint, RadonPeak)
//! over a fixed seed list.
//!
//! The harness keeps the seed list constant across refiners so the
//! resulting per-refiner numbers are directly comparable. Throughput
//! is reported in seeds/second.

use chess_corners_core::detect::detect_corners_from_response;
use chess_corners_core::imageview::ImageView;
use chess_corners_core::refine::{
    CenterOfMassConfig, CornerRefiner, ForstnerConfig, RefineContext, SaddlePointConfig,
};
use chess_corners_core::refine_radon::RadonPeakConfig;
use chess_corners_core::response::chess_response_u8;
use chess_corners_core::{ChessParams, Refiner, RefinerKind, ResponseMap};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const SUPER: usize = 8;

fn synthetic_chessboard_aa(size: usize, cell: usize, offset: (f32, f32)) -> Vec<u8> {
    let (ox, oy) = offset;
    let c = cell as f32;
    let inv_super2 = 1.0 / (SUPER * SUPER) as f32;
    let mut img = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0f32;
            for sy in 0..SUPER {
                let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                let cy = ((yf - oy) / c).floor() as i32;
                for sx in 0..SUPER {
                    let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                    let cx = ((xf - ox) / c).floor() as i32;
                    let dark_cell = (cx + cy).rem_euclid(2) == 0;
                    acc += if dark_cell { 30.0 } else { 230.0 };
                }
            }
            img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

/// Holds the frozen scene used by every refiner: image, response, seeds.
struct Scene {
    img: Vec<u8>,
    side: usize,
    resp: ResponseMap,
    seeds: Vec<[f32; 2]>,
}

fn build_scene(side: usize, cell: usize) -> Scene {
    let img = synthetic_chessboard_aa(
        side,
        cell,
        (cell as f32 / 2.0 + 0.31, cell as f32 / 2.0 + 0.47),
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
    RefineContext {
        image: ImageView::from_u8_slice(scene.side, scene.side, &scene.img),
        response: Some(&scene.resp),
    }
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
            (
                "radon_peak",
                RefinerKind::RadonPeak(RadonPeakConfig::default()),
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
