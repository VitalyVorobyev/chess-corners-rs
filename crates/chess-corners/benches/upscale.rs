//! Benchmark the optional pre-pipeline bilinear upscaling stage.

use chess_corners::upscale::{upscale_bilinear_u8, UpscaleBuffers};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn synth_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (((x * 31) ^ (y * 17)) & 0xff) as u8;
        }
    }
    out
}

fn bench_upscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("upscale_bilinear_u8");
    for (w, h) in [(320usize, 240), (640, 480), (1280, 720)] {
        let src = synth_gradient(w, h);
        for factor in [2u32, 3] {
            let dst_bytes = (w * h * (factor as usize) * (factor as usize)) as u64;
            group.throughput(Throughput::Bytes(dst_bytes));
            group.bench_with_input(
                BenchmarkId::new(format!("{w}x{h}"), format!("x{factor}")),
                &factor,
                |b, &factor| {
                    let mut buffers = UpscaleBuffers::new();
                    b.iter(|| {
                        let view = upscale_bilinear_u8(&src, w, h, factor, &mut buffers).unwrap();
                        black_box(view.data.len())
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_upscale);
criterion_main!(benches);
