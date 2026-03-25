use box_image_pyramid::{build_pyramid, ImageView, PyramidBuffers, PyramidParams};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fast_image_resize as fir;
use image::imageops::FilterType as ImageFilterType;
use image::{GrayImage, ImageBuffer, Luma};
use rgb::FromSlice;
use std::time::Duration;

const NUM_LEVELS: u8 = 6;
const MIN_SIZE: usize = 32;

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    width: usize,
    height: usize,
}

impl BenchCase {
    fn bytes(self) -> u64 {
        (self.width * self.height) as u64
    }
}

#[derive(Clone, Copy)]
struct LevelDim {
    width: u32,
    height: u32,
}

struct BenchInput {
    case: BenchCase,
    pixels: Vec<u8>,
    gray: GrayImage,
    level_dims: Vec<LevelDim>,
}

impl BenchInput {
    fn new(case: BenchCase, params: &PyramidParams) -> Self {
        let pixels = make_deterministic_gray(case.width, case.height);
        let gray = GrayImage::from_raw(case.width as u32, case.height as u32, pixels.clone())
            .expect("gray image dimensions match");
        let level_dims = pyramid_dims(case.width, case.height, params);

        Self {
            case,
            pixels,
            gray,
            level_dims,
        }
    }
}

fn make_deterministic_gray(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = ((x * 31) ^ (y * 17) ^ ((x * y) >> 3) ^ ((x + y) * 7)) & 0xff;
            pixels[idx] = value as u8;
        }
    }
    pixels
}

fn pyramid_dims(width: usize, height: usize, params: &PyramidParams) -> Vec<LevelDim> {
    if params.num_levels == 0 || width < params.min_size || height < params.min_size {
        return Vec::new();
    }

    let mut dims = Vec::with_capacity(params.num_levels as usize);
    dims.push(LevelDim {
        width: width as u32,
        height: height as u32,
    });

    let mut current_w = width;
    let mut current_h = height;

    for _ in 1..params.num_levels {
        let next_w = current_w / 2;
        let next_h = current_h / 2;

        if next_w == 0 || next_h == 0 || next_w < params.min_size || next_h < params.min_size {
            break;
        }

        dims.push(LevelDim {
            width: next_w as u32,
            height: next_h as u32,
        });
        current_w = next_w;
        current_h = next_h;
    }

    dims
}

fn alloc_gray_levels(level_dims: &[LevelDim]) -> Vec<GrayImage> {
    level_dims
        .get(1..)
        .unwrap_or(&[])
        .iter()
        .map(|dim| ImageBuffer::<Luma<u8>, Vec<u8>>::new(dim.width, dim.height))
        .collect()
}

fn sink_borrowed_pyramid(pyramid: &box_image_pyramid::Pyramid<'_>) -> usize {
    pyramid
        .levels
        .iter()
        .enumerate()
        .fold(0usize, |acc, (idx, level)| {
            acc.wrapping_add(idx)
                .wrapping_add(level.img.width)
                .wrapping_add(level.img.height)
                .wrapping_add(level.img.data.first().copied().unwrap_or_default() as usize)
        })
}

fn sink_gray_levels(base: &GrayImage, levels: &[GrayImage]) -> usize {
    let mut acc = base.width() as usize
        + base.height() as usize
        + base.as_raw().first().copied().unwrap_or_default() as usize;

    for (idx, level) in levels.iter().enumerate() {
        acc = acc
            .wrapping_add(idx)
            .wrapping_add(level.width() as usize)
            .wrapping_add(level.height() as usize)
            .wrapping_add(level.as_raw().first().copied().unwrap_or_default() as usize);
    }

    acc
}

fn sink_raw_levels(
    base: &[u8],
    base_width: usize,
    base_height: usize,
    levels: &[Vec<u8>],
) -> usize {
    let mut acc = base_width
        .wrapping_add(base_height)
        .wrapping_add(base.first().copied().unwrap_or_default() as usize);

    for (idx, level) in levels.iter().enumerate() {
        acc = acc
            .wrapping_add(idx)
            .wrapping_add(level.len())
            .wrapping_add(level.first().copied().unwrap_or_default() as usize);
    }

    acc
}

fn bench_image_resize(input: &BenchInput) -> usize {
    let mut acc = input.case.width
        + input.case.height
        + input.pixels.first().copied().unwrap_or_default() as usize;
    let mut current = input.gray.clone();

    for (idx, dim) in input.level_dims.iter().skip(1).enumerate() {
        let next =
            image::imageops::resize(&current, dim.width, dim.height, ImageFilterType::Triangle);
        acc = acc
            .wrapping_add(idx)
            .wrapping_add(next.width() as usize)
            .wrapping_add(next.height() as usize)
            .wrapping_add(next.as_raw().first().copied().unwrap_or_default() as usize);
        current = next;
    }

    acc
}
fn pyramid_benchmarks(c: &mut Criterion) {
    let params = PyramidParams {
        num_levels: NUM_LEVELS,
        min_size: MIN_SIZE,
    };
    let cases = [
        BenchCase {
            name: "640x480",
            width: 640,
            height: 480,
        },
        BenchCase {
            name: "1920x1080",
            width: 1920,
            height: 1080,
        },
        BenchCase {
            name: "4032x3024",
            width: 4032,
            height: 3024,
        },
    ];

    let inputs: Vec<BenchInput> = cases
        .iter()
        .copied()
        .map(|case| BenchInput::new(case, &params))
        .collect();

    let mut group = c.benchmark_group("pyramid_builders");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    for input in &inputs {
        group.throughput(Throughput::Bytes(input.case.bytes()));

        group.bench_with_input(
            BenchmarkId::new("box_image_pyramid/reuse_buffers", input.case.name),
            input,
            |b, input| {
                let mut buffers = PyramidBuffers::with_capacity(params.num_levels);
                let base = ImageView::new(input.case.width, input.case.height, &input.pixels)
                    .expect("base image view dimensions match");

                b.iter(|| {
                    let pyramid = build_pyramid(base, &params, &mut buffers);
                    black_box(sink_borrowed_pyramid(&pyramid));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("box_image_pyramid/fresh_buffers", input.case.name),
            input,
            |b, input| {
                let base = ImageView::new(input.case.width, input.case.height, &input.pixels)
                    .expect("base image view dimensions match");

                b.iter(|| {
                    let mut buffers = PyramidBuffers::new();
                    let pyramid = build_pyramid(base, &params, &mut buffers);
                    black_box(sink_borrowed_pyramid(&pyramid));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fast_image_resize/box", input.case.name),
            input,
            |b, input| {
                let mut levels = alloc_gray_levels(&input.level_dims);
                let mut resizer = fir::Resizer::new();
                let options = fir::ResizeOptions::new()
                    .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Box));

                b.iter(|| {
                    for i in 0..levels.len() {
                        let (head, tail) = levels.split_at_mut(i);
                        let dst = &mut tail[0];
                        if i == 0 {
                            resizer
                                .resize(&input.gray, dst, &options)
                                .expect("fast_image_resize level 1");
                        } else {
                            let src = &head[i - 1];
                            resizer
                                .resize(src, dst, &options)
                                .expect("fast_image_resize pyramid level");
                        }
                    }
                    black_box(sink_gray_levels(&input.gray, &levels));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("resize/box", input.case.name),
            input,
            |b, input| {
                let mut levels: Vec<Vec<u8>> = input.level_dims[1..]
                    .iter()
                    .map(|dim| vec![0u8; dim.width as usize * dim.height as usize])
                    .collect();
                let mut resizers = input
                    .level_dims
                    .windows(2)
                    .map(|dims| {
                        resize::new(
                            dims[0].width as usize,
                            dims[0].height as usize,
                            dims[1].width as usize,
                            dims[1].height as usize,
                            resize::Pixel::Gray8,
                            resize::Type::Custom(resize::Filter::box_filter(0.5)),
                        )
                        .expect("resize crate resizer")
                    })
                    .collect::<Vec<_>>();

                b.iter(|| {
                    for i in 0..levels.len() {
                        let (head, tail) = levels.split_at_mut(i);
                        let dst = &mut tail[0];
                        let src: &[u8] = if i == 0 { &input.pixels } else { &head[i - 1] };

                        resizers[i]
                            .resize(src.as_gray(), dst.as_gray_mut())
                            .expect("resize crate pyramid level");
                    }
                    black_box(sink_raw_levels(
                        &input.pixels,
                        input.case.width,
                        input.case.height,
                        &levels,
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image/triangle_allocating", input.case.name),
            input,
            |b, input| {
                b.iter(|| {
                    black_box(bench_image_resize(input));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, pyramid_benchmarks);
criterion_main!(benches);
