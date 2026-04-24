//! Dense refiner sweep — produces a JSON report consumed by the
//! book's Part V plots (`tools/book/plot_benchmark.py`).
//!
//! Sweeps sub-pixel offset × blur σ × noise σ × cell size against
//! every built-in refiner. Prints one JSON document to stdout:
//!
//! ```json
//! {
//!   "refiners": ["CenterOfMass", "Forstner", "SaddlePoint", "RadonPeak", "ML"],
//!   "results": [
//!     {
//!       "refiner": "CenterOfMass",
//!       "cell": 8,
//!       "blur_sigma": 0.0,
//!       "noise_sigma": 0.0,
//!       "mean": 0.0801,
//!       "p95": 0.117,
//!       "worst": 0.1226,
//!       "accept": 36,
//!       "total": 36,
//!       "errors": [0.0045, 0.0132, ...],
//!       "time_us_per_call": 0.022
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! Usage:
//! ```sh
//! cargo run --release -p chess-corners \
//!     --example bench_sweep --features ml-refiner \
//!     > book/src/img/bench/bench_sweep.json
//! ```

use std::time::Instant;

use chess_corners::ImageView;
use chess_corners_core::{
    refine::{
        CenterOfMassConfig, CenterOfMassRefiner, CornerRefiner, ForstnerConfig, ForstnerRefiner,
        RefineContext, RefineStatus, SaddlePointConfig, SaddlePointRefiner,
    },
    refine_radon::{RadonPeakConfig, RadonPeakRefiner},
    response::chess_response_u8,
    ChessParams, ResponseMap,
};

// ---------------------------------------------------------------------------
// Synthetic fixture — identical to `refiner_benchmark.rs`.

fn synthetic_chessboard_aa(
    size: usize,
    cell: usize,
    offset: (f32, f32),
    dark: u8,
    bright: u8,
) -> Vec<u8> {
    const SUPER: usize = 8;
    let (ox, oy) = offset;
    let c = cell as f32;
    let dark_f = dark as f32;
    let bright_f = bright as f32;
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
                    acc += if dark_cell { dark_f } else { bright_f };
                }
            }
            img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn gaussian_blur(img: &mut [u8], size: usize, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let klen = 2 * radius + 1;
    let mut kernel = vec![0f32; klen];
    let mut sum = 0f32;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        *k = (-(x * x) / (2.0 * sigma * sigma)).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    let mut tmp = vec![0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += img[y * size + sx] as f32 * k;
            }
            tmp[y * size + x] = acc;
        }
    }
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += tmp[sy * size + x] * k;
            }
            img[y * size + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn add_gaussian_noise(img: &mut [u8], sigma: f32, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let mut uniform = || -> f32 { (next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0) };
    for px in img.iter_mut() {
        let u1 = uniform();
        let u2 = uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let v = *px as f32 + z * sigma;
        *px = v.round().clamp(0.0, 255.0) as u8;
    }
}

// ---------------------------------------------------------------------------
// Per-condition stats bundle.

#[derive(Default)]
struct Bucket {
    errors: Vec<f32>,
    accepts: u32,
    total: u32,
    iters: u64,
    elapsed_ns: u128,
}

impl Bucket {
    fn push_err(&mut self, err: Option<f32>) {
        self.total += 1;
        if let Some(e) = err {
            self.errors.push(e);
            self.accepts += 1;
        }
    }
    fn add_time(&mut self, iters: u64, elapsed_ns: u128) {
        self.iters += iters;
        self.elapsed_ns += elapsed_ns;
    }
    fn stats(&self) -> Stats {
        let mut errs = self.errors.clone();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = if errs.is_empty() {
            0.0
        } else {
            errs.iter().sum::<f32>() / errs.len() as f32
        };
        let p95 = if errs.is_empty() {
            0.0
        } else {
            errs[((errs.len() as f32 * 0.95) as usize).min(errs.len() - 1)]
        };
        let worst = errs.last().copied().unwrap_or(0.0);
        let time_us = if self.iters == 0 {
            0.0
        } else {
            self.elapsed_ns as f64 / self.iters as f64 / 1000.0
        };
        Stats {
            mean,
            p95,
            worst,
            accept: self.accepts,
            total: self.total,
            errors: errs,
            time_us_per_call: time_us,
        }
    }
}

struct Stats {
    mean: f32,
    p95: f32,
    worst: f32,
    accept: u32,
    total: u32,
    errors: Vec<f32>,
    time_us_per_call: f64,
}

// ---------------------------------------------------------------------------
// Refiner harness.

struct ClassicRefiners {
    center: CenterOfMassRefiner,
    forstner: ForstnerRefiner,
    saddle: SaddlePointRefiner,
    radon: RadonPeakRefiner,
}

impl ClassicRefiners {
    fn new() -> Self {
        Self {
            center: CenterOfMassRefiner::new(CenterOfMassConfig::default()),
            forstner: ForstnerRefiner::new(ForstnerConfig::default()),
            saddle: SaddlePointRefiner::new(SaddlePointConfig::default()),
            radon: RadonPeakRefiner::new(RadonPeakConfig::default()),
        }
    }
}

fn bench_refiner<R: CornerRefiner>(
    refiner: &mut R,
    iters: u64,
    view: ImageView<'_>,
    response: Option<&ResponseMap>,
    seed: [f32; 2],
    truth: (f32, f32),
    bucket: &mut Bucket,
) {
    let ctx = RefineContext {
        image: Some(view),
        response,
    };
    let _ = refiner.refine(seed, ctx);
    let first = refiner.refine(seed, ctx);
    let err = if first.status == RefineStatus::Accepted {
        let dx = first.x - truth.0;
        let dy = first.y - truth.1;
        Some((dx * dx + dy * dy).sqrt())
    } else {
        None
    };
    bucket.push_err(err);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = refiner.refine(seed, ctx);
    }
    bucket.add_time(iters, start.elapsed().as_nanos());
}

// ---------------------------------------------------------------------------
// ML refiner wrapper (feature-gated).

#[cfg(feature = "ml-refiner")]
mod ml {
    use super::*;
    use chess_corners_ml::{MlModel, ModelSource};

    pub struct MlRefiner {
        pub model: MlModel,
        patch_size: usize,
        buffer: Vec<f32>,
    }

    impl MlRefiner {
        pub fn load() -> Option<Self> {
            let model = MlModel::load(ModelSource::EmbeddedDefault).ok()?;
            let patch_size = model.patch_size();
            let buffer = vec![0.0f32; patch_size * patch_size];
            Some(Self {
                model,
                patch_size,
                buffer,
            })
        }
        fn extract(&mut self, view: ImageView<'_>, x: f32, y: f32) -> Option<()> {
            let ps = self.patch_size;
            let half = (ps as f32 - 1.0) * 0.5;
            let (w, h) = (view.width as f32, view.height as f32);
            if x - half < 0.0 || y - half < 0.0 || x + half > w - 1.0 || y + half > h - 1.0 {
                return None;
            }
            for iy in 0..ps {
                let gy = y + iy as f32 - half;
                for ix in 0..ps {
                    let gx = x + ix as f32 - half;
                    self.buffer[iy * ps + ix] = view.sample_bilinear(gx, gy) / 255.0;
                }
            }
            Some(())
        }
        pub fn refine(&mut self, view: ImageView<'_>, seed: [f32; 2]) -> Option<[f32; 2]> {
            self.extract(view, seed[0], seed[1])?;
            let preds = self.model.infer_batch(&self.buffer, 1).ok()?;
            let pred = preds.first()?;
            Some([seed[0] + pred[0], seed[1] + pred[1]])
        }
    }

    pub fn bench(
        refiner: &mut MlRefiner,
        iters: u64,
        view: ImageView<'_>,
        seed: [f32; 2],
        truth: (f32, f32),
        bucket: &mut Bucket,
    ) {
        let _ = refiner.refine(view, seed);
        let first = refiner.refine(view, seed);
        let err = first.map(|r| {
            let dx = r[0] - truth.0;
            let dy = r[1] - truth.1;
            (dx * dx + dy * dy).sqrt()
        });
        bucket.push_err(err);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = refiner.refine(view, seed);
        }
        bucket.add_time(iters, start.elapsed().as_nanos());
    }
}

// ---------------------------------------------------------------------------
// Driver: sweep and emit JSON.

fn run_condition(
    blur_sigma: f32,
    noise_sigma: f32,
    cell: usize,
    seed_xor: u64,
) -> Vec<(&'static str, Stats)> {
    const SIZE: usize = 45;
    const CENTER: f32 = 22.0;
    const N: usize = 6; // 6×6 offsets per condition
    const ITERS_FAST: u64 = 200;
    const ITERS_SLOW: u64 = 20;

    let mut classic = ClassicRefiners::new();
    let mut b_center = Bucket::default();
    let mut b_forstner = Bucket::default();
    let mut b_saddle = Bucket::default();
    let mut b_radon = Bucket::default();
    #[cfg(feature = "ml-refiner")]
    let mut ml_refiner = ml::MlRefiner::load();
    #[cfg(feature = "ml-refiner")]
    let mut b_ml = Bucket::default();

    let params = ChessParams::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let mut img = synthetic_chessboard_aa(SIZE, cell, (ox, oy), 30, 230);
            gaussian_blur(&mut img, SIZE, blur_sigma);
            add_gaussian_noise(&mut img, noise_sigma, seed_xor ^ ((kx * N + ky) as u64));
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];
            let response = chess_response_u8(&img, SIZE, SIZE, &params);

            bench_refiner(
                &mut classic.center,
                ITERS_FAST,
                view,
                Some(&response),
                seed,
                (ox, oy),
                &mut b_center,
            );
            bench_refiner(
                &mut classic.forstner,
                ITERS_FAST,
                view,
                None,
                seed,
                (ox, oy),
                &mut b_forstner,
            );
            bench_refiner(
                &mut classic.saddle,
                ITERS_FAST,
                view,
                None,
                seed,
                (ox, oy),
                &mut b_saddle,
            );
            bench_refiner(
                &mut classic.radon,
                ITERS_FAST,
                view,
                None,
                seed,
                (ox, oy),
                &mut b_radon,
            );

            #[cfg(feature = "ml-refiner")]
            if let Some(r) = ml_refiner.as_mut() {
                ml::bench(r, ITERS_SLOW, view, seed, (ox, oy), &mut b_ml);
            }
        }
    }

    let mut rows = vec![
        ("CenterOfMass", b_center.stats()),
        ("Forstner", b_forstner.stats()),
        ("SaddlePoint", b_saddle.stats()),
        ("RadonPeak", b_radon.stats()),
    ];
    #[cfg(feature = "ml-refiner")]
    if ml_refiner.is_some() {
        rows.push(("ML", b_ml.stats()));
    }
    let _ = ITERS_SLOW;
    rows
}

fn emit_json_row(
    out: &mut String,
    refiner: &str,
    cell: usize,
    blur_sigma: f32,
    noise_sigma: f32,
    s: &Stats,
    first: bool,
) {
    if !first {
        out.push_str(",\n");
    }
    out.push_str(&format!(
        r#"    {{"refiner":"{}","cell":{},"blur_sigma":{:.3},"noise_sigma":{:.3},"mean":{:.6},"p95":{:.6},"worst":{:.6},"accept":{},"total":{},"time_us_per_call":{:.3},"errors":["#,
        refiner,
        cell,
        blur_sigma,
        noise_sigma,
        s.mean,
        s.p95,
        s.worst,
        s.accept,
        s.total,
        s.time_us_per_call,
    ));
    for (i, e) in s.errors.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:.6}", e));
    }
    out.push_str("]}");
}

fn main() {
    // Sweep grids. Kept narrow enough to run in ~1 min with ML enabled.
    let blur_values: &[f32] = &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
    let noise_values: &[f32] = &[0.0, 1.0, 2.0, 5.0, 8.0, 12.0];
    let cell_values: &[usize] = &[5, 6, 7, 8, 10];

    eprintln!("bench_sweep: blur×noise+cell conditions will be emitted as JSON");
    let mut json = String::from("{\n  \"schema\": \"bench_sweep/v1\",\n  \"results\": [\n");
    let mut first = true;

    // Blur sweep at noise=0, cell=8
    for &blur in blur_values {
        let rows = run_condition(blur, 0.0, 8, 0);
        for (r, s) in &rows {
            emit_json_row(&mut json, r, 8, blur, 0.0, s, first);
            first = false;
        }
        eprintln!("  blur={:.1}: done", blur);
    }
    // Noise sweep at blur=0.7, cell=8 (light blur to avoid pure aliased step)
    for &noise in noise_values {
        let rows = run_condition(0.7, noise, 8, 0xC0FFEE);
        for (r, s) in &rows {
            emit_json_row(&mut json, r, 8, 0.7, noise, s, first);
            first = false;
        }
        eprintln!("  noise={:.1}: done", noise);
    }
    // Cell size sweep at clean, cell varying
    for &cell in cell_values {
        let rows = run_condition(0.0, 0.0, cell, 0);
        for (r, s) in &rows {
            emit_json_row(&mut json, r, cell, 0.0, 0.0, s, first);
            first = false;
        }
        eprintln!("  cell={}: done", cell);
    }

    json.push_str("\n  ]\n}\n");
    print!("{}", json);
    eprintln!("bench_sweep: wrote {} bytes of JSON", json.len());
}
