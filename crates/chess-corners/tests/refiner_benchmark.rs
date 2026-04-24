//! Unified cross-refiner benchmark: accuracy + throughput.
//!
//! Sweeps sub-pixel offsets (and optionally noise/blur) on an
//! anti-aliased synthetic chessboard and compares all built-in
//! refiners plus the embedded ML model (when the `ml-refiner` feature
//! is enabled). Prints a single table per condition so the full picture
//! fits in one glance.
//!
//! Run with:
//!
//! ```sh
//! cargo test -p chess-corners --test refiner_benchmark \
//!     --all-features -- --nocapture
//! ```
//!
//! Without `--all-features` the ML column is omitted.

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
// Test fixture: anti-aliased chessboard with mild camera blur.

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
// Accuracy / timing aggregators.

/// Accuracy samples are collected once per sub-pixel offset; timing
/// samples over many iterations per offset. They are tracked with
/// independent counters so accept-rate and throughput are both
/// meaningful.
#[derive(Clone, Copy, Default)]
struct Stats {
    sum_err: f64,
    worst: f32,
    accepts: u32,
    offsets: u32,
    iters: u64,
    elapsed_ns: u128,
}

impl Stats {
    fn add_accuracy(&mut self, err: Option<f32>) {
        self.offsets += 1;
        if let Some(e) = err {
            self.sum_err += e as f64;
            self.worst = self.worst.max(e);
            self.accepts += 1;
        }
    }
    fn add_timing(&mut self, iters: u64, elapsed_ns: u128) {
        self.iters += iters;
        self.elapsed_ns += elapsed_ns;
    }
    fn mean_err(&self) -> f64 {
        if self.accepts == 0 {
            f64::NAN
        } else {
            self.sum_err / self.accepts as f64
        }
    }
    fn mean_us(&self) -> f64 {
        if self.iters == 0 {
            f64::NAN
        } else {
            self.elapsed_ns as f64 / self.iters as f64 / 1000.0
        }
    }
}

struct RefinerRow {
    name: &'static str,
    stats: Stats,
}

fn format_row(r: &RefinerRow) -> String {
    format!(
        "  {:<14} mean={:.4}  worst={:.4}  ok={:>3}/{:<3}  time={:7.2}µs",
        r.name,
        r.stats.mean_err(),
        r.stats.worst,
        r.stats.accepts,
        r.stats.offsets,
        r.stats.mean_us(),
    )
}

// ---------------------------------------------------------------------------
// Classic-refiner benchmark harness.

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
    stats: &mut Stats,
) {
    let ctx = RefineContext {
        image: Some(view),
        response,
    };

    // Warm-up (touch allocations / caches before timing).
    let _ = refiner.refine(seed, ctx);

    // Deterministic refiners: one accuracy sample per offset.
    let first = refiner.refine(seed, ctx);
    let err = if first.status == RefineStatus::Accepted {
        let dx = first.x - truth.0;
        let dy = first.y - truth.1;
        Some((dx * dx + dy * dy).sqrt())
    } else {
        None
    };
    stats.add_accuracy(err);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = refiner.refine(seed, ctx);
    }
    stats.add_timing(iters, start.elapsed().as_nanos());
}

// ---------------------------------------------------------------------------
// ML refiner harness (feature-gated).

#[cfg(feature = "ml-refiner")]
mod ml {
    use super::*;
    use chess_corners_ml::{MlModel, ModelSource};

    pub struct MlRefiner {
        pub model: MlModel,
        pub patch_size: usize,
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

        /// Extract a `patch_size × patch_size` bilinear-sampled patch
        /// centered at `(x, y)` into the internal buffer, returning
        /// `None` if the patch runs off the image.
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
        stats: &mut Stats,
    ) {
        let _ = refiner.refine(view, seed);

        let first = refiner.refine(view, seed);
        let err = first.map(|r| {
            let dx = r[0] - truth.0;
            let dy = r[1] - truth.1;
            (dx * dx + dy * dy).sqrt()
        });
        stats.add_accuracy(err);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = refiner.refine(view, seed);
        }
        stats.add_timing(iters, start.elapsed().as_nanos());
    }
}

// ---------------------------------------------------------------------------
// Sweep drivers.

/// Runs one pass across a grid of subpixel offsets. For each offset,
/// every refiner is timed by repeating the refinement `iters` times.
/// Accuracy is sampled once per offset (deterministic), so the
/// accept-rate is `accepts / offsets`.
fn run_sweep(label: &str, blur_sigma: f32, noise_sigma: f32, cell: usize, noise_seed_xor: u64) {
    const SIZE: usize = 45;
    const CENTER: f32 = 22.0;
    const N: usize = 6; // 6x6 = 36 offsets
    const ITERS: u64 = 400;

    let mut classic = ClassicRefiners::new();
    let mut center_stats = Stats::default();
    let mut forstner_stats = Stats::default();
    let mut saddle_stats = Stats::default();
    let mut radon_stats = Stats::default();
    #[cfg(feature = "ml-refiner")]
    let mut ml_refiner = ml::MlRefiner::load();
    #[cfg(feature = "ml-refiner")]
    let mut ml_stats = Stats::default();

    let params = ChessParams::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let mut img = synthetic_chessboard_aa(SIZE, cell, (ox, oy), 30, 230);
            gaussian_blur(&mut img, SIZE, blur_sigma);
            add_gaussian_noise(&mut img, noise_sigma, noise_seed_xor ^ (kx * N + ky) as u64);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];

            let response = chess_response_u8(&img, SIZE, SIZE, &params);

            bench_refiner(
                &mut classic.center,
                ITERS,
                view,
                Some(&response),
                seed,
                (ox, oy),
                &mut center_stats,
            );
            bench_refiner(
                &mut classic.forstner,
                ITERS,
                view,
                None,
                seed,
                (ox, oy),
                &mut forstner_stats,
            );
            bench_refiner(
                &mut classic.saddle,
                ITERS,
                view,
                None,
                seed,
                (ox, oy),
                &mut saddle_stats,
            );
            bench_refiner(
                &mut classic.radon,
                ITERS,
                view,
                None,
                seed,
                (ox, oy),
                &mut radon_stats,
            );

            #[cfg(feature = "ml-refiner")]
            if let Some(r) = ml_refiner.as_mut() {
                ml::bench(r, ITERS, view, seed, (ox, oy), &mut ml_stats);
            }
        }
    }

    let offsets = (N * N) as u32;
    eprintln!("=== {label} — cell={cell}px, {offsets} offsets × {ITERS} timed iters ===");
    let rows = [
        RefinerRow {
            name: "CenterOfMass",
            stats: center_stats,
        },
        RefinerRow {
            name: "Forstner",
            stats: forstner_stats,
        },
        RefinerRow {
            name: "SaddlePoint",
            stats: saddle_stats,
        },
        RefinerRow {
            name: "RadonPeak",
            stats: radon_stats,
        },
    ];
    for r in &rows {
        eprintln!("{}", format_row(r));
    }
    #[cfg(feature = "ml-refiner")]
    if ml_refiner.is_some() {
        let ml_row = RefinerRow {
            name: "ML (ONNX)",
            stats: ml_stats,
        };
        eprintln!("{}", format_row(&ml_row));
    }

    // Floor assertions that guard against regressions in the code we
    // actively tune here (RadonPeak). Other refiners are printed for
    // context but not asserted — their numbers vary more with the
    // fixture and the bench is primarily informational. At cell=5 the
    // default RadonPeak ray integrates across most of a cell and is a
    // little less accurate; the floor is relaxed there.
    if blur_sigma == 0.0 && noise_sigma == 0.0 {
        let floor = if cell >= 8 { 0.05 } else { 0.15 };
        assert!(
            radon_stats.mean_err() < floor,
            "RadonPeak clean mean {} >= {} (cell={cell})",
            radon_stats.mean_err(),
            floor,
        );
        assert_eq!(
            radon_stats.accepts, offsets,
            "RadonPeak dropped candidates on clean input"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests. Each sweep runs an independent condition.

// Cell sizes:
//   - 8 px: matches the default RadonPeak ray_radius=2×image_upsample=2
//     (4 physical px integration), a comfortable "large cell" regime.
//   - 5 px: closer to what the ML refiner was trained on (scale-1
//     corners on 21×21 patches). Included for a fair ML comparison.

#[test]
fn sweep_clean() {
    run_sweep("CLEAN (cell=8)", 0.0, 0.0, 8, 0);
}

#[test]
fn sweep_clean_small_cell() {
    run_sweep("CLEAN (cell=5)", 0.0, 0.0, 5, 0);
}

#[test]
fn sweep_blur_1_5() {
    run_sweep("BLUR σ=1.5", 1.5, 0.0, 8, 0);
}

#[test]
fn sweep_noise_5() {
    run_sweep("NOISE σ=5", 0.7, 5.0, 8, 0xC0FFEE);
}

#[test]
fn sweep_noise_10() {
    run_sweep("NOISE σ=10", 0.7, 10.0, 8, 0xBADF00D);
}
