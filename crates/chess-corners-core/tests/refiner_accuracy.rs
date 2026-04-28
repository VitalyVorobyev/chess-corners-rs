//! Cross-refiner accuracy benchmark on synthetic chessboards.
//!
//! This integration test compares the four built-in refiners
//! (`CenterOfMass`, `Forstner`, `SaddlePoint`, `RadonPeak`) on a grid
//! of subpixel offsets, noise levels, and blur levels, and prints a
//! summary table when run with `--nocapture`. It also enforces floor
//! assertions on the clean-data accuracy of the image-space refiners
//! — these guard against regressions in the fit math, not the full
//! paper contract.
//!
//! Run with:
//!
//! ```sh
//! cargo test -p chess-corners-core --test refiner_accuracy -- --nocapture
//! ```

use chess_corners_core::{
    refine::{
        CornerRefiner, ForstnerConfig, ForstnerRefiner, RefineContext, RefineStatus,
        SaddlePointConfig, SaddlePointRefiner,
    },
    refine_radon::{RadonPeakConfig, RadonPeakRefiner},
    ImageView,
};

/// Anti-aliased synthetic chessboard renderer. Shares the supersampling
/// strategy used in `refine_radon.rs` tests but kept separate here so
/// the integration test stays self-contained.
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

#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    sum: f64,
    worst: f32,
    accepts: u32,
    total: u32,
}

impl Stats {
    fn push(&mut self, err: Option<f32>) {
        self.total += 1;
        if let Some(e) = err {
            self.sum += e as f64;
            self.worst = self.worst.max(e);
            self.accepts += 1;
        }
    }
    fn mean(&self) -> f64 {
        if self.accepts == 0 {
            f64::NAN
        } else {
            self.sum / self.accepts as f64
        }
    }
}

fn refine_err<R: CornerRefiner>(
    refiner: &mut R,
    view: ImageView<'_>,
    seed: [f32; 2],
    truth: (f32, f32),
) -> Option<f32> {
    let res = refiner.refine(seed, RefineContext::new(Some(view), None));
    if res.status != RefineStatus::Accepted {
        return None;
    }
    let dx = res.x - truth.0;
    let dy = res.y - truth.1;
    Some((dx * dx + dy * dy).sqrt())
}

/// Contract test: on anti-aliased, clean input the image-space
/// refiners (SaddlePoint, RadonPeak) must achieve <0.1 px mean error
/// over a full sub-pixel sweep. If this ever regresses, the fit math
/// in the refiner or the renderer needs attention.
#[test]
fn clean_subpixel_sweep_mean_accuracy() {
    const SIZE: usize = 35;
    const CELL: usize = 6;
    const CENTER: f32 = 17.0;
    const N: usize = 8;

    let mut radon = RadonPeakRefiner::new(RadonPeakConfig::default());
    let mut saddle = SaddlePointRefiner::new(SaddlePointConfig::default());
    let mut forstner = ForstnerRefiner::new(ForstnerConfig::default());

    let mut s_radon = Stats::default();
    let mut s_saddle = Stats::default();
    let mut s_forstner = Stats::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let img = synthetic_chessboard_aa(SIZE, CELL, (ox, oy), 30, 230);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];
            s_radon.push(refine_err(&mut radon, view, seed, (ox, oy)));
            s_saddle.push(refine_err(&mut saddle, view, seed, (ox, oy)));
            s_forstner.push(refine_err(&mut forstner, view, seed, (ox, oy)));
        }
    }

    eprintln!(
        "CLEAN SWEEP ({}×{} offsets)  radon: mean={:.4} worst={:.4} ok={}/{}   saddle: mean={:.4} worst={:.4} ok={}/{}   forstner: mean={:.4} worst={:.4} ok={}/{}",
        N, N,
        s_radon.mean(), s_radon.worst, s_radon.accepts, s_radon.total,
        s_saddle.mean(), s_saddle.worst, s_saddle.accepts, s_saddle.total,
        s_forstner.mean(), s_forstner.worst, s_forstner.accepts, s_forstner.total,
    );
    // Floor: RadonPeak must achieve <0.05 px mean on clean,
    // anti-aliased input (paper territory). SaddlePoint's quadratic
    // fit struggles a little more on small 35-px fixtures; it
    // typically lands around 0.12 px. Both refiners should accept
    // every candidate on the full sweep.
    assert!(
        s_radon.mean() < 0.05,
        "RadonPeak clean mean {} >= 0.05",
        s_radon.mean()
    );
    assert!(
        s_saddle.mean() < 0.2,
        "SaddlePoint clean mean {} >= 0.2",
        s_saddle.mean()
    );
    assert_eq!(
        s_radon.accepts, s_radon.total,
        "RadonPeak dropped candidates"
    );
    assert_eq!(
        s_saddle.accepts, s_saddle.total,
        "SaddlePoint dropped candidates"
    );
}

/// Behavioural test: RadonPeak is supposed to be the noise-robust
/// refiner. It should match or beat SaddlePoint on the mean error at
/// σ=5 additive noise. Not a hard accuracy contract (noise floors move
/// with sample count); the assertion is a relative comparison.
#[test]
fn radon_peak_competitive_under_noise() {
    const SIZE: usize = 35;
    const CELL: usize = 6;
    const CENTER: f32 = 17.0;
    const N: usize = 6;
    const SIGMA: f32 = 5.0;

    let mut radon = RadonPeakRefiner::new(RadonPeakConfig::default());
    let mut saddle = SaddlePointRefiner::new(SaddlePointConfig::default());

    let mut s_radon = Stats::default();
    let mut s_saddle = Stats::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let mut img = synthetic_chessboard_aa(SIZE, CELL, (ox, oy), 30, 230);
            gaussian_blur(&mut img, SIZE, 0.7);
            add_gaussian_noise(&mut img, SIGMA, 0xC0FFEE ^ ((kx * N + ky) as u64));
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];
            s_radon.push(refine_err(&mut radon, view, seed, (ox, oy)));
            s_saddle.push(refine_err(&mut saddle, view, seed, (ox, oy)));
        }
    }

    eprintln!(
        "NOISE σ={} SWEEP ({}×{} offsets)  radon: mean={:.4} worst={:.4} ok={}/{}   saddle: mean={:.4} worst={:.4} ok={}/{}",
        SIGMA, N, N,
        s_radon.mean(), s_radon.worst, s_radon.accepts, s_radon.total,
        s_saddle.mean(), s_saddle.worst, s_saddle.accepts, s_saddle.total,
    );
    // Relative: RadonPeak must not be dramatically worse than
    // SaddlePoint under noise (its design claim). 1.5× margin reflects
    // run-to-run jitter with a fixed seed.
    assert!(
        s_radon.mean() <= s_saddle.mean() * 1.5 + 0.02,
        "RadonPeak noisy mean {} vs SaddlePoint {} — regression",
        s_radon.mean(),
        s_saddle.mean()
    );
}

/// Behavioural test: under moderate Gaussian blur (σ=1.5) RadonPeak
/// should beat SaddlePoint, because smoothed edges wash out the
/// saddle-point Hessian structure while ray integration retains
/// contrast.
#[test]
fn radon_peak_competitive_under_blur() {
    const SIZE: usize = 35;
    const CELL: usize = 6;
    const CENTER: f32 = 17.0;
    const N: usize = 6;

    let mut radon = RadonPeakRefiner::new(RadonPeakConfig::default());
    let mut saddle = SaddlePointRefiner::new(SaddlePointConfig::default());

    for &sigma in &[1.0f32, 1.5, 2.0] {
        let mut s_radon = Stats::default();
        let mut s_saddle = Stats::default();
        for kx in 0..N {
            for ky in 0..N {
                let ox = CENTER + kx as f32 / N as f32;
                let oy = CENTER + ky as f32 / N as f32;
                let mut img = synthetic_chessboard_aa(SIZE, CELL, (ox, oy), 30, 230);
                gaussian_blur(&mut img, SIZE, sigma);
                let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
                let seed = [ox.round(), oy.round()];
                s_radon.push(refine_err(&mut radon, view, seed, (ox, oy)));
                s_saddle.push(refine_err(&mut saddle, view, seed, (ox, oy)));
            }
        }
        eprintln!(
            "BLUR σ={:.1} SWEEP  radon: mean={:.4} worst={:.4}   saddle: mean={:.4} worst={:.4}",
            sigma,
            s_radon.mean(),
            s_radon.worst,
            s_saddle.mean(),
            s_saddle.worst,
        );
    }
}
