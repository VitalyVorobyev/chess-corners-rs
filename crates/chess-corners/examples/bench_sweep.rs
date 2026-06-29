//! **Dev-only tool — excluded from the published package.**
//!
//! Dense refiner sweep — produces a JSON report consumed by the
//! book's Part V plots (`tools/book/plot_benchmark.py`).
//!
//! Sweeps sub-pixel offset × blur σ × noise σ × cell size against
//! every built-in refiner. Prints one JSON document to stdout:
//!
//! ```json
//! {
//!   "refiners": ["CenterOfMass", "Forstner", "SaddlePoint", "ML"],
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

use chess_corners::low_level::ImageView;
use chess_corners_core::chess_response_u8;
use chess_corners_core::unstable::ChessParams;
use chess_corners_testutil::{aa_chessboard, add_gaussian_noise, gaussian_blur};

#[path = "../tests/common/refiner_harness.rs"]
mod harness;
use harness::{bench_refiner, ClassicRefiners};

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
// Aggregator wiring: route the shared harness samples into this file's `Bucket`.

impl harness::RefineSampleSink for Bucket {
    fn record_accuracy(&mut self, err: Option<f32>) {
        self.push_err(err);
    }
    fn record_timing(&mut self, iters: u64, elapsed_ns: u128) {
        self.add_time(iters, elapsed_ns);
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
    #[cfg(feature = "ml-refiner")]
    let mut ml_refiner = harness::ml::MlRefiner::load();
    #[cfg(feature = "ml-refiner")]
    let mut b_ml = Bucket::default();

    let params = ChessParams::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let mut img = aa_chessboard(SIZE, cell, (ox, oy), 30, 230);
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

            #[cfg(feature = "ml-refiner")]
            if let Some(r) = ml_refiner.as_mut() {
                harness::ml::bench(r, ITERS_SLOW, view, seed, (ox, oy), &mut b_ml);
            }
        }
    }

    #[cfg_attr(not(feature = "ml-refiner"), allow(unused_mut))]
    let mut rows = vec![
        ("CenterOfMass", b_center.stats()),
        ("Forstner", b_forstner.stats()),
        ("SaddlePoint", b_saddle.stats()),
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
