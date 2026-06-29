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

use chess_corners::low_level::ImageView;
use chess_corners_core::chess_response_u8;
use chess_corners_core::unstable::ChessParams;
use chess_corners_testutil::{aa_chessboard, add_gaussian_noise, gaussian_blur};

#[path = "common/refiner_harness.rs"]
mod harness;
use harness::{bench_refiner, ClassicRefiners};

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
// Aggregator wiring: route the shared harness samples into this file's `Stats`.

impl harness::RefineSampleSink for Stats {
    fn record_accuracy(&mut self, err: Option<f32>) {
        self.add_accuracy(err);
    }
    fn record_timing(&mut self, iters: u64, elapsed_ns: u128) {
        self.add_timing(iters, elapsed_ns);
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
    #[cfg(feature = "ml-refiner")]
    let mut ml_refiner = harness::ml::MlRefiner::load();
    #[cfg(feature = "ml-refiner")]
    let mut ml_stats = Stats::default();

    let params = ChessParams::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let mut img = aa_chessboard(SIZE, cell, (ox, oy), 30, 230);
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

            #[cfg(feature = "ml-refiner")]
            if let Some(r) = ml_refiner.as_mut() {
                harness::ml::bench(r, ITERS, view, seed, (ox, oy), &mut ml_stats);
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

    // Floor assertion guarding against regressions in the SaddlePoint
    // fit math on clean input. Other refiners are printed for context
    // but not asserted — their numbers vary more with the fixture and
    // the bench is primarily informational.
    if blur_sigma == 0.0 && noise_sigma == 0.0 {
        assert!(
            saddle_stats.mean_err() < 0.2,
            "SaddlePoint clean mean {} >= 0.2 (cell={cell})",
            saddle_stats.mean_err(),
        );
        assert_eq!(
            saddle_stats.accepts, offsets,
            "SaddlePoint dropped candidates on clean input"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests. Each sweep runs an independent condition.

// Cell sizes:
//   - 8 px: a comfortable "large cell" regime.
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
