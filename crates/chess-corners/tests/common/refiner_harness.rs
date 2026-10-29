//! Shared refiner benchmark harness for the `chess-corners` facade.
//!
//! Both `tests/refiner_benchmark.rs` (the accuracy/throughput table) and
//! `examples/bench_sweep.rs` (the JSON sweep behind the book plots) drive the
//! same built-in refiners over a grid of subpixel offsets. They differ
//! only in how they *aggregate* the per-sample results, so the refiner
//! plumbing lives here and each driver provides its own aggregator through
//! the [`RefineSampleSink`] trait.
//!
//! The file is included via `#[path = "common/refiner_harness.rs"]` from the
//! test and `#[path = "../tests/common/refiner_harness.rs"]` from the
//! example (a `benches/common/`-style shared module, kept out of `tests/`'s
//! auto-discovered target list by living in a subdirectory).

use std::time::Instant;

use chess_corners_core::{
    CenterOfMassConfig, CenterOfMassRefiner, CornerRefiner, ImageView, RefineContext, RefineStatus,
    ResponseMap, SaddlePointConfig, SaddlePointRefiner,
};

/// Sink for the two per-refiner measurements a driver records: one accuracy
/// sample per subpixel offset (`None` when the refiner rejected the seed) and
/// the cumulative wall-clock cost of `iters` timed refinements. Implemented by
/// each driver's own aggregator (the accuracy-table `Stats`, the JSON
/// `Bucket`) so [`bench_refiner`] stays aggregator-agnostic.
pub trait RefineSampleSink {
    /// Record one accuracy sample (`Some(err_px)` if accepted, else `None`).
    fn record_accuracy(&mut self, err: Option<f32>);
    /// Record the elapsed nanoseconds spent on `iters` timed refinements.
    fn record_timing(&mut self, iters: u64, elapsed_ns: u128);
}

/// The built-in refiners, constructed once at their default
/// configs and reused across the whole sweep so the per-refiner numbers are
/// directly comparable.
pub struct BuiltinRefiners {
    pub center: CenterOfMassRefiner,
    pub saddle: SaddlePointRefiner,
}

impl BuiltinRefiners {
    pub fn new() -> Self {
        Self {
            center: CenterOfMassRefiner::new(CenterOfMassConfig::default()),
            saddle: SaddlePointRefiner::new(SaddlePointConfig::default()),
        }
    }
}

impl Default for BuiltinRefiners {
    fn default() -> Self {
        Self::new()
    }
}

/// Refine `seed` once for an accuracy sample, then `iters` times for a timing
/// sample, pushing both into `sink`. The accuracy refiners are deterministic,
/// so a single sample per offset is exact.
pub fn bench_refiner<R: CornerRefiner, S: RefineSampleSink>(
    refiner: &mut R,
    iters: u64,
    view: ImageView<'_>,
    response: Option<&ResponseMap>,
    seed: [f32; 2],
    truth: (f32, f32),
    sink: &mut S,
) {
    let ctx = RefineContext::new(Some(view), response);

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
    sink.record_accuracy(err);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = refiner.refine(seed, ctx);
    }
    sink.record_timing(iters, start.elapsed().as_nanos());
}
