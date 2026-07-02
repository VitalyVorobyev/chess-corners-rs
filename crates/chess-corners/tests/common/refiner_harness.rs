//! Shared classic-refiner benchmark harness for the `chess-corners` facade.
//!
//! Both `tests/refiner_benchmark.rs` (the accuracy/throughput table) and
//! `examples/bench_sweep.rs` (the JSON sweep behind the book plots) drive the
//! same three built-in refiners over a grid of subpixel offsets. They differ
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
    CenterOfMassConfig, CenterOfMassRefiner, CornerRefiner, ForstnerConfig, ForstnerRefiner,
    ImageView, RefineContext, RefineStatus, ResponseMap, SaddlePointConfig, SaddlePointRefiner,
};

/// Sink for the two per-refiner measurements a driver records: one accuracy
/// sample per subpixel offset (`None` when the refiner rejected the seed) and
/// the cumulative wall-clock cost of `iters` timed refinements. Implemented by
/// each driver's own aggregator (the accuracy-table `Stats`, the JSON
/// `Bucket`) so [`bench_refiner`] and [`ml::bench`] stay aggregator-agnostic.
pub trait RefineSampleSink {
    /// Record one accuracy sample (`Some(err_px)` if accepted, else `None`).
    fn record_accuracy(&mut self, err: Option<f32>);
    /// Record the elapsed nanoseconds spent on `iters` timed refinements.
    fn record_timing(&mut self, iters: u64, elapsed_ns: u128);
}

/// The three classic (non-ML) refiners, constructed once at their default
/// configs and reused across the whole sweep so the per-refiner numbers are
/// directly comparable.
pub struct ClassicRefiners {
    pub center: CenterOfMassRefiner,
    pub forstner: ForstnerRefiner,
    pub saddle: SaddlePointRefiner,
}

impl ClassicRefiners {
    pub fn new() -> Self {
        Self {
            center: CenterOfMassRefiner::new(CenterOfMassConfig::default()),
            forstner: ForstnerRefiner::new(ForstnerConfig::default()),
            saddle: SaddlePointRefiner::new(SaddlePointConfig::default()),
        }
    }
}

impl Default for ClassicRefiners {
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

// ---------------------------------------------------------------------------
// ML refiner harness (feature-gated).

#[cfg(feature = "ml-refiner")]
pub mod ml {
    use super::{ImageView, Instant, RefineSampleSink};
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

        /// Extract a `patch_size × patch_size` bilinear-sampled patch centered
        /// at `(x, y)` into the internal buffer, returning `None` if the patch
        /// runs off the image.
        fn extract(&mut self, view: ImageView<'_>, x: f32, y: f32) -> Option<()> {
            let ps = self.patch_size;
            let half = (ps as f32 - 1.0) * 0.5;
            let (w, h) = (view.width() as f32, view.height() as f32);
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

    pub fn bench<S: RefineSampleSink>(
        refiner: &mut MlRefiner,
        iters: u64,
        view: ImageView<'_>,
        seed: [f32; 2],
        truth: (f32, f32),
        sink: &mut S,
    ) {
        let _ = refiner.refine(view, seed);

        let first = refiner.refine(view, seed);
        let err = first.map(|r| {
            let dx = r[0] - truth.0;
            let dy = r[1] - truth.1;
            (dx * dx + dy * dy).sqrt()
        });
        sink.record_accuracy(err);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = refiner.refine(view, seed);
        }
        sink.record_timing(iters, start.elapsed().as_nanos());
    }
}
