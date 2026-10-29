//! Image-space refiner accuracy on synthetic chessboards.
//!
//! This integration test runs the `SaddlePoint` refiner on a grid of
//! subpixel offsets and prints a summary when run with `--nocapture`. It
//! enforces a floor assertion on the clean-data accuracy of the fit — a
//! guard against regressions in the fit math, not the full paper contract.
//!
//! Run with:
//!
//! ```sh
//! cargo test -p chess-corners-core --test refiner_accuracy -- --nocapture
//! ```

use chess_corners_core::{
    CornerRefiner, ImageView, RefineContext, RefineStatus, SaddlePointConfig, SaddlePointRefiner,
};
use chess_corners_testutil::aa_chessboard;

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

/// Contract test: on anti-aliased, clean input the saddle-point refiner
/// must achieve a good mean error over a full sub-pixel sweep. If this
/// ever regresses, the fit math in the refiner or the renderer needs
/// attention.
#[test]
fn clean_subpixel_sweep_mean_accuracy() {
    const SIZE: usize = 35;
    const CELL: usize = 6;
    const CENTER: f32 = 17.0;
    const N: usize = 8;

    let mut saddle = SaddlePointRefiner::new(SaddlePointConfig::default());

    let mut s_saddle = Stats::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let img = aa_chessboard(SIZE, CELL, (ox, oy), 30, 230);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];
            s_saddle.push(refine_err(&mut saddle, view, seed, (ox, oy)));
        }
    }

    eprintln!(
        "CLEAN SWEEP ({}×{} offsets)  saddle: mean={:.4} worst={:.4} ok={}/{}",
        N,
        N,
        s_saddle.mean(),
        s_saddle.worst,
        s_saddle.accepts,
        s_saddle.total,
    );
    // Floor: the re-centred SaddlePoint fit lands within a few 10⁻⁴ px on
    // these clean fixtures and should accept every candidate on the full
    // sweep.
    assert!(
        s_saddle.mean() < 0.01,
        "SaddlePoint clean mean {} >= 0.01",
        s_saddle.mean()
    );
    assert_eq!(
        s_saddle.accepts, s_saddle.total,
        "SaddlePoint dropped candidates"
    );
}
