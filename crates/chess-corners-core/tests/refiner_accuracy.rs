//! Cross-refiner accuracy benchmark on synthetic chessboards.
//!
//! This integration test compares the image-space refiners
//! (`Forstner`, `SaddlePoint`) on a grid of subpixel offsets and prints
//! a summary table when run with `--nocapture`. It also enforces a floor
//! assertion on the clean-data accuracy of the SaddlePoint fit — a guard
//! against regressions in the fit math, not the full paper contract.
//!
//! Run with:
//!
//! ```sh
//! cargo test -p chess-corners-core --test refiner_accuracy -- --nocapture
//! ```

use chess_corners_core::{
    CornerRefiner, ForstnerConfig, ForstnerRefiner, ImageView, RefineContext, RefineStatus,
    SaddlePointConfig, SaddlePointRefiner,
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

/// Contract test: on anti-aliased, clean input the image-space refiners
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
    let mut forstner = ForstnerRefiner::new(ForstnerConfig::default());

    let mut s_saddle = Stats::default();
    let mut s_forstner = Stats::default();

    for kx in 0..N {
        for ky in 0..N {
            let ox = CENTER + kx as f32 / N as f32;
            let oy = CENTER + ky as f32 / N as f32;
            let img = synthetic_chessboard_aa(SIZE, CELL, (ox, oy), 30, 230);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();
            let seed = [ox.round(), oy.round()];
            s_saddle.push(refine_err(&mut saddle, view, seed, (ox, oy)));
            s_forstner.push(refine_err(&mut forstner, view, seed, (ox, oy)));
        }
    }

    eprintln!(
        "CLEAN SWEEP ({}×{} offsets)  saddle: mean={:.4} worst={:.4} ok={}/{}   forstner: mean={:.4} worst={:.4} ok={}/{}",
        N, N,
        s_saddle.mean(), s_saddle.worst, s_saddle.accepts, s_saddle.total,
        s_forstner.mean(), s_forstner.worst, s_forstner.accepts, s_forstner.total,
    );
    // Floor: SaddlePoint's quadratic fit typically lands around 0.12 px
    // on small 35-px fixtures and should accept every candidate on the
    // full sweep.
    assert!(
        s_saddle.mean() < 0.2,
        "SaddlePoint clean mean {} >= 0.2",
        s_saddle.mean()
    );
    assert_eq!(
        s_saddle.accepts, s_saddle.total,
        "SaddlePoint dropped candidates"
    );
}
