//! Recall / precision / p95-error guard against optimization
//! regressions in the ChESS and Radon detector hot paths.
//!
//! For each detector preset (`single_scale`, `multiscale`, `radon`),
//! the test:
//!
//! 1. Renders a deterministic anti-aliased chessboard with a known
//!    grid of interior corners.
//! 2. Runs `find_chess_corners_u8` with the preset.
//! 3. Greedily matches each detected corner to the nearest ground-truth
//!    corner within `MATCH_THRESHOLD_PX`.
//! 4. Asserts that recall, precision, and p95 subpixel error stay
//!    within the bounds defined by `Bounds`.
//!
//! These bounds are intentionally generous so SIMD / rayon / numerical
//! refactors don't trip the test on benign FP-order changes — the goal
//! is to catch *regressions*, not to replace the per-refiner accuracy
//! sweep in `refiner_benchmark.rs`.

use chess_corners::{find_chess_corners_u8, ChessConfig, CornerDescriptor};

const SUPER: usize = 8;
const MATCH_THRESHOLD_PX: f32 = 1.5;

/// Per-detector accuracy bounds. Picked from a clean run on the
/// scalar reference; bench results above this threshold mean an
/// optimization changed correctness, not just speed.
struct Bounds {
    min_recall: f32,
    min_precision: f32,
    max_p95_err: f32,
}

fn synthetic_chessboard(size: usize, cell: usize, offset: (f32, f32)) -> Vec<u8> {
    let (ox, oy) = offset;
    let c = cell as f32;
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
                    acc += if dark_cell { 30.0 } else { 230.0 };
                }
            }
            img[y * size + x] = (acc * inv_super2).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn ground_truth_corners(size: usize, cell: usize, offset: (f32, f32)) -> Vec<(f32, f32)> {
    let (ox, oy) = offset;
    let mut out = Vec::new();
    let mut k = 1;
    loop {
        let gy = oy + cell as f32 * k as f32;
        if gy >= size as f32 - 2.0 {
            break;
        }
        let mut j = 1;
        loop {
            let gx = ox + cell as f32 * j as f32;
            if gx >= size as f32 - 2.0 {
                break;
            }
            // Border ChESS / Radon won't see the outermost ring; clip
            // ground truth that's too close to the image edge so the
            // recall floor stays tight on the interior corners alone.
            if gx > 8.0 && gy > 8.0 && gx < size as f32 - 8.0 && gy < size as f32 - 8.0 {
                out.push((gx, gy));
            }
            j += 1;
        }
        k += 1;
    }
    out
}

struct MatchStats {
    recall: f32,
    precision: f32,
    p95_err: f32,
    tp: usize,
    fp: usize,
    fn_count: usize,
}

fn match_detections(detected: &[CornerDescriptor], gt: &[(f32, f32)]) -> MatchStats {
    let mut det_used = vec![false; detected.len()];
    let mut errs: Vec<f32> = Vec::new();
    let mut matched = 0usize;
    for (gx, gy) in gt.iter() {
        let mut best: Option<(usize, f32)> = None;
        for (i, d) in detected.iter().enumerate() {
            if det_used[i] {
                continue;
            }
            let dx = d.x - *gx;
            let dy = d.y - *gy;
            let err = (dx * dx + dy * dy).sqrt();
            if err <= MATCH_THRESHOLD_PX && best.map(|(_, e)| err < e).unwrap_or(true) {
                best = Some((i, err));
            }
        }
        if let Some((i, e)) = best {
            det_used[i] = true;
            errs.push(e);
            matched += 1;
        }
    }
    let tp = matched;
    let fp = det_used.iter().filter(|&&u| !u).count();
    let fn_count = gt.len() - matched;
    let recall = if gt.is_empty() {
        1.0
    } else {
        matched as f32 / gt.len() as f32
    };
    let precision = if detected.is_empty() {
        1.0
    } else {
        tp as f32 / detected.len() as f32
    };
    let p95_err = if errs.is_empty() {
        0.0
    } else {
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((errs.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
        errs[idx]
    };
    MatchStats {
        recall,
        precision,
        p95_err,
        tp,
        fp,
        fn_count,
    }
}

fn check(label: &str, cfg: &ChessConfig, img: &[u8], side: usize, gt: &[(f32, f32)], b: &Bounds) {
    let detected = find_chess_corners_u8(img, side as u32, side as u32, cfg);
    let stats = match_detections(&detected, gt);
    eprintln!(
        "[{label}] det={}, gt={}, tp={}, fp={}, fn={}, recall={:.3}, precision={:.3}, p95_err={:.3}px",
        detected.len(),
        gt.len(),
        stats.tp,
        stats.fp,
        stats.fn_count,
        stats.recall,
        stats.precision,
        stats.p95_err,
    );
    assert!(
        stats.recall >= b.min_recall,
        "[{label}] recall {:.3} < min {:.3}",
        stats.recall,
        b.min_recall,
    );
    assert!(
        stats.precision >= b.min_precision,
        "[{label}] precision {:.3} < min {:.3}",
        stats.precision,
        b.min_precision,
    );
    assert!(
        stats.p95_err <= b.max_p95_err,
        "[{label}] p95_err {:.3} > max {:.3}",
        stats.p95_err,
        b.max_p95_err,
    );
}

// Bounds are calibrated against the scalar (no-features) reference
// run. The single-scale ChESS detector picks up a handful of extra
// corners outside the masked-edge GT region, so its precision floor
// is set just below the observed value. The Radon detector emits
// many low-strength satellite peaks on an idealized synthetic
// chessboard (every cell-edge mid-point also produces a positive
// (max−min)² response) — the floor here is set just below the
// observed precision so an optimization that *worsens* precision
// fails the test, but the existing high-FP behaviour is permitted.
#[test]
fn chess_single_scale_meets_accuracy_floor() {
    const SIDE: usize = 256;
    const CELL: usize = 24;
    let off = (CELL as f32 / 2.0 + 0.31, CELL as f32 / 2.0 + 0.47);
    let img = synthetic_chessboard(SIDE, CELL, off);
    let gt = ground_truth_corners(SIDE, CELL, off);
    let cfg = ChessConfig::single_scale();
    let bounds = Bounds {
        min_recall: 0.98,
        min_precision: 0.78,
        max_p95_err: 0.20,
    };
    check("chess_single", &cfg, &img, SIDE, &gt, &bounds);
}

#[test]
fn chess_multiscale_meets_accuracy_floor() {
    const SIDE: usize = 256;
    const CELL: usize = 24;
    let off = (CELL as f32 / 2.0 + 0.31, CELL as f32 / 2.0 + 0.47);
    let img = synthetic_chessboard(SIDE, CELL, off);
    let gt = ground_truth_corners(SIDE, CELL, off);
    let cfg = ChessConfig::multiscale();
    let bounds = Bounds {
        min_recall: 0.85,
        min_precision: 0.75,
        max_p95_err: 0.30,
    };
    check("chess_multiscale", &cfg, &img, SIDE, &gt, &bounds);
}

#[test]
fn radon_meets_accuracy_floor() {
    const SIDE: usize = 256;
    const CELL: usize = 24;
    let off = (CELL as f32 / 2.0 + 0.31, CELL as f32 / 2.0 + 0.47);
    let img = synthetic_chessboard(SIDE, CELL, off);
    let gt = ground_truth_corners(SIDE, CELL, off);
    let cfg = ChessConfig::radon();
    let bounds = Bounds {
        min_recall: 0.95,
        min_precision: 0.07,
        max_p95_err: 0.20,
    };
    check("radon", &cfg, &img, SIDE, &gt, &bounds);
}
