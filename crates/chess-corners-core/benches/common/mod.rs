//! Shared synthetic-image helpers for the `chess-corners-core` benches.
//!
//! Lives in a `benches/common/` subdirectory (not a top-level
//! `benches/*.rs`) so Cargo does not treat it as its own bench target;
//! each bench binary pulls it in with `mod common;`. Centralising the
//! generators keeps every core bench on the exact same deterministic
//! boards (SOLID-01: one synthetic-board helper rather than a per-bench
//! copy).
//!
//! Three board flavours share the same cell geometry and intensity
//! levels, so the same corner coordinate lands on a real 4-quadrant
//! crossing in each:
//!
//! - [`synth_chessboard`] — hard 40/215 steps. The tanh ring/disk model
//!   cannot represent an instantaneous step, so even a perfect corner
//!   fits with a large relative residual (`rel_rms ≈ 0.47`). This is the
//!   worst case for the orientation fits.
//! - [`synth_chessboard_soft`] — anti-aliased (8× supersampled + mild
//!   box blur). The graded edge matches the tanh model, so a clean
//!   corner has a *low* `rel_rms`, exercising RingFit's fast 2nd-harmonic
//!   seed path and DiskFit's lazy-gate short-circuit.
//! - [`synth_chessboard_warped`] — the soft board under a fixed affine
//!   skew so the corner's axis separation is far from 90°. This is the
//!   regime DiskFit's full disk estimator targets.

// This module is `mod common;`-included into every bench binary, but
// each bench uses only the subset of generators it needs. Allow
// dead_code so the unused-here generators don't trip `-D warnings` in
// the response/refiner benches that only want `synth_chessboard`.
#![allow(dead_code)]

/// Dark-cell intensity (gray levels) shared by every generated board.
const DARK: u8 = 40;
/// Bright-cell intensity (gray levels) shared by every generated board.
const BRIGHT: u8 = 215;

/// Supersampling factor for the anti-aliased generators. Each output
/// pixel averages a `SUPER × SUPER` grid of hard-pattern sub-samples,
/// softening every step into a graded edge.
const SUPER: usize = 8;

/// Box-blur passes applied to the anti-aliased boards after
/// supersampling. Supersampling alone leaves a ~1 px edge, which is
/// *sharper* than the ring model's effective edge width (`r/β = 1.25 px`
/// at radius 5, i.e. a 10–90% rise of ~2.7 px). Two 3×3 box passes widen
/// the data edge into that sweet spot, dropping the clean-corner
/// `rel_rms` to ~0.01 — well below DiskFit's 0.04 lazy gate and
/// RingFit's 0.12 robust trigger — so the soft fixture reliably reaches
/// the fast/lazy paths. Empirically chosen (PERF-12 probe).
const SOFT_BLUR_PASSES: u32 = 2;

/// Affine shear applied by [`synth_chessboard_warped`]. The vertical
/// edge family is mapped to direction `(SHEAR, 1)`; at `SHEAR = 0.6` the
/// corner's two axes sit at 0° and `atan2(1, 0.6) ≈ 59°`, an axis
/// separation ~59° (measured ~61° after the soft blur) — well outside
/// DiskFit's [70°, 90°] lazy-gate band, so the full disk estimator runs.
/// The margin is kept generous so the full-disk path stays reachable
/// independent of the exact [`SOFT_BLUR_PASSES`] setting.
const WARP_SHEAR: f32 = 0.6;

/// Image side and corner used by the single-corner orientation
/// fixtures. The soft and warped boards are calibrated so this corner
/// lands on a real 4-quadrant crossing; the warp's fixed row is
/// `ORIENT_CORNER.1`, keeping the corner stationary while the axes skew.
pub const ORIENT_DIM: usize = 256;

/// Benchmark corner coordinate (a 90° checkerboard crossing on the hard
/// and soft boards, a ~63° crossing on the warped board).
pub const ORIENT_CORNER: (f32, f32) = (130.0, 130.0);

/// Cell size in pixels: ~25 cells across the shorter image dimension
/// (floor 8), matching a typical calibration target in frame. Shared by
/// every generator so corner coordinates coincide across boards.
#[inline]
fn cell_size(w: usize, h: usize) -> i32 {
    (h.min(w) / 25).max(8) as i32
}

/// Hard checkerboard cell parity: `true` selects the dark level.
#[inline]
fn is_dark_cell(cx: i32, cy: i32) -> bool {
    (cx + cy).rem_euclid(2) == 0
}

/// Deterministic hard-edge chessboard image in row-major 8-bit
/// grayscale.
///
/// Corner density is roughly 25 cells across the shorter image
/// dimension, which matches a typical calibration target in frame. The
/// pattern is fully deterministic so response timings are comparable
/// across machines and across feature builds.
pub fn synth_chessboard(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = (x as i32) / cell;
            let cy = (y as i32) / cell;
            out[y * w + x] = if is_dark_cell(cx, cy) { DARK } else { BRIGHT };
        }
    }
    out
}

/// Soft-edge (anti-aliased) chessboard.
///
/// Same geometry and levels as [`synth_chessboard`], but each pixel is
/// the average of an `SUPER × SUPER` grid of hard-pattern sub-samples
/// followed by [`SOFT_BLUR_PASSES`] 3×3 box passes. The resulting graded
/// edge matches the tanh intensity model, so a clean corner fits with a
/// small relative residual.
pub fn synth_chessboard_soft(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    render_soft(w, h, cell, SOFT_BLUR_PASSES, |x, y| (x, y))
}

/// Soft-edge chessboard under a fixed affine skew.
///
/// Built like [`synth_chessboard_soft`] but each sub-sample's lookup
/// coordinate is run through the inverse of a horizontal shear pinned to
/// the row `ORIENT_CORNER.1`. The corner at [`ORIENT_CORNER`] stays put
/// while its axes skew to a ~63° separation, the projective-skew regime
/// the full-disk estimator targets. Calibrated for the [`ORIENT_DIM`]
/// orientation fixtures.
pub fn synth_chessboard_warped(w: usize, h: usize) -> Vec<u8> {
    let cell = cell_size(w, h);
    let py = ORIENT_CORNER.1;
    // Forward edges: (1,0) -> (1,0) and (0,1) -> (SHEAR,1). The inverse
    // map used to resample the source pattern is x' = x - SHEAR·(y - py),
    // y' = y; the row y = py is fixed, so the corner does not move.
    render_soft(w, h, cell, SOFT_BLUR_PASSES, move |x, y| {
        (x - WARP_SHEAR * (y - py), y)
    })
}

/// Render an anti-aliased chessboard. `map` transforms an output pixel
/// coordinate into the board coordinate looked up against the hard cell
/// parity: the identity yields an axis-aligned soft board, an inverse
/// shear yields a skewed one. Anti-aliasing comes from `SUPER × SUPER`
/// box supersampling; `blur_passes` 3×3 box passes then widen the edge
/// toward the ring/disk model. Fully deterministic — no RNG.
fn render_soft<F>(w: usize, h: usize, cell: i32, blur_passes: u32, map: F) -> Vec<u8>
where
    F: Fn(f32, f32) -> (f32, f32),
{
    let inv = 1.0 / (SUPER * SUPER) as f32;
    let cellf = cell as f32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for sy in 0..SUPER {
                let yf = y as f32 + (sy as f32 + 0.5) / SUPER as f32 - 0.5;
                for sx in 0..SUPER {
                    let xf = x as f32 + (sx as f32 + 0.5) / SUPER as f32 - 0.5;
                    let (bx, by) = map(xf, yf);
                    let cx = (bx / cellf).floor() as i32;
                    let cy = (by / cellf).floor() as i32;
                    acc += if is_dark_cell(cx, cy) {
                        DARK as f32
                    } else {
                        BRIGHT as f32
                    };
                }
            }
            out[y * w + x] = (acc * inv).round().clamp(0.0, 255.0) as u8;
        }
    }
    for _ in 0..blur_passes {
        out = box_blur_3x3(&out, w, h);
    }
    out
}

/// One 3×3 box blur pass with edge clamping. Mirrors the mild camera-PSF
/// smoothing used by the Radon-peak refiner's synthetic fixtures.
fn box_blur_3x3(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for ky in -1i32..=1 {
                let sy = (y as i32 + ky).clamp(0, h as i32 - 1) as usize;
                for kx in -1i32..=1 {
                    let sx = (x as i32 + kx).clamp(0, w as i32 - 1) as usize;
                    acc += src[sy * w + sx] as u32;
                }
            }
            out[y * w + x] = (acc / 9) as u8;
        }
    }
    out
}
