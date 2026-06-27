//! Shared synthetic-image helpers for the `chess-corners-core` benches.
//!
//! Lives in a `benches/common/` subdirectory (not a top-level
//! `benches/*.rs`) so Cargo does not treat it as its own bench target;
//! each bench binary pulls it in with `mod common;`. Centralising the
//! generator keeps the ChESS-response and Radon-response benches on the
//! exact same deterministic board (SOLID-01: one synthetic-board helper
//! rather than a per-bench copy).

/// Deterministic hard-edge chessboard image in row-major 8-bit grayscale.
///
/// Corner density is roughly 25 cells across the shorter image
/// dimension, which matches a typical calibration target in frame. The
/// pattern is fully deterministic so response timings are comparable
/// across machines and across feature builds.
pub fn synth_chessboard(w: usize, h: usize) -> Vec<u8> {
    let cell = (h.min(w) / 25).max(8) as i32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = (x as i32) / cell;
            let cy = (y as i32) / cell;
            out[y * w + x] = if (cx + cy) & 1 == 0 { 40 } else { 215 };
        }
    }
    out
}
