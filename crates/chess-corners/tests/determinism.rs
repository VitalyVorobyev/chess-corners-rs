//! Guards the determinism contract from `AGENTS.md` §4 ("Coding
//! conventions" → "Determinism"): "If parallelism is enabled, final
//! output ordering must be deterministic (sort by stable keys)."
//!
//! The `rayon` paths in this workspace hold that contract by
//! construction rather than via an explicit post-hoc sort: response
//! rows are split into independent chunks and reassembled by row
//! index (`enumerate()` + ordered `collect()`), so the scan order —
//! not any data-dependent ordering — determines the output order.
//! Nothing currently in the pipeline would stop a future refactor
//! from swapping an ordered `collect()` for an unordered one (e.g.
//! draining a channel), so this test pins the observable contract:
//! repeated detection on the same input must return the same corners
//! in the same order, bit-for-bit, every time.
//!
//! Only meaningful with the `rayon` feature enabled — the sequential
//! path has no reordering to guard against. Run via
//! `cargo test -p chess-corners --features rayon` (or `--all-features`).

#![cfg(feature = "rayon")]

use chess_corners::{CornerDescriptor, Detector, DetectorConfig};
use chess_corners_testutil::aa_chessboard;

fn synthetic_board(size: usize) -> Vec<u8> {
    aa_chessboard(size, 12, (0.3, -0.4), 20, 220)
}

/// Bit-for-bit comparison: `f32::to_bits` distinguishes `+0.0`/`-0.0`
/// and propagates through `NaN` payloads (relevant for `axes: None`
/// vs. a fitted axis), so it is stricter than `==` and exactly what
/// "byte-identical" means for float output.
fn assert_descriptors_bit_identical(label: &str, a: &[CornerDescriptor], b: &[CornerDescriptor]) {
    assert_eq!(a.len(), b.len(), "{label}: corner count differs");
    for (i, (da, db)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            da.x.to_bits(),
            db.x.to_bits(),
            "{label}: corner {i} x differs"
        );
        assert_eq!(
            da.y.to_bits(),
            db.y.to_bits(),
            "{label}: corner {i} y differs"
        );
        assert_eq!(
            da.response.to_bits(),
            db.response.to_bits(),
            "{label}: corner {i} response differs"
        );
        match (da.axes, db.axes) {
            (None, None) => {}
            (Some(axa), Some(axb)) => {
                for k in 0..2 {
                    assert_eq!(
                        axa[k].angle.to_bits(),
                        axb[k].angle.to_bits(),
                        "{label}: corner {i} axis {k} angle differs"
                    );
                    assert_eq!(
                        axa[k].sigma.to_bits(),
                        axb[k].sigma.to_bits(),
                        "{label}: corner {i} axis {k} sigma differs"
                    );
                }
            }
            _ => panic!("{label}: corner {i} axes presence differs"),
        }
    }
}

/// Repeated calls on one detector instance (reusing its scratch
/// buffers across frames, the normal calling pattern) must produce
/// identical, identically-ordered output every time.
#[test]
fn repeated_detection_on_reused_detector_is_byte_identical_and_stably_ordered() {
    let size = 160usize;
    let img = synthetic_board(size);
    let cfg = DetectorConfig::chess_multiscale();
    let mut detector = Detector::new(cfg).expect("valid config");

    let first = detector
        .detect_u8(&img, size as u32, size as u32)
        .expect("dimensions match");
    assert!(!first.is_empty(), "fixture should yield corners");

    for run in 1..=5 {
        let again = detector
            .detect_u8(&img, size as u32, size as u32)
            .expect("dimensions match");
        assert_descriptors_bit_identical(&format!("reused detector, run {run}"), &first, &again);
    }
}

/// A freshly constructed detector (no warm scratch buffers) must
/// agree with a reused one — proves the ordering guarantee doesn't
/// accidentally depend on buffer state left over from a previous call.
#[test]
fn fresh_detector_matches_reused_detector() {
    let size = 160usize;
    let img = synthetic_board(size);
    let cfg = DetectorConfig::chess_multiscale();

    let mut warm = Detector::new(cfg).expect("valid config");
    let warm_baseline = warm
        .detect_u8(&img, size as u32, size as u32)
        .expect("dimensions match");
    // Run a few more times to actually warm the reusable buffers.
    for _ in 0..3 {
        let _ = warm
            .detect_u8(&img, size as u32, size as u32)
            .expect("dimensions match");
    }
    let warm_again = warm
        .detect_u8(&img, size as u32, size as u32)
        .expect("dimensions match");
    assert_descriptors_bit_identical("warm detector", &warm_baseline, &warm_again);

    let mut fresh = Detector::new(cfg).expect("valid config");
    let fresh_result = fresh
        .detect_u8(&img, size as u32, size as u32)
        .expect("dimensions match");
    assert_descriptors_bit_identical("fresh vs. warm detector", &warm_baseline, &fresh_result);
}

/// Same contract on the Radon strategy, which has its own independent
/// rayon row-chunking in `detect_peaks_from_radon`.
#[test]
fn repeated_radon_detection_is_byte_identical_and_stably_ordered() {
    let size = 160usize;
    let img = synthetic_board(size);
    let cfg = DetectorConfig::radon();
    let mut detector = Detector::new(cfg).expect("valid config");

    let first = detector
        .detect_u8(&img, size as u32, size as u32)
        .expect("dimensions match");
    assert!(!first.is_empty(), "fixture should yield corners");

    for run in 1..=5 {
        let again = detector
            .detect_u8(&img, size as u32, size as u32)
            .expect("dimensions match");
        assert_descriptors_bit_identical(&format!("radon, run {run}"), &first, &again);
    }
}
