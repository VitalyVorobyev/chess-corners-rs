//! End-to-end pipeline test for the Radon detection strategy.
//!
//! The core-level test in `chess-corners-core/tests/radon_vs_chess.rs`
//! already exercises the raw Radon detector (`radon_response_u8`,
//! `detect_peaks_from_radon`). This file is the facade-level twin:
//! it verifies that selecting [`DetectionStrategy::Radon`] and driving
//! the public `Detector::detect_u8` entry point routes through the
//! Radon path end-to-end, produces `CornerDescriptor` values in
//! base-image coordinates, and beats the ChESS default on a hostile
//! fixture.

use chess_corners::{DetectionStrategy, Detector, DetectorConfig};
use chess_corners_testutil::{aa_chessboard, expected_corner_count, gaussian_blur};

fn detect_u8(
    img: &[u8],
    width: usize,
    height: usize,
    cfg: &DetectorConfig,
) -> Vec<chess_corners::CornerDescriptor> {
    let mut detector = Detector::new(*cfg).expect("config valid");
    detector
        .detect_u8(img, width as u32, height as u32)
        .expect("image dimensions match")
}

/// Narrow-contrast, heavily-blurred chessboard — the fixture style
/// from `chess-corners-core/tests/radon_vs_chess.rs`. ChESS's 5-px
/// ring kernel cannot recover a useful SR-DR signature from this, but
/// the Radon integral is still well-defined.
fn hostile_chessboard(size: usize, cell: usize, offset: (f32, f32)) -> Vec<u8> {
    let dark = 108u8;
    let bright = 138u8;
    let mut img = aa_chessboard(size, cell, offset, dark, bright);
    gaussian_blur(&mut img, size, 2.5);
    img
}

#[test]
fn radon_mode_beats_chess_default_end_to_end() {
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = hostile_chessboard(SIZE, CELL, offset);

    // Baseline: default facade config — ChESS canonical ring, zero
    // threshold, CenterOfMass refiner. This is what most users would
    // reach for first.
    let chess_cfg = DetectorConfig::default();
    let chess_corners = detect_u8(&img, SIZE, SIZE, &chess_cfg);

    // Radon preset: `DetectorConfig::radon()` selects the Radon strategy
    // and keeps single-scale.
    let mut radon_cfg = DetectorConfig::radon();
    if let DetectionStrategy::Radon(radon) = &mut radon_cfg.strategy {
        radon.image_upsample = 2;
    }
    let radon_corners = detect_u8(&img, SIZE, SIZE, &radon_cfg);

    let expected = expected_corner_count(SIZE, CELL, offset, 20);
    eprintln!(
        "hostile {}x{} cell={} σ=2.5 blur: expected ~{}; ChESS={}, Radon={}",
        SIZE,
        SIZE,
        CELL,
        expected,
        chess_corners.len(),
        radon_corners.len(),
    );

    // Contract: the Radon pipeline must recover substantially more
    // corners than ChESS defaults on this fixture, with real
    // subpixel coordinates (strictly inside the image, not clipped).
    assert!(
        radon_corners.len() > chess_corners.len() + 8,
        "Radon must beat ChESS+8 on hostile fixture; got ChESS={}, Radon={}",
        chess_corners.len(),
        radon_corners.len(),
    );
    let recovery = radon_corners.len() as f32 / expected as f32;
    assert!(
        recovery >= 0.6,
        "Radon recovered only {:.0}% of corners ({} / {})",
        recovery * 100.0,
        radon_corners.len(),
        expected,
    );
    for c in &radon_corners {
        assert!(c.x > 0.0 && c.x < SIZE as f32, "x out of image: {:?}", c);
        assert!(c.y > 0.0 && c.y < SIZE as f32, "y out of image: {:?}", c);
    }
}

#[test]
fn radon_mode_agrees_with_chess_on_clean_fixture() {
    // Sanity: on a clean, high-contrast board, switching to the Radon
    // strategy must not regress accuracy below what the ChESS path delivers.
    // This catches misconfiguration drift where the Radon path picks
    // up a bogus threshold or loses half its corners to a bug.
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = aa_chessboard(SIZE, CELL, offset, 30, 230);

    let chess_corners = detect_u8(&img, SIZE, SIZE, &DetectorConfig::default());
    let radon_corners = detect_u8(&img, SIZE, SIZE, &DetectorConfig::radon());

    let expected = expected_corner_count(SIZE, CELL, offset, 20);
    // Both paths should land within 20 % of the ground truth count.
    assert!(
        chess_corners.len() as f32 >= 0.8 * expected as f32,
        "ChESS under-recovered on clean board: {} / {}",
        chess_corners.len(),
        expected,
    );
    assert!(
        radon_corners.len() as f32 >= 0.8 * expected as f32,
        "Radon under-recovered on clean board: {} / {}",
        radon_corners.len(),
        expected,
    );
}

#[test]
fn radon_mode_corners_are_subpixel_accurate() {
    // Contract: on a clean board, of the corners the Radon pipeline
    // locks onto within 1 px of a ground-truth grid intersection, the
    // mean residual must be well under 0.2 px. This guards against
    // coordinate scaling bugs (e.g. if `image_upsample=2` offsets are
    // not divided back to the input frame) and peak-fit regressions,
    // while tolerating the detector's tendency to over-produce
    // side-lobe peaks on the SAT-based response — those show up as
    // snapped-distant corners and are filtered out of the residual
    // calculation (we count them separately below).
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = aa_chessboard(SIZE, CELL, offset, 30, 230);

    let corners = detect_u8(&img, SIZE, SIZE, &DetectorConfig::radon());
    assert!(!corners.is_empty(), "Radon found no corners on clean board");

    let mut total_err = 0.0f64;
    let mut counted = 0;
    let mut total = 0;
    for c in &corners {
        total += 1;
        let kx = ((c.x - offset.0) / CELL as f32).round();
        let ky = ((c.y - offset.1) / CELL as f32).round();
        let tx = offset.0 + kx * CELL as f32;
        let ty = offset.1 + ky * CELL as f32;
        let dx = (c.x - tx) as f64;
        let dy = (c.y - ty) as f64;
        let d = (dx * dx + dy * dy).sqrt();
        if d < 1.0 {
            total_err += d;
            counted += 1;
        }
    }
    assert!(counted > 0, "no Radon corners snapped to ground truth");
    let mean_err = total_err / counted as f64;
    eprintln!(
        "Radon accuracy on clean fixture: {:.4} px over {}/{} corners within 1 px of grid",
        mean_err, counted, total
    );
    assert!(
        mean_err < 0.2,
        "Radon mean error {:.4} px exceeds 0.2 px bar",
        mean_err
    );
    // At least half of emitted corners should snap to real grid
    // intersections; more than that means we're returning noise.
    let snap_rate = counted as f32 / total as f32;
    assert!(
        snap_rate >= 0.5,
        "only {:.0}% of Radon corners landed within 1 px of a grid intersection",
        snap_rate * 100.0,
    );
}

/// Every coarse-to-fine corner is a single-scale corner: same position,
/// same response.
fn assert_multiscale_subset(
    label: &str,
    single: &[chess_corners::CornerDescriptor],
    multi: &[chess_corners::CornerDescriptor],
) {
    for m in multi {
        assert!(
            single
                .iter()
                .any(|s| (s.x - m.x).hypot(s.y - m.y) < 1e-3 && s.response == m.response),
            "{label}: multiscale corner ({}, {}) response {} is not a single-scale corner",
            m.x,
            m.y,
            m.response
        );
    }
}

/// The coarse-to-fine Radon preset applies the relative threshold against
/// the frame, not against each refinement window, and seeds every board
/// corner: on the test images it returns single-scale corners only, at
/// the same positions, and on `mid` and `large` all of them. On `small` (a
/// ChArUco board) the single-scale pass also reports X-junctions between
/// marker bits, which the half-resolution level does not resolve.
#[test]
fn radon_multiscale_returns_single_scale_corners_on_test_images() {
    for (name, min_fraction) in [("small", 0.8), ("mid", 1.0), ("large", 1.0)] {
        let path = format!("{}/../../testimages/{name}.png", env!("CARGO_MANIFEST_DIR"));
        let img = image::open(&path).expect("test image").to_luma8();
        let single = Detector::new(DetectorConfig::radon())
            .unwrap()
            .detect(&img)
            .unwrap();
        let multi = Detector::new(DetectorConfig::radon_multiscale())
            .unwrap()
            .detect(&img)
            .unwrap();
        eprintln!(
            "{name}: single-scale {} corners, multiscale {}",
            single.len(),
            multi.len()
        );
        assert!(!single.is_empty());
        assert_multiscale_subset(name, &single, &multi);
        assert!(
            multi.len() as f32 >= min_fraction * single.len() as f32,
            "{name}: multiscale {} of {} single-scale corners",
            multi.len(),
            single.len()
        );
    }
}

/// A board whose contrast fades across the frame: the faint corners fall
/// below the relative threshold of the full frame but would top every
/// refinement window that holds no bright corner. Multiscale and
/// single-scale Radon agree corner for corner.
#[test]
fn radon_multiscale_matches_single_scale_on_fading_board() {
    const SIZE: usize = 480;
    const CELL: usize = 24;
    let offset = (11.3, 13.7);
    let mut img = aa_chessboard(SIZE, CELL, offset, 40, 220);
    for y in 0..SIZE {
        for x in 0..SIZE {
            // Contrast falls linearly from 1 at the left edge to 0.15 at
            // the right edge, about a mid-gray of 130.
            let gain = 1.0 - 0.85 * x as f32 / (SIZE - 1) as f32;
            let v = &mut img[y * SIZE + x];
            *v = (130.0 + gain * (f32::from(*v) - 130.0)).round() as u8;
        }
    }
    let single = detect_u8(&img, SIZE, SIZE, &DetectorConfig::radon());
    let multi = detect_u8(&img, SIZE, SIZE, &DetectorConfig::radon_multiscale());
    let expected = expected_corner_count(SIZE, CELL, offset, 20);
    eprintln!(
        "fading board: {expected} corners, single-scale {}, multiscale {}",
        single.len(),
        multi.len()
    );
    // The relative threshold keeps only the high-contrast part of the board.
    assert!(single.len() > expected / 4 && single.len() < expected);
    assert_multiscale_subset("fading board", &single, &multi);
    assert_eq!(multi.len(), single.len());
}
