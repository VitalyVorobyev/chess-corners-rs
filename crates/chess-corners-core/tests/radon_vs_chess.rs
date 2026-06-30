//! ChESS-hostile fixture test.
//!
//! The Radon detector was added because there are frames where the
//! ChESS ring kernel in [`chess_response_u8`](chess_corners_core::chess_response_u8)
//! produces no useful signal — heavy blur or low contrast dilutes the
//! bimodal 5 px ring that ChESS's `SR − DR − 16·|μₙ − μₗ|` formula
//! depends on. This test constructs such a fixture and asserts that
//! the Radon detector recovers corners the ChESS path misses.
//!
//! The assertion isn't "ChESS finds zero and Radon finds everything" —
//! ChESS can still latch onto some corners at any scale. What we pin
//! here is the *relative* gap: Radon should find substantially more
//! corners under hostile conditions, proving the code path earns its
//! keep.

use chess_corners_core::{
    chess_response_u8, detect_corners_from_response, detect_peaks_from_radon, radon_response_u8,
    ChessParams, RadonBuffers, RadonDetectorParams,
};
use chess_corners_testutil::{aa_chessboard, expected_corner_count, gaussian_blur};

/// Render a chessboard, then simulate a hostile capture: heavy
/// Gaussian blur and intensity-compressed values. Tuned so the ChESS
/// ring kernel's SR-DR signature sinks into the noise floor: a
/// large-σ blur washes out the 5-px bimodal ring, and a compressed
/// contrast range narrows the intra-ring differences the kernel
/// depends on.
fn hostile_chessboard(size: usize, cell: usize, offset: (f32, f32)) -> Vec<u8> {
    // Narrow contrast band — 30 gray levels.
    let dark = 108u8;
    let bright = 138u8;
    let img = aa_chessboard(size, cell, offset, dark, bright);
    let mut blurred = img;
    // Large blur radius vs. cell size so the ring sees near-uniform
    // grey at every position.
    gaussian_blur(&mut blurred, size, 2.5);
    blurred
}

#[test]
fn radon_beats_chess_on_blurred_low_contrast_board() {
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = hostile_chessboard(SIZE, CELL, offset);

    // ChESS default pipeline: canonical 5-ring, strict positive `R`.
    let chess_params = ChessParams::default();
    let chess_resp = chess_response_u8(&img, SIZE, SIZE, &chess_params);
    let chess_corners = detect_corners_from_response(&chess_resp, &chess_params);

    // Radon detector.
    let mut radon_params = RadonDetectorParams::default();
    radon_params.image_upsample = 2;
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(&img, SIZE, SIZE, &radon_params, &mut buffers);
    let radon_corners = detect_peaks_from_radon(&resp, &radon_params);

    let expected = expected_corner_count(SIZE, CELL, offset, 20);
    eprintln!(
        "ChESS-hostile ({}×{}, cell={}, σ=2.5 blur, contrast=108..138): expected~{} corners",
        SIZE, SIZE, CELL, expected
    );
    eprintln!(
        "  ChESS found {} corners; Radon found {} corners",
        chess_corners.len(),
        radon_corners.len()
    );

    // The contract: Radon must recover substantially more corners
    // than ChESS on this hostile fixture. If ChESS happens to find
    // some, we still expect Radon to beat it by a wide margin.
    assert!(
        radon_corners.len() > chess_corners.len() + 8,
        "expected Radon > ChESS + 8 on hostile fixture; got ChESS={}, Radon={}",
        chess_corners.len(),
        radon_corners.len(),
    );
    // And the Radon detector must land close to the true corner count.
    let radon_recovery = radon_corners.len() as f32 / expected as f32;
    assert!(
        radon_recovery >= 0.6,
        "Radon recovered only {:.0}% of corners on hostile fixture",
        radon_recovery * 100.0
    );
}

#[test]
fn both_paths_agree_on_clean_fixture() {
    // Sanity check: on a clean, high-contrast board, both paths
    // should recover most corners. This protects against over-tuning
    // the Radon detector so that it ONLY works on pathological
    // inputs.
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = aa_chessboard(SIZE, CELL, offset, 30, 230);

    let chess_params = ChessParams::default();
    let chess_resp = chess_response_u8(&img, SIZE, SIZE, &chess_params);
    let chess_corners = detect_corners_from_response(&chess_resp, &chess_params);

    let mut radon_params = RadonDetectorParams::default();
    radon_params.image_upsample = 2;
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(&img, SIZE, SIZE, &radon_params, &mut buffers);
    let radon_corners = detect_peaks_from_radon(&resp, &radon_params);

    let expected = expected_corner_count(SIZE, CELL, offset, 20);
    eprintln!(
        "Clean fixture: expected~{}; ChESS={}; Radon={}",
        expected,
        chess_corners.len(),
        radon_corners.len()
    );
    assert!(
        chess_corners.len() * 2 >= expected,
        "ChESS under-recovery on clean"
    );
    assert!(
        radon_corners.len() * 2 >= expected,
        "Radon under-recovery on clean"
    );
}
