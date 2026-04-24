//! ChESS-hostile fixture test.
//!
//! The Radon detector was added because there are frames where the
//! ChESS ring kernel in [`chess_response_u8`](chess_corners_core::response::chess_response_u8)
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
    detect::detect_corners_from_response, detect_corners_from_radon, radon_response_u8,
    response::chess_response_u8, ChessParams, RadonBuffers, RadonDetectorParams,
};

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

fn aa_chessboard(size: usize, cell: usize, offset: (f32, f32), dark: u8, bright: u8) -> Vec<u8> {
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

fn gaussian_blur(img: &mut [u8], size: usize, sigma: f32) {
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let klen = 2 * radius + 1;
    let mut kernel = vec![0f32; klen];
    let mut sum = 0f32;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        *k = (-(x * x) / (2.0 * sigma * sigma)).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    let mut tmp = vec![0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += img[y * size + sx] as f32 * k;
            }
            tmp[y * size + x] = acc;
        }
    }
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += tmp[sy * size + x] * k;
            }
            img[y * size + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Count the ground-truth corners visible in an `size × size` image
/// given a chessboard with `cell` cells and `offset` origin. A
/// "visible" corner sits on an `(offset.x + k·cell, offset.y +
/// m·cell)` junction strictly inside a `border`-wide interior.
fn expected_corner_count(size: usize, cell: usize, offset: (f32, f32), border: usize) -> usize {
    let lo = border as f32;
    let hi = (size - border) as f32;
    let (ox, oy) = offset;
    let c = cell as f32;
    let mut n = 0;
    let mut k = -((size as f32) / c).ceil() as i32;
    while (k as f32) * c + ox < hi {
        let kx = ox + (k as f32) * c;
        if kx >= lo && kx <= hi {
            let mut m = -((size as f32) / c).ceil() as i32;
            while (m as f32) * c + oy < hi {
                let my = oy + (m as f32) * c;
                if my >= lo && my <= hi {
                    n += 1;
                }
                m += 1;
            }
        }
        k += 1;
    }
    n
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
    let radon_params = RadonDetectorParams {
        image_upsample: 2,
        ..RadonDetectorParams::default()
    };
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(&img, SIZE, SIZE, &radon_params, &mut buffers);
    let radon_corners = detect_corners_from_radon(&resp, &radon_params);

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

    let radon_params = RadonDetectorParams {
        image_upsample: 2,
        ..RadonDetectorParams::default()
    };
    let mut buffers = RadonBuffers::new();
    let resp = radon_response_u8(&img, SIZE, SIZE, &radon_params, &mut buffers);
    let radon_corners = detect_corners_from_radon(&resp, &radon_params);

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
