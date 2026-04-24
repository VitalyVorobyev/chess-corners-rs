//! End-to-end pipeline test for `DetectorMode::Radon`.
//!
//! The core-level test in `chess-corners-core/tests/radon_vs_chess.rs`
//! already exercises the raw Radon detector (`radon_response_u8`,
//! `detect_corners_from_radon`). This file is the facade-level twin:
//! it verifies that flipping `ChessConfig::detector_mode` to
//! `DetectorMode::Radon` and calling the public `find_chess_corners`
//! entry point routes through the Radon path end-to-end, produces
//! `CornerDescriptor` values in base-image coordinates, and beats the
//! ChESS default on a hostile fixture.

use chess_corners::{find_chess_corners, ChessConfig, ImageView, RadonDetectorParams};

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
fn radon_mode_beats_chess_default_end_to_end() {
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = hostile_chessboard(SIZE, CELL, offset);
    let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();

    // Baseline: default facade config — ChESS canonical ring, zero
    // threshold, CenterOfMass refiner. This is what most users would
    // reach for first.
    let chess_cfg = ChessConfig::default();
    let chess_corners = find_chess_corners(view, &chess_cfg);

    // Radon preset: `ChessConfig::radon()` flips detector_mode and
    // keeps single-scale.
    let mut radon_cfg = ChessConfig::radon();
    radon_cfg.radon_detector = RadonDetectorParams {
        image_upsample: 2,
        ..RadonDetectorParams::default()
    };
    let radon_corners = find_chess_corners(view, &radon_cfg);

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
    // Sanity: on a clean, high-contrast board, flipping detector_mode
    // must not regress accuracy below what the ChESS path delivers.
    // This catches misconfiguration drift where the Radon path picks
    // up a bogus threshold or loses half its corners to a bug.
    const SIZE: usize = 129;
    const CELL: usize = 10;
    let offset = (13.4, 14.7);
    let img = aa_chessboard(SIZE, CELL, offset, 30, 230);
    let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();

    let chess_corners = find_chess_corners(view, &ChessConfig::default());
    let radon_corners = find_chess_corners(view, &ChessConfig::radon());

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
    let view = ImageView::from_u8_slice(SIZE, SIZE, &img).unwrap();

    let corners = find_chess_corners(view, &ChessConfig::radon());
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
