//! Accuracy of [`estimate_noise_sigma_u8`] on synthetic chessboards with
//! known noise.
//!
//! The reference is the realised noise: the standard deviation of
//! `noisy − clean` over the whole image, which includes the 8-bit rounding
//! of the noise (≈ `1/12` gray level² on top of the drawn σ²). Under
//! signal-dependent noise it is the root of the mean per-pixel variance.
//!
//! The full grid is a measurement (`--ignored --nocapture`, release); the
//! always-on tests check a few cells against a fixed tolerance.

use chess_corners_core::estimate_noise_sigma_u8;
use chess_corners_testutil::{
    add_gaussian_noise, add_signal_dependent_noise, area_chessboard, gaussian_blur,
    rotated_chessboard,
};

#[derive(Clone, Copy, Debug)]
enum Noise {
    White(f32),
    /// Variance `a + b·I`, `b = 8/255`.
    SignalDependent(f32),
}

impl Noise {
    fn label(self) -> String {
        match self {
            Noise::White(s) => format!("white {s}"),
            Noise::SignalDependent(a) => format!("sd a={a}"),
        }
    }
}

/// Board geometry.
#[derive(Clone, Copy, Debug)]
enum Geometry {
    /// Exact-area board whose second lattice axis is at this angle (rad).
    Area(f32),
    /// Supersampled square board rotated by this angle (degrees).
    Rotated(f32),
}

const AXIS: Geometry = Geometry::Area(core::f32::consts::FRAC_PI_2);
const SHEAR: Geometry = Geometry::Area(core::f32::consts::FRAC_PI_3);

/// Clean board, 40/200, blurred.
fn clean_board(size: usize, cell: usize, geometry: Geometry, blur: f32) -> Vec<u8> {
    let mut clean = match geometry {
        Geometry::Area(angle) => area_chessboard(size, cell, (0.37, 0.71), angle, 40, 200),
        Geometry::Rotated(deg) => {
            let center = (size as f32 / 2.0 + 0.37, size as f32 / 2.0 + 0.71);
            rotated_chessboard(size, cell, center, deg.to_radians(), 40, 200)
        }
    };
    gaussian_blur(&mut clean, size, blur);
    clean
}

/// Add noise to `clean`. Returns `(noisy, realised σ)`.
fn add_noise(clean: &[u8], noise: Noise, seed: u64) -> (Vec<u8>, f32) {
    let mut noisy = clean.to_vec();
    match noise {
        Noise::White(s) => add_gaussian_noise(&mut noisy, s, seed),
        Noise::SignalDependent(a) => add_signal_dependent_noise(&mut noisy, a, 8.0 / 255.0, seed),
    }
    let n = noisy.len() as f64;
    let (mut s1, mut s2) = (0f64, 0f64);
    for (&a, &b) in noisy.iter().zip(clean) {
        let d = f64::from(a) - f64::from(b);
        s1 += d;
        s2 += d * d;
    }
    let var = s2 / n - (s1 / n).powi(2);
    (noisy, var.sqrt() as f32)
}

/// Relative error of the estimate against the realised σ, `None` when the
/// estimator declines.
fn rel_err(clean: &[u8], size: usize, noise: Noise, seed: u64) -> Option<f32> {
    let (img, truth) = add_noise(clean, noise, seed);
    estimate_noise_sigma_u8(&img, size, size).map(|est| est / truth - 1.0)
}

#[test]
fn estimate_tracks_white_noise_on_boards() {
    // Sharp sheared, mildly blurred, heavily blurred (no flat area at all),
    // and rotated boards, including sharp edges along the window diagonal.
    let cases = [
        (8, SHEAR, 0.0),
        (8, AXIS, 0.8),
        (8, AXIS, 2.5),
        (8, Geometry::Rotated(45.0), 0.0),
        (12, Geometry::Rotated(45.0), 0.0),
        (12, Geometry::Rotated(22.5), 0.8),
    ];
    for (cell, geometry, blur) in cases {
        let clean = clean_board(160, cell, geometry, blur);
        for sigma in [1.0f32, 4.0] {
            let e = rel_err(&clean, 160, Noise::White(sigma), 7).unwrap_or_else(|| {
                panic!("no estimate: cell {cell}, {geometry:?}, blur {blur}, σ {sigma}")
            });
            assert!(
                e.abs() < 0.10,
                "cell {cell}, {geometry:?}, blur {blur}, σ {sigma}: relative error {e:+.3}"
            );
        }
    }
}

#[test]
fn estimate_declines_or_holds_without_flat_windows() {
    // Sharp small cells, sheared or at 45°: hardly any 4×4 window is free of
    // edges. The estimator either declines or stays accurate.
    for (cell, geometry, blur) in [
        (5, SHEAR, 0.0),
        (6, Geometry::Rotated(45.0), 0.0),
        (5, AXIS, 2.5),
    ] {
        let clean = clean_board(160, cell, geometry, blur);
        if let Some(e) = rel_err(&clean, 160, Noise::White(1.0), 7) {
            assert!(e.abs() < 0.10, "cell {cell}, {geometry:?}: {e:+.3}");
        }
    }
}

#[test]
#[ignore = "measurement: prints the noise-estimate accuracy grid"]
fn noise_estimate_grid() {
    let size = 256;
    let noises = [
        Noise::White(0.5),
        Noise::White(1.0),
        Noise::White(2.0),
        Noise::White(4.0),
        Noise::White(8.0),
        Noise::SignalDependent(1.0),
        Noise::SignalDependent(4.0),
    ];
    let geometries: [(&str, Geometry, [usize; 3]); 7] = [
        ("axis", AXIS, [5, 8, 12]),
        ("skew60", SHEAR, [5, 8, 12]),
        ("rot0", Geometry::Rotated(0.0), [6, 8, 12]),
        ("rot10", Geometry::Rotated(10.0), [6, 8, 12]),
        ("rot22.5", Geometry::Rotated(22.5), [6, 8, 12]),
        ("rot30", Geometry::Rotated(30.0), [6, 8, 12]),
        ("rot45", Geometry::Rotated(45.0), [6, 8, 12]),
    ];
    let header: Vec<String> = noises.iter().map(|n| n.label()).collect();
    println!("relative error of the estimate vs realised σ (mean over 3 seeds; `none`: declined)");
    println!("geometry | cell | blur | {}", header.join(" | "));
    let mut summary = Vec::new();
    for (geom, geometry, cells) in geometries {
        // Worst relative error per noise column, and declined cells.
        let mut worst = [0f32; 7];
        let mut declined = [0usize; 7];
        for cell in cells {
            for blur in [0.0f32, 0.8, 1.5, 2.5] {
                let clean = clean_board(size, cell, geometry, blur);
                let mut row = Vec::new();
                for (k, &noise) in noises.iter().enumerate() {
                    let errs: Vec<Option<f32>> = (0..3)
                        .map(|s| rel_err(&clean, size, noise, 100 + s))
                        .collect();
                    let n_none = errs.iter().filter(|e| e.is_none()).count();
                    if n_none > 0 {
                        declined[k] += 1;
                        row.push(format!("none×{n_none}"));
                        continue;
                    }
                    let e = errs.iter().map(|e| e.unwrap()).sum::<f32>() / 3.0;
                    if e.abs() > worst[k].abs() {
                        worst[k] = e;
                    }
                    row.push(format!("{e:+.3}"));
                }
                println!("{geom} | {cell} | {blur} | {}", row.join(" | "));
            }
        }
        let cols: Vec<String> = worst
            .iter()
            .zip(declined)
            .map(|(e, d)| {
                if d > 0 {
                    format!("{e:+.3} ({d} none)")
                } else {
                    format!("{e:+.3}")
                }
            })
            .collect();
        summary.push(format!("{geom} | {}", cols.join(" | ")));
    }
    println!("\nworst relative error per geometry (cells with none: declined)");
    println!("geometry | {}", header.join(" | "));
    for line in summary {
        println!("{line}");
    }

    // The realised σ includes the 8-bit rounding: compare with √(σ² + 1/12).
    let clean = clean_board(size, 12, AXIS, 0.0);
    for sigma in [0.5f32, 1.0, 2.0] {
        let (img, truth) = add_noise(&clean, Noise::White(sigma), 100);
        let est = estimate_noise_sigma_u8(&img, size, size).unwrap();
        let quantized = (sigma * sigma + 1.0 / 12.0).sqrt();
        println!(
            "white {sigma}: realised {truth:.3}, √(σ² + 1/12) {quantized:.3}, estimate {est:.3}"
        );
    }

    // A dark half near black (mean 2σ above 0): clipped there, the windows
    // near the rail are skipped by their mean.
    for sigma in [2.0f32, 4.0] {
        let mut clean = vec![128u8; size * size];
        let dark = (2.0 * sigma).round() as u8;
        clean[..size * size / 2].fill(dark);
        let (img, _) = add_noise(&clean, Noise::White(sigma), 100);
        let est = estimate_noise_sigma_u8(&img, size, size).unwrap();
        let quantized = (sigma * sigma + 1.0 / 12.0).sqrt();
        println!("half dark at {dark}, white {sigma}: estimate {est:.3} vs {quantized:.3}");
    }
}
