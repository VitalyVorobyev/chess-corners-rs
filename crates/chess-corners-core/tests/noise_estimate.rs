//! Accuracy of [`estimate_noise_sigma_u8`] on synthetic chessboards with
//! known noise.
//!
//! The reference is the realised noise: the standard deviation of
//! `noisy − clean` over the whole image, which includes the 8-bit rounding
//! of the noise (≈ `1/12` gray level² on top of the drawn σ²). Under
//! signal-dependent noise it is the root of the mean per-pixel variance.
//!
//! The full grid is a measurement (`--ignored --nocapture`, release); the
//! always-on test checks a few cells against a fixed tolerance.

use chess_corners_core::estimate_noise_sigma_u8;
use chess_corners_testutil::{
    add_gaussian_noise, add_signal_dependent_noise, area_chessboard, gaussian_blur,
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

/// Render, blur, add noise. Returns `(noisy, realised σ)`.
fn board(
    size: usize,
    cell: usize,
    axis_angle: f32,
    blur: f32,
    noise: Noise,
    seed: u64,
) -> (Vec<u8>, f32) {
    let mut clean = area_chessboard(size, cell, (0.37, 0.71), axis_angle, 40, 200);
    gaussian_blur(&mut clean, size, blur);
    let mut noisy = clean.clone();
    match noise {
        Noise::White(s) => add_gaussian_noise(&mut noisy, s, seed),
        Noise::SignalDependent(a) => add_signal_dependent_noise(&mut noisy, a, 8.0 / 255.0, seed),
    }
    let n = noisy.len() as f64;
    let (mut s1, mut s2) = (0f64, 0f64);
    for (&a, &b) in noisy.iter().zip(&clean) {
        let d = f64::from(a) - f64::from(b);
        s1 += d;
        s2 += d * d;
    }
    let var = s2 / n - (s1 / n).powi(2);
    (noisy, var.sqrt() as f32)
}

const AXIS: f32 = core::f32::consts::FRAC_PI_2;
const SHEAR: f32 = core::f32::consts::FRAC_PI_3;

/// Relative error of the estimate against the realised σ, `None` when the
/// estimator declines.
fn rel_err(
    size: usize,
    cell: usize,
    angle: f32,
    blur: f32,
    noise: Noise,
    seed: u64,
) -> Option<f32> {
    let (img, truth) = board(size, cell, angle, blur, noise, seed);
    estimate_noise_sigma_u8(&img, size, size).map(|est| est / truth - 1.0)
}

#[test]
fn estimate_tracks_white_noise_on_boards() {
    // Sharp sheared, mildly blurred and heavily blurred boards (the last
    // has no flat area at all).
    for &(cell, angle, blur) in &[(8, SHEAR, 0.0), (8, AXIS, 0.8), (5, AXIS, 2.5)] {
        for &sigma in &[1.0f32, 4.0] {
            let e = rel_err(160, cell, angle, blur, Noise::White(sigma), 7)
                .unwrap_or_else(|| panic!("no estimate: cell {cell}, blur {blur}, σ {sigma}"));
            assert!(
                e.abs() < 0.10,
                "cell {cell}, angle {angle}, blur {blur}, σ {sigma}: relative error {e:+.3}"
            );
        }
    }
}

#[test]
fn estimate_declines_without_flat_windows() {
    // Sharp 5 px cells, sheared: every 4×4 window straddles an edge, so the
    // gate leaves too few windows and no estimate is reported.
    assert_eq!(rel_err(160, 5, SHEAR, 0.0, Noise::White(1.0), 7), None);
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
    println!("relative error of the estimate vs realised σ (mean over 3 seeds; `none`: declined)");
    println!("geometry | cell | blur | {}", {
        let v: Vec<String> = noises.iter().map(|n| n.label()).collect();
        v.join(" | ")
    });
    let mut worst = [0f32; 7];
    for &(geom, angle) in &[("axis", AXIS), ("skew60", SHEAR)] {
        for &cell in &[5usize, 8, 12] {
            for &blur in &[0.0f32, 0.8, 1.5, 2.5] {
                let mut row = Vec::new();
                for (k, &noise) in noises.iter().enumerate() {
                    let errs: Vec<Option<f32>> = (0..3)
                        .map(|s| rel_err(size, cell, angle, blur, noise, 100 + s))
                        .collect();
                    if errs.iter().any(Option::is_none) {
                        let n = errs.iter().filter(|e| e.is_none()).count();
                        row.push(format!("none×{n}"));
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
    }
    let w: Vec<String> = worst.iter().map(|e| format!("{e:+.3}")).collect();
    println!("worst | | | {}", w.join(" | "));
}
