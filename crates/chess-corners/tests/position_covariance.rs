//! Position covariance on `CornerDescriptor`, end to end through the
//! facade: which configurations report it, and whether it is calibrated.
//!
//! Calibration is checked the way a downstream calibration uses it: whiten
//! each matched corner's error against the truth with its reported px²
//! covariance and average, `σ̂₀² = mean(eᵀ Σ⁻¹ e) / 2`, which is 1 when the
//! covariance is right. The boards are formed as a camera forms them
//! (Gaussian lens blur, then exact integration over each pixel), so the
//! corner positions are known exactly.

use std::sync::OnceLock;

use chess_corners::{
    ChessRefiner, CornerDescriptor, Detector, DetectorConfig, MultiscaleConfig, RadonRefiner, Roi,
    Uncertainty, UpscaleConfig,
};
use chess_corners_core::estimate_noise_sigma_u8;
use chess_corners_testutil::{add_gaussian_noise, optical_chessboard, Lattice};
use rayon::prelude::*;

const SIZE: usize = 200;
const DEG: f32 = core::f32::consts::PI / 180.0;

/// Board geometry, contrast, lens blur and pixel noise.
#[derive(Clone, Copy, Debug)]
struct Fixture {
    cell: f32,
    /// Angle of the first lattice axis from the image x axis.
    rotation: f32,
    /// Angle between the lattice axes (π/2: square board).
    axis_angle: f32,
    dark: u8,
    bright: u8,
    /// Gaussian lens PSF σ, px.
    psf: f32,
    sigma: f32,
}

impl Fixture {
    /// Square, grid-aligned 8 px board, 40/200.
    fn new(psf: f32, sigma: f32) -> Self {
        Self {
            cell: 8.0,
            rotation: 0.0,
            axis_angle: 90.0 * DEG,
            dark: 40,
            bright: 200,
            psf,
            sigma,
        }
    }

    fn rotated(self, degrees: f32) -> Self {
        Self {
            rotation: degrees * DEG,
            ..self
        }
    }

    /// Lattice with a corner at `(3, 3) + phase`.
    fn lattice(&self, phase: (f32, f32)) -> Lattice {
        Lattice::new(self.cell, (3.0 + phase.0, 3.0 + phase.1))
            .rotated(self.rotation)
            .with_axis_angle(self.axis_angle)
    }

    fn clean(&self, lattice: &Lattice) -> Vec<u8> {
        optical_chessboard(SIZE, lattice, self.psf, self.dark, self.bright)
    }

    /// `clean` with white noise. Returns the image and the realised noise σ
    /// (`noisy − clean`, rounding included).
    fn noisy(&self, clean: &[u8], seed: u64) -> (Vec<u8>, f32) {
        let mut img = clean.to_vec();
        add_gaussian_noise(&mut img, self.sigma, seed);
        let n = img.len() as f64;
        let ss: f64 = img
            .iter()
            .zip(clean)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        (img, (ss / n).sqrt() as f32)
    }
}

/// The standard image of the non-ignored tests: grid-aligned, lens blur
/// 0.8 px, σₙ = 2, rendered once.
fn standard_image() -> &'static [u8] {
    static IMG: OnceLock<Vec<u8>> = OnceLock::new();
    IMG.get_or_init(|| {
        let fx = Fixture::new(0.8, 2.0);
        let clean = fx.clean(&fx.lattice((0.3, 0.6)));
        fx.noisy(&clean, 7).0
    })
}

fn saddle() -> DetectorConfig {
    DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::saddle_point())
}

fn radon_saddle() -> DetectorConfig {
    DetectorConfig::radon().with_radon(|r| r.refiner = RadonRefiner::saddle_point())
}

/// Every localizer, as the facade composes them.
fn localizers() -> [(&'static str, DetectorConfig); 5] {
    [
        ("ChESS + CenterOfMass", DetectorConfig::chess()),
        (
            "ChESS + Förstner",
            DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::forstner()),
        ),
        ("ChESS + SaddlePoint", saddle()),
        ("Radon peak fit", DetectorConfig::radon()),
        ("Radon + SaddlePoint", radon_saddle()),
    ]
}

/// Whitening statistics over a set of boards.
#[derive(Default)]
struct Whitened {
    /// Σ eᵀΣ⁻¹e with the reported covariance.
    sum: f64,
    /// Corners with a covariance, matched to the truth.
    n: usize,
    /// Detections farther than 0.5 px from any lattice corner.
    unmatched: usize,
}

impl Whitened {
    fn add(&mut self, lattice: &Lattice, corners: &[CornerDescriptor]) {
        for c in corners {
            let Some(cov) = c.position_covariance else {
                continue;
            };
            let (tx, ty) = lattice.nearest_corner(c.x, c.y);
            let (ex, ey) = (f64::from(c.x - tx), f64::from(c.y - ty));
            if ex.abs() >= 0.5 || ey.abs() >= 0.5 {
                self.unmatched += 1;
                continue;
            }
            let (a, b, d) = (f64::from(cov.xx), f64::from(cov.xy), f64::from(cov.yy));
            self.sum += (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / (a * d - b * b);
            self.n += 1;
        }
    }

    /// σ̂₀² with the reported covariance.
    fn sigma0_sq(&self) -> f64 {
        self.sum / (2.0 * self.n as f64)
    }
}

/// Subpixel phase of board `k`: a 5×5 grid, so grid-aligned boards see
/// every phase (rotated boards vary it across the board anyway).
fn phase(k: usize) -> (f32, f32) {
    ((k % 5) as f32 / 5.0, (k / 5 % 5) as f32 / 5.0)
}

/// Clean renders of `boards` phases of `fixture`, with their lattices.
fn clean_boards(fixture: Fixture, boards: usize) -> Vec<(Lattice, Vec<u8>)> {
    (0..boards)
        .into_par_iter()
        .map(|k| {
            let lattice = fixture.lattice(phase(k));
            let clean = fixture.clean(&lattice);
            (lattice, clean)
        })
        .collect()
}

/// [`run_with`] for ChESS with the saddle-point refiner.
fn run(fixture: Fixture, clean: &[(Lattice, Vec<u8>)], known: bool) -> (Whitened, f64, usize) {
    run_with(saddle(), fixture, clean, known)
}

/// Detect with `config` on noisy copies of `clean` boards; `known` supplies
/// the realised noise level instead of estimating it. Also returns the mean
/// estimated σ relative to the realised one, over the boards with an
/// estimate, and the number of boards without one.
fn run_with(
    config: DetectorConfig,
    fixture: Fixture,
    clean: &[(Lattice, Vec<u8>)],
    known: bool,
) -> (Whitened, f64, usize) {
    let mut w = Whitened::default();
    let (mut ratio, mut declined) = (0f64, 0usize);
    for (k, (lattice, clean)) in clean.iter().enumerate() {
        let (img, realised) = fixture.noisy(clean, 1000 + k as u64);
        match estimate_noise_sigma_u8(&img, SIZE, SIZE) {
            Some(est) => ratio += f64::from(est / realised),
            None => declined += 1,
        }
        let cfg = if known {
            config.with_noise_sigma(realised)
        } else {
            config
        };
        let corners = Detector::new(cfg)
            .unwrap()
            .detect_u8(&img, SIZE as u32, SIZE as u32)
            .unwrap();
        w.add(lattice, &corners);
    }
    let estimated = clean.len() - declined;
    (w, ratio / estimated.max(1) as f64, declined)
}

/// The reported covariance — estimated σ through the refiner's noise term,
/// plus its systematic term — is conservative on ideal synthetic corners:
/// the systematic term carries a floor measured on real images, larger than
/// the error these boards have. (The noise term alone is checked against
/// Monte Carlo scatter in `chess-corners-core`.)
#[test]
fn saddle_point_covariance_is_conservative_on_ideal_boards() {
    let fx = Fixture::new(0.8, 2.0);
    let (w, _, declined) = run(fx, &clean_boards(fx, 1), false);
    assert_eq!(declined, 0);
    assert!(w.n > 300, "too few corners with a covariance: {}", w.n);
    assert_eq!(w.unmatched, 0);
    let s0 = w.sigma0_sq();
    assert!(
        (0.03..1.0).contains(&s0),
        "σ̂₀² = {s0:.3} over {} corners",
        w.n
    );
}

/// On a board rotated off the pixel grid, where the re-centred fit keeps a
/// small phase-dependent error, the covariance is still conservative.
#[test]
fn covariance_is_conservative_on_a_rotated_board() {
    let fx = Fixture::new(0.8, 2.0).rotated(30.0);
    let (w, _, declined) = run(fx, &clean_boards(fx, 1), false);
    assert_eq!(declined, 0);
    assert!(w.n > 300, "too few corners with a covariance: {}", w.n);
    assert_eq!(w.unmatched, 0);
    let s0 = w.sigma0_sq();
    assert!(
        (0.03..1.0).contains(&s0),
        "σ̂₀² = {s0:.3} over {} corners",
        w.n
    );
}

/// `Σ = max(σ², 1/12)·C + Σ_sys` for every localizer: affine in the
/// supplied noise variance, and constant below the 8-bit rounding variance.
#[test]
fn covariance_scales_with_the_supplied_noise() {
    let img = standard_image();
    let detect = |cfg: DetectorConfig| {
        Detector::new(cfg)
            .unwrap()
            .detect_u8(img, SIZE as u32, SIZE as u32)
            .unwrap()
    };
    for (name, cfg) in localizers() {
        let [s1, s2, s4] = [1.0f32, 2.0, 4.0].map(|s| detect(cfg.with_noise_sigma(s)));
        assert_eq!(s1.len(), s4.len(), "{name}");
        let mut checked = 0;
        for ((a, b), c) in s1.iter().zip(&s2).zip(&s4) {
            assert_eq!((a.x, a.y), (c.x, c.y), "{name}");
            let (Some(ca), Some(cb), Some(cc)) = (
                a.position_covariance,
                b.position_covariance,
                c.position_covariance,
            ) else {
                assert!(a.position_covariance.is_none() && c.position_covariance.is_none());
                continue;
            };
            // Variances 1, 4, 16: (Σ₁₆ − Σ₁) = 5·(Σ₄ − Σ₁).
            for (va, vb, vc) in [
                (ca.xx, cb.xx, cc.xx),
                (ca.xy, cb.xy, cc.xy),
                (ca.yy, cb.yy, cc.yy),
            ] {
                let scale = cc.xx.max(cc.yy);
                assert!(
                    ((vc - va) - 5.0 * (vb - va)).abs() <= 1e-3 * scale,
                    "{name}: {ca:?} {cb:?} {cc:?}"
                );
            }
            assert!(cc.xx > ca.xx && cc.yy > ca.yy, "{name}");
            checked += 1;
        }
        assert!(checked > 100, "{name}: {checked}");

        // The noise variance is at least the 8-bit rounding variance 1/12,
        // so a lower (or zero) σ gives the same covariance as √(1/12).
        let at_rounding = detect(cfg.with_noise_sigma((1.0f32 / 12.0).sqrt()));
        for sigma in [0.0f32, 0.1] {
            let low = detect(cfg.with_noise_sigma(sigma));
            for (a, b) in low.iter().zip(&at_rounding) {
                assert_eq!(a.position_covariance, b.position_covariance, "{name}");
            }
        }
    }
}

#[test]
fn covariance_is_reported_only_where_valid() {
    let img = standard_image();
    let count = |cfg: DetectorConfig| {
        let corners = Detector::new(cfg)
            .unwrap()
            .detect_u8(img, SIZE as u32, SIZE as u32)
            .unwrap();
        assert!(!corners.is_empty());
        let with = corners
            .iter()
            .filter(|c| c.position_covariance.is_some())
            .count();
        (with, corners.len())
    };
    // Every localizer, single scale or coarse-to-fine.
    for (name, cfg) in localizers() {
        for cfg in [
            cfg,
            cfg.with_multiscale(MultiscaleConfig::pyramid_default()),
        ] {
            let (with, all) = count(cfg);
            assert!(with * 10 >= all * 9, "{name}: {with} of {all}: {cfg:?}");
        }
        // No covariance: the `Off` tier, or an upscaled input.
        for cfg in [
            cfg.with_uncertainty(Uncertainty::Off),
            cfg.with_upscale(UpscaleConfig::fixed(2)),
        ] {
            assert_eq!(count(cfg).0, 0, "{name}: {cfg:?}");
        }
    }
}

/// The tier never moves a corner: positions under `Off` and `Propagated`
/// are identical for every localizer, single scale and coarse-to-fine.
#[test]
fn positions_do_not_depend_on_the_tier() {
    let img = standard_image();
    for (name, cfg) in localizers() {
        for cfg in [
            cfg,
            cfg.with_multiscale(MultiscaleConfig::pyramid_default()),
        ] {
            let detect = |u: Uncertainty| {
                Detector::new(cfg.with_uncertainty(u))
                    .unwrap()
                    .detect_u8(img, SIZE as u32, SIZE as u32)
                    .unwrap()
            };
            let (on, off) = (detect(Uncertainty::Propagated), detect(Uncertainty::Off));
            assert_eq!(on.len(), off.len(), "{name}");
            for (a, b) in on.iter().zip(&off) {
                assert_eq!(
                    (a.x.to_bits(), a.y.to_bits()),
                    (b.x.to_bits(), b.y.to_bits()),
                    "{name}"
                );
                assert!(b.position_covariance.is_none());
            }
        }
    }
}

/// Radon with saddle-point refinement keeps every peak: a corner whose
/// refinement converged has moved off the peak fit, and every other corner
/// keeps its peak-fit position and covariance exactly.
#[test]
fn radon_keeps_the_peak_fit_where_refinement_does_not_converge() {
    let img = standard_image();
    let detect = |cfg: DetectorConfig| {
        Detector::new(cfg)
            .unwrap()
            .detect_u8(img, SIZE as u32, SIZE as u32)
            .unwrap()
    };
    let peaks = detect(DetectorConfig::radon());
    let refined = detect(radon_saddle());
    assert_eq!(peaks.len(), refined.len());
    let (mut moved, mut kept) = (0, 0);
    for (p, r) in peaks.iter().zip(&refined) {
        assert!((p.x - r.x).abs() < 0.5 && (p.y - r.y).abs() < 0.5);
        if (p.x, p.y) == (r.x, r.y) {
            assert_eq!(p.position_covariance, r.position_covariance);
            kept += 1;
        } else {
            assert!(r.position_covariance.is_some());
            moved += 1;
        }
    }
    assert!(moved > 300, "{moved} refined, {kept} kept");
}

/// Radon's saddle-point refinement is conservative like the ChESS one.
#[test]
fn radon_covariance_is_conservative_on_ideal_boards() {
    let fx = Fixture::new(0.8, 2.0);
    let mut w = Whitened::default();
    for (k, (lattice, clean)) in clean_boards(fx, 2).iter().enumerate() {
        let (img, _) = fx.noisy(clean, 2000 + k as u64);
        let corners = Detector::new(radon_saddle())
            .unwrap()
            .detect_u8(&img, SIZE as u32, SIZE as u32)
            .unwrap();
        w.add(lattice, &corners);
    }
    assert!(w.n > 600, "too few corners with a covariance: {}", w.n);
    assert_eq!(w.unmatched, 0);
    let s0 = w.sigma0_sq();
    assert!(
        (0.03..1.0).contains(&s0),
        "σ̂₀² = {s0:.3} over {} corners",
        w.n
    );
}

#[test]
fn roi_covariance_matches_full_frame() {
    let img = standard_image();
    let mut det = Detector::new(saddle()).unwrap();
    let full = det.detect_u8(img, SIZE as u32, SIZE as u32).unwrap();
    let roi = Roi::new(60, 60, 140, 140).unwrap();
    let sub = det
        .detect_u8_roi(img, SIZE as u32, SIZE as u32, roi)
        .unwrap();
    let mut checked = 0;
    for s in &sub {
        let f = full
            .iter()
            .find(|f| (f.x - s.x).abs() < 1e-3 && (f.y - s.y).abs() < 1e-3)
            .expect("ROI corner in full frame");
        let (Some(cs), Some(cf)) = (s.position_covariance, f.position_covariance) else {
            continue;
        };
        assert!((cs.xx - cf.xx).abs() <= 1e-3 * cf.xx);
        checked += 1;
    }
    assert!(checked > 20);
}

#[test]
fn invalid_noise_sigma_is_rejected() {
    for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
        assert!(Detector::new(saddle().with_noise_sigma(bad)).is_err());
    }
    assert!(Detector::new(saddle().with_noise_sigma(0.0)).is_ok());
}

#[test]
#[ignore = "measurement: σ̂₀² of the reported covariance per localizer"]
fn position_covariance_calibration() {
    // (label, cell, rotation°, axis angle°, dark, bright)
    let mut boards: Vec<(String, f32, f32, f32, u8, u8)> = Vec::new();
    for cell in [8.0f32, 12.0] {
        for rot in [0.0f32, 8.0, 45.0] {
            boards.push((format!("rot{rot} c{cell}"), cell, rot, 90.0, 40, 200));
        }
        boards.push((format!("skew60 c{cell}"), cell, 0.0, 60.0, 40, 200));
    }
    let psfs = [0.5f32, 1.5, 2.5];
    let sigmas = [1.0f32, 2.0, 8.0];
    println!("{SIZE}² boards, 5 per row; σ̂₀² with the noise σ estimated / supplied");
    println!("localizer | board | psf | σ | σ̂₀² est | σ̂₀² known | corners | unmatched | no σ̂");
    for (name, cfg) in localizers() {
        for (label, cell, rot, axis, dark, bright) in &boards {
            for &psf in &psfs {
                let base = Fixture {
                    cell: *cell,
                    rotation: rot * DEG,
                    axis_angle: axis * DEG,
                    dark: *dark,
                    bright: *bright,
                    psf,
                    sigma: 0.0,
                };
                let clean = clean_boards(base, 5);
                let rows: Vec<_> = sigmas
                    .par_iter()
                    .map(|&sigma| {
                        let fx = Fixture { sigma, ..base };
                        let est = run_with(cfg, fx, &clean, false);
                        let known = run_with(cfg, fx, &clean, true);
                        (sigma, est, known)
                    })
                    .collect();
                for (sigma, (est, _, declined), (known, _, _)) in rows {
                    println!(
                        "{name} | {label} | {psf} | {sigma} | {:.3} | {:.3} | {} | {} | {declined}",
                        est.sigma0_sq(),
                        known.sigma0_sq(),
                        est.n,
                        est.unmatched,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "measurement: noise estimate and its cost on the test images"]
fn noise_estimate_on_test_images() {
    use std::time::Instant;
    let min_time = |f: &mut dyn FnMut()| {
        (0..20)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    };
    println!("image | size | σ̂ (gray levels) | estimate ms | saddle-point detect ms | corners with covariance");
    for name in ["small", "mid", "large"] {
        let path = format!("{}/../../testimages/{name}.png", env!("CARGO_MANIFEST_DIR"));
        let img = image::open(&path).expect("test image").to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let sigma = estimate_noise_sigma_u8(img.as_raw(), w, h);
        let est_ms = min_time(&mut || {
            std::hint::black_box(estimate_noise_sigma_u8(img.as_raw(), w, h));
        });
        let mut det =
            Detector::new(saddle().with_multiscale(MultiscaleConfig::pyramid_default())).unwrap();
        let corners = det.detect(&img).unwrap();
        let det_ms = min_time(&mut || {
            std::hint::black_box(det.detect(&img).unwrap());
        });
        let with = corners
            .iter()
            .filter(|c| c.position_covariance.is_some())
            .count();
        println!(
            "{name} | {w}×{h} | {sigma:?} | {est_ms:.3} | {det_ms:.3} | {with}/{}",
            corners.len()
        );
    }
}

#[test]
#[ignore = "measurement: additivity of the noise estimate on the test images"]
fn noise_estimate_additivity() {
    // Adding white noise σₐ to an image adds σₐ² + 1/12 (rounding) to the
    // noise variance the estimator sees: σ̂²(img + n) − σ̂²(img) should
    // match it whatever the image's own noise is.
    println!("image | σ̂(img) | σₐ | σ̂(img + n) | (σ̂² − σ̂₀²) / (σₐ² + 1/12)");
    for name in ["small", "mid", "large"] {
        let path = format!("{}/../../testimages/{name}.png", env!("CARGO_MANIFEST_DIR"));
        let img = image::open(&path).expect("test image").to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let base = estimate_noise_sigma_u8(img.as_raw(), w, h).map_or(f64::NAN, f64::from);
        for sigma_a in [1.0f32, 2.0, 4.0] {
            let mut ratios = Vec::new();
            let mut sigmas = Vec::new();
            for seed in 0..3 {
                let mut noisy = img.as_raw().clone();
                add_gaussian_noise(&mut noisy, sigma_a, 500 + seed);
                let s = estimate_noise_sigma_u8(&noisy, w, h).map_or(f64::NAN, f64::from);
                let added = f64::from(sigma_a * sigma_a) + 1.0 / 12.0;
                sigmas.push(s);
                ratios.push((s * s - base * base) / added);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "{name} | {base:.3} | {sigma_a} | {:.3} | {:.3}",
                mean(&sigmas),
                mean(&ratios)
            );
        }
    }
}
