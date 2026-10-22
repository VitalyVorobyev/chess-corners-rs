//! Position covariance on `CornerDescriptor`, end to end through the
//! facade: which configurations report it, and whether it is calibrated.
//!
//! Calibration is checked the way a downstream calibration uses it: whiten
//! each matched corner's error against the truth with its reported px²
//! covariance and average, `σ̂₀² = mean(eᵀ Σ⁻¹ e) / 2`, which is 1 when the
//! covariance is right. The boards are exact pixel-area renders, so the
//! corner positions are known exactly.

use chess_corners::{
    ChessRefiner, CornerDescriptor, Detector, DetectorConfig, MultiscaleConfig, Roi, UpscaleConfig,
};
use chess_corners_core::estimate_noise_sigma_u8;
use chess_corners_testutil::{add_gaussian_noise, area_chessboard, gaussian_blur};

const PHASE: (f32, f32) = (0.3, 0.6);

const SIZE: usize = 200;
const CELL: f32 = 8.0;
/// Model-error floor the detector adds per axis, px.
const FLOOR: f64 = 0.002;

/// Board geometry and contrast.
#[derive(Clone, Copy, Debug)]
struct Fixture {
    /// Angle of the second lattice axis (π/2: axis-aligned).
    axis_angle: f32,
    dark: u8,
    bright: u8,
    blur: f32,
    sigma: f32,
}

impl Fixture {
    fn new(blur: f32, sigma: f32) -> Self {
        Self {
            axis_angle: core::f32::consts::FRAC_PI_2,
            dark: 40,
            bright: 200,
            blur,
            sigma,
        }
    }

    /// Exact-area board, blurred, with white noise. Returns the image, its
    /// corner offset and the realised noise σ (`noisy − clean`, rounding
    /// included).
    fn render(&self, phase: (f32, f32), seed: u64) -> (Vec<u8>, (f32, f32), f32) {
        let offset = (3.0 + phase.0, 3.0 + phase.1);
        let mut clean = area_chessboard(
            SIZE,
            CELL as usize,
            offset,
            self.axis_angle,
            self.dark,
            self.bright,
        );
        gaussian_blur(&mut clean, SIZE, self.blur);
        let mut img = clean.clone();
        add_gaussian_noise(&mut img, self.sigma, seed);
        let n = img.len() as f64;
        let ss: f64 = img
            .iter()
            .zip(&clean)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        (img, offset, (ss / n).sqrt() as f32)
    }

    /// Error of `c` against the nearest lattice corner of `offset`.
    fn error(&self, c: &CornerDescriptor, offset: (f32, f32)) -> (f32, f32) {
        let (sin, cos) = self.axis_angle.sin_cos();
        let m = ((c.y - offset.1) / (CELL * sin)).round();
        let k = ((c.x - offset.0 - m * CELL * cos) / CELL).round();
        let tx = offset.0 + CELL * (k + m * cos);
        let ty = offset.1 + CELL * m * sin;
        (c.x - tx, c.y - ty)
    }
}

fn saddle() -> DetectorConfig {
    DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::saddle_point())
}

/// Model-error floors (px per axis) compared by the measurement; the
/// reported covariance carries [`FLOOR`].
const CANDIDATE_FLOORS: [f64; 7] = [0.0, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015];

/// Whitening statistics over a set of boards.
#[derive(Default)]
struct Whitened {
    /// Σ eᵀΣ⁻¹e with the reported covariance.
    sum: f64,
    /// Σ eᵀΣ⁻¹e with the floor replaced by each [`CANDIDATE_FLOORS`] entry.
    sum_candidates: [f64; CANDIDATE_FLOORS.len()],
    /// Corners with a covariance, matched to the truth.
    n: usize,
    /// Detections farther than 0.5 px from any lattice corner.
    unmatched: usize,
}

impl Whitened {
    fn add(&mut self, fixture: &Fixture, corners: &[CornerDescriptor], offset: (f32, f32)) {
        for c in corners {
            let Some(cov) = c.position_covariance else {
                continue;
            };
            let (ex, ey) = fixture.error(c, offset);
            if ex.abs() >= 0.5 || ey.abs() >= 0.5 {
                self.unmatched += 1;
                continue;
            }
            let (ex, ey) = (f64::from(ex), f64::from(ey));
            // Whitened squared error with `df2` added to the diagonal.
            let quad = |df2: f64| {
                let (a, b, d) = (
                    f64::from(cov.xx) + df2,
                    f64::from(cov.xy),
                    f64::from(cov.yy) + df2,
                );
                (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / (a * d - b * b)
            };
            self.sum += quad(0.0);
            for (acc, f) in self.sum_candidates.iter_mut().zip(CANDIDATE_FLOORS) {
                *acc += quad(f * f - FLOOR * FLOOR);
            }
            self.n += 1;
        }
    }

    /// σ̂₀² with the reported covariance.
    fn sigma0_sq(&self) -> f64 {
        self.sum / (2.0 * self.n as f64)
    }

    /// σ̂₀² with the floor replaced by `CANDIDATE_FLOORS[k]`.
    fn sigma0_sq_with(&self, k: usize) -> f64 {
        self.sum_candidates[k] / (2.0 * self.n as f64)
    }
}

/// Detect on `boards` renders of `fixture`; `known` supplies the realised
/// noise level instead of estimating it. Also returns the mean estimated σ
/// relative to the realised one, over the boards with an estimate, and the
/// number of boards without one.
fn run(fixture: Fixture, boards: usize, known: bool) -> (Whitened, f64, usize) {
    let mut w = Whitened::default();
    let (mut ratio, mut declined) = (0f64, 0usize);
    for k in 0..boards {
        let phase = ((k % 5) as f32 / 5.0, (k / 5 % 5) as f32 / 5.0);
        let (img, offset, realised) = fixture.render(phase, 1000 + k as u64);
        match estimate_noise_sigma_u8(&img, SIZE, SIZE) {
            Some(est) => ratio += f64::from(est / realised),
            None => declined += 1,
        }
        let cfg = if known {
            saddle().with_noise_sigma(realised)
        } else {
            saddle()
        };
        let corners = Detector::new(cfg)
            .unwrap()
            .detect_u8(&img, SIZE as u32, SIZE as u32)
            .unwrap();
        w.add(&fixture, &corners, offset);
    }
    let estimated = boards - declined;
    (w, ratio / estimated.max(1) as f64, declined)
}

/// The noise part of the reported covariance — estimated σ times the
/// refiner's per-noise-variance covariance — is calibrated end to end, and
/// the model-error floor only makes it more conservative.
#[test]
fn saddle_point_covariance_is_calibrated() {
    let (w, _, declined) = run(Fixture::new(0.8, 2.0), 1, false);
    assert_eq!(declined, 0);
    let (with_floor, noise_only) = (w.sigma0_sq(), w.sigma0_sq_with(0));
    assert!(w.n > 300, "too few corners with a covariance: {}", w.n);
    assert_eq!(w.unmatched, 0);
    assert!(
        (0.8..1.25).contains(&noise_only),
        "noise-only σ̂₀² = {noise_only:.3} over {} corners",
        w.n
    );
    assert!(with_floor < noise_only);
}

#[test]
fn covariance_scales_with_the_supplied_noise() {
    let (img, _, _) = Fixture::new(0.8, 2.0).render(PHASE, 7);
    let detect = |cfg: DetectorConfig| {
        Detector::new(cfg)
            .unwrap()
            .detect_u8(&img, SIZE as u32, SIZE as u32)
            .unwrap()
    };
    let lo = detect(saddle().with_noise_sigma(1.0));
    let hi = detect(saddle().with_noise_sigma(4.0));
    assert_eq!(lo.len(), hi.len());
    let floor2 = (FLOOR * FLOOR) as f32;
    let mut checked = 0;
    for (a, b) in lo.iter().zip(&hi) {
        let (Some(ca), Some(cb)) = (a.position_covariance, b.position_covariance) else {
            assert!(a.position_covariance.is_none() && b.position_covariance.is_none());
            continue;
        };
        // (σ²·C + f²) at σ = 4 is 16× the noise term at σ = 1, plus the floor.
        let expect_xx = 16.0 * (ca.xx - floor2) + floor2;
        assert!(
            (cb.xx - expect_xx).abs() <= 1e-4 * expect_xx,
            "{ca:?} {cb:?}"
        );
        assert!((cb.xy - 16.0 * ca.xy).abs() <= 1e-4 * cb.xx);
        checked += 1;
    }
    assert!(checked > 100);

    // The noise variance is at least the 8-bit rounding variance 1/12, so a
    // lower (or zero) σ gives the same covariance as √(1/12).
    let at_rounding = detect(saddle().with_noise_sigma((1.0f32 / 12.0).sqrt()));
    for sigma in [0.0f32, 0.1] {
        let low = detect(saddle().with_noise_sigma(sigma));
        for (a, b) in low.iter().zip(&at_rounding) {
            assert_eq!(a.position_covariance, b.position_covariance);
        }
    }
}

#[test]
fn covariance_is_reported_only_where_valid() {
    let (img, _, _) = Fixture::new(0.8, 2.0).render(PHASE, 7);
    let count = |cfg: DetectorConfig| {
        let corners = Detector::new(cfg)
            .unwrap()
            .detect_u8(&img, SIZE as u32, SIZE as u32)
            .unwrap();
        assert!(!corners.is_empty());
        let with = corners
            .iter()
            .filter(|c| c.position_covariance.is_some())
            .count();
        (with, corners.len())
    };
    // Saddle point on the input image, single scale or coarse-to-fine.
    for cfg in [
        saddle(),
        saddle().with_multiscale(MultiscaleConfig::pyramid_default()),
    ] {
        let (with, all) = count(cfg);
        assert!(with * 10 >= all * 9, "{with} of {all}");
    }
    // No covariance: other refiners, Radon, or an upscaled input.
    for cfg in [
        DetectorConfig::chess(),
        DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::forstner()),
        DetectorConfig::radon(),
        saddle().with_upscale(UpscaleConfig::fixed(2)),
    ] {
        assert_eq!(count(cfg).0, 0, "{cfg:?}");
    }
}

#[test]
fn roi_covariance_matches_full_frame() {
    let (img, _, _) = Fixture::new(0.8, 2.0).render(PHASE, 7);
    let mut det = Detector::new(saddle()).unwrap();
    let full = det.detect_u8(&img, SIZE as u32, SIZE as u32).unwrap();
    let roi = Roi::new(60, 60, 140, 140).unwrap();
    let sub = det
        .detect_u8_roi(&img, SIZE as u32, SIZE as u32, roi)
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
#[ignore = "measurement: prints σ̂₀² of the reported covariance"]
fn position_covariance_calibration() {
    println!("cell {CELL}, 25 boards of {SIZE}² per row; σ̂₀² with the {FLOOR} px floor");
    println!("board | blur | σ | σ̂/σ | estimated σ | supplied σ | corners | unmatched | no σ̂");
    let boards: [(&str, f32, u8, u8); 3] = [
        ("axis 40/200", core::f32::consts::FRAC_PI_2, 40, 200),
        ("axis 90/150", core::f32::consts::FRAC_PI_2, 90, 150),
        ("skew60 40/200", core::f32::consts::FRAC_PI_3, 40, 200),
    ];
    // Per candidate floor: worst |ln σ̂₀²| (estimated σ), its σ̂₀² and the
    // row, over rows with at least 1 000 matched corners, and over those
    // whose detections also match the truth (≤ 1% unmatched).
    let mut worst: [[(f64, f64, String); 2]; CANDIDATE_FLOORS.len()] =
        std::array::from_fn(|_| std::array::from_fn(|_| (0.0, 1.0, String::new())));
    for (label, axis_angle, dark, bright) in boards {
        for &blur in &[0.0f32, 0.8, 1.5, 2.5] {
            for &sigma in &[1.0f32, 2.0, 4.0, 8.0] {
                let fixture = Fixture {
                    axis_angle,
                    dark,
                    bright,
                    ..Fixture::new(blur, sigma)
                };
                let (est, ratio, declined) = run(fixture, 25, false);
                let (known, _, _) = run(fixture, 25, true);
                println!(
                    "{label} | {blur} | {sigma} | {ratio:.3} | {:.3} | {:.3} | {} | {} | {declined}",
                    est.sigma0_sq(),
                    known.sigma0_sq(),
                    est.n,
                    est.unmatched
                );
                if est.n < 1000 {
                    continue;
                }
                let clean = est.unmatched * 100 <= est.n;
                for (k, ws) in worst.iter_mut().enumerate() {
                    let v = est.sigma0_sq_with(k);
                    for (w, include) in ws.iter_mut().zip([true, clean]) {
                        if include && v.ln().abs() > w.0 {
                            *w = (v.ln().abs(), v, format!("{label} blur {blur} σ {sigma}"));
                        }
                    }
                }
            }
        }
    }
    println!("\nf (px) | worst |ln σ̂₀²| (σ̂₀²), all | row | ≤ 1% unmatched | row");
    for (f, [(l, v, row), (lc, vc, rowc)]) in CANDIDATE_FLOORS.iter().zip(&worst) {
        println!("{f} | {l:.3} ({v:.3}) | {row} | {lc:.3} ({vc:.3}) | {rowc}");
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
