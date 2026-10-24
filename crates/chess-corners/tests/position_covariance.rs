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
    UpscaleConfig,
};
use chess_corners_core::estimate_noise_sigma_u8;
use chess_corners_testutil::{add_gaussian_noise, optical_chessboard, Lattice};
use rayon::prelude::*;

const SIZE: usize = 200;
/// Model-error floor the detector adds per axis, px.
const FLOOR: f64 = 0.006;
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

/// Model-error floors (px per axis) compared by the measurement; the
/// reported covariance carries [`FLOOR`].
const CANDIDATE_FLOORS: [f64; 10] = [
    0.0, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.015,
];

/// Whitening statistics over a set of boards.
#[derive(Default)]
struct Whitened {
    /// Σ eᵀΣ⁻¹e with the floor replaced by each [`CANDIDATE_FLOORS`] entry.
    sum_candidates: [f64; CANDIDATE_FLOORS.len()],
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

/// Detect on noisy copies of `clean` boards; `known` supplies the realised
/// noise level instead of estimating it. Also returns the mean estimated σ
/// relative to the realised one, over the boards with an estimate, and the
/// number of boards without one.
fn run(fixture: Fixture, clean: &[(Lattice, Vec<u8>)], known: bool) -> (Whitened, f64, usize) {
    let mut w = Whitened::default();
    let (mut ratio, mut declined) = (0f64, 0usize);
    for (k, (lattice, clean)) in clean.iter().enumerate() {
        let (img, realised) = fixture.noisy(clean, 1000 + k as u64);
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
        w.add(lattice, &corners);
    }
    let estimated = clean.len() - declined;
    (w, ratio / estimated.max(1) as f64, declined)
}

/// The noise part of the reported covariance — estimated σ times the
/// refiner's per-noise-variance covariance — is calibrated end to end, and
/// the model-error floor only makes it more conservative.
#[test]
fn saddle_point_covariance_is_calibrated() {
    let fx = Fixture::new(0.8, 2.0);
    let (w, _, declined) = run(fx, &clean_boards(fx, 1), false);
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

/// On a board rotated off the pixel grid, where the re-centred fit keeps a
/// small phase-dependent error, the floor keeps the covariance calibrated.
#[test]
fn covariance_is_calibrated_on_a_rotated_board() {
    let fx = Fixture::new(0.8, 2.0).rotated(30.0);
    let (w, _, declined) = run(fx, &clean_boards(fx, 1), false);
    assert_eq!(declined, 0);
    assert!(w.n > 300, "too few corners with a covariance: {}", w.n);
    assert_eq!(w.unmatched, 0);
    let s0 = w.sigma0_sq();
    assert!(
        (0.75..1.25).contains(&s0),
        "σ̂₀² = {s0:.3} over {} corners",
        w.n
    );
}

#[test]
fn covariance_scales_with_the_supplied_noise() {
    let img = standard_image();
    let detect = |cfg: DetectorConfig| {
        Detector::new(cfg)
            .unwrap()
            .detect_u8(img, SIZE as u32, SIZE as u32)
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
    // Saddle point on the input image, single scale or coarse-to-fine,
    // after either detector.
    for cfg in [
        saddle(),
        saddle().with_multiscale(MultiscaleConfig::pyramid_default()),
        radon_saddle(),
        radon_saddle().with_multiscale(MultiscaleConfig::pyramid_default()),
    ] {
        let (with, all) = count(cfg);
        assert!(with * 10 >= all * 9, "{with} of {all}: {cfg:?}");
    }
    // No covariance: other refiners, Radon's peak fit (its default), or an
    // upscaled input.
    for cfg in [
        DetectorConfig::chess(),
        DetectorConfig::chess().with_chess(|c| c.refiner = ChessRefiner::forstner()),
        DetectorConfig::radon(),
        saddle().with_upscale(UpscaleConfig::fixed(2)),
        radon_saddle().with_upscale(UpscaleConfig::fixed(2)),
    ] {
        assert_eq!(count(cfg).0, 0, "{cfg:?}");
    }
}

/// Radon with saddle-point refinement keeps every peak: a corner whose
/// refinement carries a covariance has moved off the peak fit, and every
/// other corner keeps its peak-fit position exactly.
#[test]
fn radon_keeps_the_peak_fit_where_refinement_has_no_covariance() {
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
        if r.position_covariance.is_some() {
            moved += 1;
        } else {
            assert_eq!((p.x, p.y), (r.x, r.y));
            kept += 1;
        }
    }
    assert!(moved > 300, "{moved} refined, {kept} kept");
}

/// Radon's saddle-point refinement is calibrated like the ChESS one.
#[test]
fn radon_covariance_is_calibrated() {
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
        (0.75..1.25).contains(&s0),
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

/// One measured row: its label and the whitening statistics with the
/// estimated and the supplied noise level.
struct Measured {
    label: String,
    est: Whitened,
    known: Whitened,
    ratio: f64,
    declined: usize,
}

#[test]
#[ignore = "measurement: σ̂₀² of the reported covariance and the floor minimax"]
fn position_covariance_calibration() {
    // (label, cell, rotation°, axis angle°, dark, bright)
    let mut boards: Vec<(String, f32, f32, f32, u8, u8)> = Vec::new();
    for cell in [8.0f32, 12.0] {
        for rot in [0.0f32, 2.0, 5.0, 8.0, 15.0, 30.0, 45.0] {
            boards.push((format!("rot{rot} c{cell}"), cell, rot, 90.0, 40, 200));
        }
        boards.push((format!("skew60 c{cell}"), cell, 0.0, 60.0, 40, 200));
    }
    boards.push(("axis 90/150 c8".into(), 8.0, 0.0, 90.0, 90, 150));
    let psfs = [0.5f32, 0.8, 1.5, 2.5];
    let sigmas = [1.0f32, 2.0, 4.0, 8.0];

    let mut rows: Vec<Measured> = Vec::new();
    println!("{SIZE}² boards, 25 per row (5×5 phases); σ̂₀² with the {FLOOR} px floor");
    println!("board | psf | σ | σ̂/σ | σ̂₀² est | σ̂₀² known | corners | unmatched | no σ̂ | noise-free RMS px/axis");
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
            let clean = clean_boards(base, 25);
            // Noise-free error of the converged corners (any σ supplied
            // gives them a covariance, which marks convergence).
            let mut sq = (0.0f64, 0usize);
            for (lattice, img) in &clean {
                let corners = Detector::new(saddle().with_noise_sigma(1.0))
                    .unwrap()
                    .detect_u8(img, SIZE as u32, SIZE as u32)
                    .unwrap();
                for c in corners.iter().filter(|c| c.position_covariance.is_some()) {
                    let (tx, ty) = lattice.nearest_corner(c.x, c.y);
                    let (ex, ey) = (f64::from(c.x - tx), f64::from(c.y - ty));
                    if ex.abs() < 0.5 && ey.abs() < 0.5 {
                        sq = (sq.0 + ex * ex + ey * ey, sq.1 + 1);
                    }
                }
            }
            let rms = (sq.0 / (2.0 * sq.1.max(1) as f64)).sqrt();
            let measured: Vec<Measured> = sigmas
                .par_iter()
                .map(|&sigma| {
                    let fx = Fixture { sigma, ..base };
                    let (est, ratio, declined) = run(fx, &clean, false);
                    let (known, _, _) = run(fx, &clean, true);
                    Measured {
                        label: format!("{label} psf {psf} σ {sigma}"),
                        est,
                        known,
                        ratio,
                        declined,
                    }
                })
                .collect();
            for (m, sigma) in measured.into_iter().zip(sigmas) {
                println!(
                    "{label} | {psf} | {sigma} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {rms:.4}",
                    m.ratio,
                    m.est.sigma0_sq(),
                    m.known.sigma0_sq(),
                    m.est.n,
                    m.est.unmatched,
                    m.declined
                );
                rows.push(m);
            }
        }
    }

    // Per candidate floor: worst |ln σ̂₀²| (estimated σ) over the rows whose
    // detections match the truth (≥ 1 000 matched, ≤ 1% unmatched), over
    // all of them and over the grid-aligned and rotated ones separately.
    let clean = |m: &Measured| m.est.n >= 1000 && m.est.unmatched * 100 <= m.est.n;
    // Rotated boards are labelled `rot<deg>`; everything else (`rot0`,
    // `skew60`, `axis 90/150`) has an axis on the pixel grid.
    let rotated = |l: &str| l.starts_with("rot") && !l.starts_with("rot0 ");
    let groups = [
        ("all", None),
        ("grid-aligned", Some(false)),
        ("rotated", Some(true)),
    ];
    println!("\nf (px) | worst |ln σ̂₀²| (σ̂₀², row) per group; [min, max] σ̂₀²");
    for (k, f) in CANDIDATE_FLOORS.iter().enumerate() {
        let mut line = format!("{f}");
        for (name, want) in groups {
            let vals: Vec<(f64, &str)> = rows
                .iter()
                .filter(|m| clean(m) && want.is_none_or(|w| rotated(&m.label) == w))
                .map(|m| (m.est.sigma0_sq_with(k), m.label.as_str()))
                .collect();
            let worst = vals
                .iter()
                .max_by(|a, b| a.0.ln().abs().total_cmp(&b.0.ln().abs()))
                .unwrap();
            let lo = vals.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
            let hi = vals.iter().map(|v| v.0).fold(0.0, f64::max);
            line += &format!(
                " | {name}: {:.3} ({:.3}, {}) [{lo:.2}, {hi:.2}]",
                worst.0.ln().abs(),
                worst.0,
                worst.1
            );
        }
        println!("{line}");
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
