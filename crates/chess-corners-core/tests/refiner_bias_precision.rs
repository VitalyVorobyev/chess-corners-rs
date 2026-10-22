//! Bias / precision split of the built-in refiners on synthetic corners
//! with exact ground truth.
//!
//! A refiner's position error has two parts. *Bias* is the deterministic
//! error of the noise-free estimate (the model does not match the corner,
//! the window is asymmetric about it). *Precision* is the scatter that pixel
//! noise adds around that biased estimate. A noise-only covariance (the
//! saddle-point refiner's `covariance_per_noise_variance`) describes the
//! second part only. This sweep measures both, per refiner and condition,
//! and checks the predicted saddle-point σ against the scatter about the
//! per-phase mean (`z_prec`) and against the truth (`z_total`), and the
//! constant model-error floor that brings the latter to the former.
//!
//! Fixtures are [`area_chessboard`] boards (exact pixel-area rendering, so
//! the corner sits exactly on the lattice), axis-aligned or sheared to a
//! 60° axis angle, optionally blurred, then corrupted with white or
//! signal-dependent Gaussian noise. Seeds are the truth rounded to the
//! nearest pixel. The CenterOfMass refiner reads the ChESS response of the
//! same noisy image.
//!
//! Not a gate. Writes `out/uq01/refiner_accuracy.json` at the workspace
//! root and prints summary tables:
//!
//! ```sh
//! cargo test -p chess-corners-core --release --test refiner_bias_precision \
//!     -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;

use chess_corners_core::{
    chess_response_u8, CenterOfMassConfig, CenterOfMassRefiner, ChessParams, CornerRefiner,
    ForstnerConfig, ForstnerRefiner, ImageView, RefineContext, RefineResult, RefineStatus,
    SaddlePointConfig, SaddlePointRefiner,
};
use chess_corners_testutil::{
    aa_chessboard, add_gaussian_noise, add_signal_dependent_noise, area_chessboard, gaussian_blur,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Grid.

const SIZE: usize = 41;
/// Integer part of the corner position; the subpixel phase is added to it.
const CENTER: f32 = 20.0;
/// Subpixel phases per axis: `k / PHASES` for `k = 0..PHASES`.
const PHASES: usize = 5;
/// Noise realizations per (condition, phase). The pooled σ of a cell rests
/// on `PHASES² · (DRAWS − 1)` degrees of freedom; one phase's σ on
/// `DRAWS − 1` (relative standard error `1/sqrt(2·199) = 5%`).
const DRAWS: usize = 200;
const BLURS: [f32; 4] = [0.0, 0.8, 1.5, 2.5];
const CELLS: [usize; 3] = [5, 8, 12];
const CONTRASTS: [(u8, u8); 2] = [(40, 200), (90, 150)];
/// `(name, angle between the lattice axes in degrees)`.
const GEOMETRIES: [(&str, f32); 2] = [("axis", 90.0), ("skew60", 60.0)];
/// Candidate model-error floors `f` (px per axis): σ̂₀² is also evaluated
/// with `Σ + f²·I` to find the floor that absorbs the residual bias.
const FLOORS: [f64; 8] = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05];
/// Variance of the 8-bit rounding applied after every degradation.
const QUANT_VAR: f64 = 1.0 / 12.0;

/// Signal-dependent noise variance `a + b·I`: `b = 8/255` puts 1 gray
/// level² at black and 9 at white (`SD1`); `SD4` is four times that.
const SD_SLOPE: f32 = 8.0 / 255.0;

#[derive(Clone, Copy, Debug)]
enum Noise {
    White(f32),
    Signal { a: f32, b: f32 },
}

const NOISES: [Noise; 7] = [
    Noise::White(0.0),
    Noise::White(1.0),
    Noise::White(2.0),
    Noise::White(4.0),
    Noise::White(8.0),
    Noise::Signal {
        a: 1.0,
        b: SD_SLOPE,
    },
    Noise::Signal {
        a: 4.0,
        b: 4.0 * SD_SLOPE,
    },
];

impl Noise {
    fn is_zero(self) -> bool {
        matches!(self, Noise::White(s) if s <= 0.0)
    }
    fn label(self) -> String {
        match self {
            Noise::White(s) => format!("white {s}"),
            Noise::Signal { a, .. } => format!("sd{a}"),
        }
    }
    fn json(self) -> Value {
        match self {
            Noise::White(s) => json!({ "kind": "white", "sigma": s }),
            Noise::Signal { a, b } => json!({ "kind": "signal", "a": a, "b": b }),
        }
    }
    fn apply(self, img: &mut [u8], seed: u64) {
        match self {
            Noise::White(s) => add_gaussian_noise(img, s, seed),
            Noise::Signal { a, b } => add_signal_dependent_noise(img, a, b, seed),
        }
    }
    /// True per-pixel noise variance (gray levels²) for the saddle-point
    /// prediction: white noise plus rounding, or for signal-dependent noise
    /// the mean of `a + b·I` over the 5×5 fit patch of the clean image.
    fn patch_variance(self, clean: &[u8], seed: (i32, i32)) -> f64 {
        match self {
            Noise::White(s) => f64::from(s * s) + QUANT_VAR,
            Noise::Signal { a, b } => {
                let mut sum = 0.0;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let i = clean[(seed.1 + dy) as usize * SIZE + (seed.0 + dx) as usize];
                        sum += f64::from((a + b * f32::from(i)).max(0.0));
                    }
                }
                sum / 25.0 + QUANT_VAR
            }
        }
    }
}

const REFINERS: [&str; 3] = ["CenterOfMass", "Forstner", "SaddlePoint"];
const SADDLE: usize = 2;

/// One clean fixture: everything but the phase and the noise.
#[derive(Clone, Copy, Debug)]
struct Fixture {
    geometry: usize,
    cell: usize,
    contrast: usize,
    blur: usize,
}

// ---------------------------------------------------------------------------
// Per-phase accumulation (all draws of one phase live in one work item).

/// Statistics of one refiner at one subpixel phase under one noise model.
#[derive(Clone, Debug, Default)]
struct PhaseStats {
    n_total: usize,
    n_acc: usize,
    /// Accepted results exactly at the seed (no refinement happened).
    n_at_seed: usize,
    /// Mean error vector (px) over accepted draws.
    bias: [f64; 2],
    /// Σ (e − bias)² per axis.
    ss: [f64; 2],
    /// Σ |e|².
    sum_sq_err: f64,
    /// Σ |e|.
    sum_abs_err: f64,
    z: Option<ZStats>,
}

/// Saddle-point prediction checks at one phase. `σ_pred` is
/// `sqrt(v · covariance_per_noise_variance)` with `v` the true per-pixel
/// noise variance.
#[derive(Clone, Debug, Default)]
struct ZStats {
    /// Σ ((e − bias)/σ_pred)² per axis.
    zp2: [f64; 2],
    /// Σ (e/σ_pred)² per axis.
    zt2: [f64; 2],
    /// Σ (e − bias)ᵀ Σ⁻¹ (e − bias).
    chi_prec: f64,
    /// Σ eᵀ Σ⁻¹ e.
    chi_total: f64,
    /// Σ eᵀ (Σ + f²·I)⁻¹ e per [`FLOORS`] entry.
    chi_floor: [f64; FLOORS.len()],
    /// Mean predicted radial σ, `sqrt(trace Σ)`.
    sigma_pred: f64,
}

struct Sample {
    err: [f64; 2],
    at_seed: bool,
    /// Predicted position covariance (px²) `[xx, xy, yy]`, saddle only.
    cov: Option<[f64; 3]>,
}

fn phase_stats(total: usize, samples: &[Sample]) -> PhaseStats {
    let n = samples.len();
    let mut st = PhaseStats {
        n_total: total,
        n_acc: n,
        ..Default::default()
    };
    if n == 0 {
        return st;
    }
    for s in samples {
        st.n_at_seed += usize::from(s.at_seed);
        st.bias[0] += s.err[0] / n as f64;
        st.bias[1] += s.err[1] / n as f64;
        st.sum_sq_err += s.err[0].powi(2) + s.err[1].powi(2);
        st.sum_abs_err += s.err[0].hypot(s.err[1]);
    }
    for s in samples {
        for k in 0..2 {
            st.ss[k] += (s.err[k] - st.bias[k]).powi(2);
        }
    }
    if samples.iter().all(|s| s.cov.is_some()) {
        let mut z = ZStats::default();
        for s in samples {
            let [xx, xy, yy] = s.cov.unwrap();
            let d = [s.err[0] - st.bias[0], s.err[1] - st.bias[1]];
            let det = xx * yy - xy * xy;
            let maha =
                |v: [f64; 2]| (yy * v[0] * v[0] - 2.0 * xy * v[0] * v[1] + xx * v[1] * v[1]) / det;
            z.zp2[0] += d[0] * d[0] / xx;
            z.zp2[1] += d[1] * d[1] / yy;
            z.zt2[0] += s.err[0].powi(2) / xx;
            z.zt2[1] += s.err[1].powi(2) / yy;
            z.chi_prec += maha(d);
            z.chi_total += maha(s.err);
            for (chi, f) in z.chi_floor.iter_mut().zip(FLOORS) {
                let (fx, fy) = (xx + f * f, yy + f * f);
                let e = s.err;
                *chi += (fy * e[0] * e[0] - 2.0 * xy * e[0] * e[1] + fx * e[1] * e[1])
                    / (fx * fy - xy * xy);
            }
            z.sigma_pred += (xx + yy).sqrt() / n as f64;
        }
        st.z = Some(z);
    }
    st
}

fn accepted<R: CornerRefiner>(
    refiner: &mut R,
    seed: [f32; 2],
    ctx: RefineContext<'_>,
) -> Option<RefineResult> {
    let res = refiner.refine(seed, ctx);
    (res.status == RefineStatus::Accepted).then_some(res)
}

/// The three refiners at their default configs.
struct Refiners {
    center: CenterOfMassRefiner,
    forstner: ForstnerRefiner,
    saddle: SaddlePointRefiner,
}

impl Refiners {
    fn new() -> Self {
        Self {
            center: CenterOfMassRefiner::new(CenterOfMassConfig::default()),
            forstner: ForstnerRefiner::new(ForstnerConfig::default()),
            saddle: SaddlePointRefiner::new(SaddlePointConfig::default()),
        }
    }

    /// Accepted results of each refiner on `img`, seeded at `seed`.
    fn run(&mut self, img: &[u8], seed: [f32; 2]) -> [Option<RefineResult>; 3] {
        let view = ImageView::from_u8_slice(SIZE, SIZE, img).unwrap();
        let response = chess_response_u8(img, SIZE, SIZE, &ChessParams::default());
        let image_ctx = RefineContext::new(Some(view), None);
        let resp_ctx = RefineContext::new(None, Some(&response));
        [
            accepted(&mut self.center, seed, resp_ctx),
            accepted(&mut self.forstner, seed, image_ctx),
            accepted(&mut self.saddle, seed, image_ctx),
        ]
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// True corner position at subpixel phase `(kx, ky)`.
fn truth_of(phase: (usize, usize)) -> (f32, f32) {
    (
        CENTER + phase.0 as f32 / PHASES as f32,
        CENTER + phase.1 as f32 / PHASES as f32,
    )
}

/// Phase index `ky·PHASES + kx` to `(kx, ky)`.
fn phase_of(index: usize) -> (usize, usize) {
    (index % PHASES, index / PHASES)
}

fn render(fx: Fixture, phase: (usize, usize)) -> (Vec<u8>, (f32, f32)) {
    let truth = truth_of(phase);
    let (dark, bright) = CONTRASTS[fx.contrast];
    let angle = GEOMETRIES[fx.geometry].1.to_radians();
    let mut img = area_chessboard(SIZE, CELLS[fx.cell], truth, angle, dark, bright);
    gaussian_blur(&mut img, SIZE, BLURS[fx.blur]);
    (img, truth)
}

/// Every refiner × noise model at one fixture and phase:
/// `out[noise][refiner]`.
fn run_item(item: usize, fx: Fixture, phase: (usize, usize)) -> Vec<[PhaseStats; 3]> {
    let (clean, truth) = render(fx, phase);
    let seed = [truth.0.round(), truth.1.round()];
    let seed_px = (seed[0] as i32, seed[1] as i32);
    let mut refiners = Refiners::new();
    let mut img = clean.clone();
    NOISES
        .iter()
        .enumerate()
        .map(|(ni, &noise)| {
            let draws = if noise.is_zero() { 1 } else { DRAWS };
            let var = (!noise.is_zero()).then(|| noise.patch_variance(&clean, seed_px));
            let mut samples: [Vec<Sample>; 3] = Default::default();
            for d in 0..draws {
                img.copy_from_slice(&clean);
                noise.apply(
                    &mut img,
                    splitmix64(((item * NOISES.len() + ni) * DRAWS + d) as u64),
                );
                for (ri, res) in refiners.run(&img, seed).into_iter().enumerate() {
                    let Some(res) = res else { continue };
                    let err = [f64::from(res.x - truth.0), f64::from(res.y - truth.1)];
                    let cov = match (var, res.covariance_per_noise_variance) {
                        (Some(v), Some(c)) => Some([
                            v * f64::from(c.xx),
                            v * f64::from(c.xy),
                            v * f64::from(c.yy),
                        ]),
                        _ => None,
                    };
                    let at_seed = [res.x, res.y] == seed;
                    samples[ri].push(Sample { err, at_seed, cov });
                }
            }
            std::array::from_fn(|ri| phase_stats(draws, &samples[ri]))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cell aggregation (pool the phases of one refiner × condition).

#[derive(Debug, Default)]
struct CellSummary {
    acceptance: f64,
    at_seed: f64,
    rms_bias: f64,
    rms_bias_raw: f64,
    max_bias: f64,
    sigma: [f64; 2],
    sigma_2d: f64,
    rmse: f64,
    mean_abs_err: f64,
    z: Option<ZSummary>,
}

#[derive(Debug, Default)]
struct ZSummary {
    z_prec_std: [f64; 2],
    z_total_rms: [f64; 2],
    s0sq_prec: f64,
    s0sq_total: f64,
    /// σ̂₀² about the truth with `Σ + f²·I`, per [`FLOORS`] entry.
    s0sq_floor: [f64; FLOORS.len()],
    median_bias_over_pred: f64,
    sigma_pred: f64,
}

impl ZSummary {
    /// Smallest [`FLOORS`] entry (0 if none is needed) that brings σ̂₀²
    /// about the truth within 0.05 of the precision-only σ̂₀² (or of 1, if
    /// larger): the floor that absorbs the residual bias. `None` beyond the
    /// grid.
    fn floor_needed(&self) -> Option<f64> {
        let target = self.s0sq_prec.max(1.0) + 0.05;
        if self.s0sq_total <= target {
            return Some(0.0);
        }
        FLOORS
            .iter()
            .zip(self.s0sq_floor)
            .find(|(_, v)| *v <= target)
            .map(|(f, _)| *f)
    }
}

fn summarize(phases: &[&PhaseStats]) -> CellSummary {
    let n_total: usize = phases.iter().map(|p| p.n_total).sum();
    let n_acc: usize = phases.iter().map(|p| p.n_acc).sum();
    let mut out = CellSummary {
        acceptance: n_acc as f64 / n_total as f64,
        at_seed: phases.iter().map(|p| p.n_at_seed).sum::<usize>() as f64 / n_acc.max(1) as f64,
        ..Default::default()
    };
    let seen: Vec<&&PhaseStats> = phases.iter().filter(|p| p.n_acc > 0).collect();
    if seen.is_empty() {
        out.rms_bias = f64::NAN;
        return out;
    }
    let b2 = |p: &PhaseStats| p.bias[0].powi(2) + p.bias[1].powi(2);
    let mean_b2 = seen.iter().map(|p| b2(p)).sum::<f64>() / seen.len() as f64;
    out.rms_bias_raw = mean_b2.sqrt();
    out.max_bias = seen.iter().map(|p| b2(p).sqrt()).fold(0.0, f64::max);
    out.rmse = (seen.iter().map(|p| p.sum_sq_err).sum::<f64>() / n_acc as f64).sqrt();
    out.mean_abs_err = seen.iter().map(|p| p.sum_abs_err).sum::<f64>() / n_acc as f64;

    let dof: usize = seen.iter().map(|p| p.n_acc.saturating_sub(1)).sum();
    if dof == 0 {
        out.rms_bias = out.rms_bias_raw;
        return out;
    }
    for k in 0..2 {
        out.sigma[k] = (seen.iter().map(|p| p.ss[k]).sum::<f64>() / dof as f64).sqrt();
    }
    out.sigma_2d = out.sigma[0].hypot(out.sigma[1]);
    // A phase mean over n draws carries σ²/n of noise; remove it from the
    // mean squared bias, with each phase's own σ² (acceptance and scatter
    // differ by phase at low SNR). A single-draw phase gives no estimate.
    let noise_in_bias = seen
        .iter()
        .map(|p| {
            let n = p.n_acc as f64;
            if p.n_acc < 2 {
                0.0
            } else {
                (p.ss[0] + p.ss[1]) / (n - 1.0) / n
            }
        })
        .sum::<f64>()
        / seen.len() as f64;
    out.rms_bias = (mean_b2 - noise_in_bias).max(0.0).sqrt();

    let zs: Vec<(&PhaseStats, &ZStats)> = seen
        .iter()
        .filter_map(|p| p.z.as_ref().map(|z| (**p, z)))
        .collect();
    if !zs.is_empty() {
        let zdof: f64 = zs.iter().map(|(p, _)| (p.n_acc - 1) as f64).sum();
        let zn: f64 = zs.iter().map(|(p, _)| p.n_acc as f64).sum();
        let mut ratios: Vec<f64> = zs
            .iter()
            .map(|(p, z)| b2(p).sqrt() / z.sigma_pred)
            .collect();
        ratios.sort_by(f64::total_cmp);
        let sum = |f: &dyn Fn(&ZStats) -> f64| zs.iter().map(|(_, z)| f(z)).sum::<f64>();
        out.z = Some(ZSummary {
            z_prec_std: [
                (sum(&|z| z.zp2[0]) / zdof).sqrt(),
                (sum(&|z| z.zp2[1]) / zdof).sqrt(),
            ],
            z_total_rms: [
                (sum(&|z| z.zt2[0]) / zn).sqrt(),
                (sum(&|z| z.zt2[1]) / zn).sqrt(),
            ],
            s0sq_prec: sum(&|z| z.chi_prec) / (2.0 * zdof),
            s0sq_total: sum(&|z| z.chi_total) / (2.0 * zn),
            s0sq_floor: std::array::from_fn(|i| sum(&|z| z.chi_floor[i]) / (2.0 * zn)),
            median_bias_over_pred: ratios[ratios.len() / 2],
            sigma_pred: zs
                .iter()
                .map(|(p, z)| z.sigma_pred * p.n_acc as f64)
                .sum::<f64>()
                / zn,
        });
    }
    out
}

/// Least-squares fit of the per-phase bias as a linear function of the
/// corner's offset from the fit-window centre, `b = (G − I)·u` with
/// `u = truth − round(truth)`: a refiner that shrinks the offset toward the
/// window centre by a fixed factor has `G = g·I`, `g < 1`. Returns `G` and
/// the radial RMS of the bias left after removing the linear part, or
/// `None` when a phase has no accepted result.
fn linear_gain(phases: &[&PhaseStats]) -> Option<([[f64; 2]; 2], f64)> {
    let mut suu = [[0.0f64; 2]; 2];
    let mut sbu = [[0.0f64; 2]; 2];
    let mut pairs = Vec::with_capacity(phases.len());
    for (i, p) in phases.iter().enumerate() {
        if p.n_acc == 0 {
            return None;
        }
        let t = truth_of(phase_of(i));
        let u = [f64::from(t.0 - t.0.round()), f64::from(t.1 - t.1.round())];
        for a in 0..2 {
            for c in 0..2 {
                suu[a][c] += u[a] * u[c];
                sbu[a][c] += p.bias[a] * u[c];
            }
        }
        pairs.push((u, p.bias));
    }
    let det = suu[0][0] * suu[1][1] - suu[0][1] * suu[1][0];
    let inv = [
        [suu[1][1] / det, -suu[0][1] / det],
        [-suu[1][0] / det, suu[0][0] / det],
    ];
    let m: [[f64; 2]; 2] = std::array::from_fn(|a| {
        std::array::from_fn(|c| sbu[a][0] * inv[0][c] + sbu[a][1] * inv[1][c])
    });
    let resid = pairs
        .iter()
        .map(|(u, b)| {
            (0..2)
                .map(|a| (b[a] - m[a][0] * u[0] - m[a][1] * u[1]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / pairs.len() as f64;
    let g = [[m[0][0] + 1.0, m[0][1]], [m[1][0], m[1][1] + 1.0]];
    Some((g, resid.sqrt()))
}

// ---------------------------------------------------------------------------
// Driver.

struct Results {
    fixtures: Vec<Fixture>,
    /// `stats[fixture][phase][noise][refiner]`.
    stats: Vec<Vec<Vec<[PhaseStats; 3]>>>,
}

impl Results {
    fn phases(&self, fx: usize, noise: usize, refiner: usize) -> Vec<&PhaseStats> {
        self.stats[fx]
            .iter()
            .map(|per_noise| &per_noise[noise][refiner])
            .collect()
    }

    fn cell(&self, fx: usize, noise: usize, refiner: usize) -> CellSummary {
        summarize(&self.phases(fx, noise, refiner))
    }

    fn gain(&self, fx: usize, noise: usize, refiner: usize) -> Option<([[f64; 2]; 2], f64)> {
        linear_gain(&self.phases(fx, noise, refiner))
    }

    fn find(&self, geometry: usize, cell: usize, contrast: usize, blur: usize) -> usize {
        self.fixtures
            .iter()
            .position(|f| {
                f.geometry == geometry && f.cell == cell && f.contrast == contrast && f.blur == blur
            })
            .unwrap()
    }
}

fn run_grid() -> Results {
    let mut fixtures = Vec::new();
    for geometry in 0..GEOMETRIES.len() {
        for cell in 0..CELLS.len() {
            for contrast in 0..CONTRASTS.len() {
                for blur in 0..BLURS.len() {
                    fixtures.push(Fixture {
                        geometry,
                        cell,
                        contrast,
                        blur,
                    });
                }
            }
        }
    }
    let items: Vec<(Fixture, (usize, usize))> = fixtures
        .iter()
        .flat_map(|&fx| (0..PHASES * PHASES).map(move |p| (fx, phase_of(p))))
        .collect();

    // Work items are independent and seeded by index, so the split across
    // threads does not change any number.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut slots: Vec<Option<Vec<[PhaseStats; 3]>>> = vec![None; items.len()];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let items = &items;
                scope.spawn(move || {
                    (t..items.len())
                        .step_by(threads)
                        .map(|i| (i, run_item(i, items[i].0, items[i].1)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, r) in h.join().unwrap() {
                slots[i] = Some(r);
            }
        }
    });
    let mut flat = slots.into_iter().map(Option::unwrap);
    let stats = fixtures
        .iter()
        .map(|_| flat.by_ref().take(PHASES * PHASES).collect())
        .collect();
    Results { fixtures, stats }
}

/// Noise-free mean |error| and RMS bias on the 8×-supersampled
/// [`aa_chessboard`] against the exact-area board, axis-aligned, 40/200:
/// how much of a clean-image error figure comes from the renderer.
fn renderer_check() -> Vec<Value> {
    let mut rows = Vec::new();
    let mut refiners = Refiners::new();
    for &cell in &CELLS {
        for &blur in &[0.0f32, 1.5] {
            let mut errs: [[Vec<[f64; 2]>; 3]; 2] = Default::default();
            for p in 0..PHASES * PHASES {
                let truth = truth_of(phase_of(p));
                let seed = [truth.0.round(), truth.1.round()];
                let boards = [
                    aa_chessboard(SIZE, cell, truth, 40, 200),
                    area_chessboard(SIZE, cell, truth, 90f32.to_radians(), 40, 200),
                ];
                for (bi, mut img) in boards.into_iter().enumerate() {
                    gaussian_blur(&mut img, SIZE, blur);
                    for (ri, res) in refiners.run(&img, seed).into_iter().enumerate() {
                        if let Some(r) = res {
                            errs[bi][ri].push([f64::from(r.x - truth.0), f64::from(r.y - truth.1)]);
                        }
                    }
                }
            }
            for (ri, name) in REFINERS.iter().enumerate() {
                let stat = |e: &[[f64; 2]]| {
                    let n = e.len() as f64;
                    (
                        e.iter().map(|v| v[0].hypot(v[1])).sum::<f64>() / n,
                        (e.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>() / n).sqrt(),
                    )
                };
                let (ss_mean, ss_rms) = stat(&errs[0][ri]);
                let (ex_mean, ex_rms) = stat(&errs[1][ri]);
                rows.push(json!({
                    "refiner": name, "cell": cell, "blur": blur,
                    "supersampled": { "mean_abs_err": ss_mean, "rms_bias": ss_rms },
                    "exact_area": { "mean_abs_err": ex_mean, "rms_bias": ex_rms },
                }));
            }
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// Report.

fn fmt(v: f64) -> String {
    if v.is_nan() {
        "–".into()
    } else {
        format!("{v:.3}")
    }
}

fn blur_header() -> String {
    BLURS
        .iter()
        .map(|b| format!("blur {b}"))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn print_tables(r: &Results, renderer: &[Value]) {
    let mut s = String::new();
    let (axis, c8, hi) = (0, 1, 0);
    let white = [1usize, 2, 3, 4];

    // 1. Bias floor per refiner at noise 0.
    let _ = writeln!(s, "\n### Noise-free RMS bias (px), axis-aligned, 40/200\n");
    let _ = writeln!(s, "| refiner | cell | {} |", blur_header());
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for (ri, name) in REFINERS.iter().enumerate() {
        for (ci, cell) in CELLS.iter().enumerate() {
            let row: Vec<String> = (0..BLURS.len())
                .map(|b| fmt(r.cell(r.find(axis, ci, hi, b), 0, ri).rms_bias))
                .collect();
            let _ = writeln!(s, "| {name} | {cell} | {} |", row.join(" | "));
        }
    }

    // 2. Bias vs precision under white noise, cell 8.
    let _ = writeln!(
        s,
        "\n### RMS bias / σ (px) under white noise, axis-aligned, cell 8, 40/200\n"
    );
    let _ = writeln!(s, "| refiner | blur | σₙ=1 | σₙ=2 | σₙ=4 | σₙ=8 |");
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for (ri, name) in REFINERS.iter().enumerate() {
        for (b, blur) in BLURS.iter().enumerate() {
            let fx = r.find(axis, c8, hi, b);
            let row: Vec<String> = white
                .iter()
                .map(|&n| {
                    let c = r.cell(fx, n, ri);
                    format!("{} / {}", fmt(c.rms_bias), fmt(c.sigma_2d))
                })
                .collect();
            let _ = writeln!(s, "| {name} | {blur} | {} |", row.join(" | "));
        }
    }

    // 3. SaddlePoint: a-posteriori variance factor about the truth.
    let _ = writeln!(
        s,
        "\n### SaddlePoint σ̂₀² = mean(eᵀΣ⁻¹e)/2 about the truth, axis-aligned, cell 8 (median |bias|/σ_pred)\n"
    );
    let header: Vec<String> = NOISES[1..].iter().map(|n| n.label()).collect();
    let _ = writeln!(s, "| contrast | blur | {} |", header.join(" | "));
    let _ = writeln!(s, "|---|---|{}", "---|".repeat(header.len()));
    for (contrast, (d, br)) in CONTRASTS.iter().enumerate() {
        for (b, blur) in BLURS.iter().enumerate() {
            let fx = r.find(axis, c8, contrast, b);
            let row: Vec<String> = (1..NOISES.len())
                .map(|n| {
                    let z = r.cell(fx, n, SADDLE).z.unwrap();
                    format!("{:.1} ({:.1})", z.s0sq_total, z.median_bias_over_pred)
                })
                .collect();
            let _ = writeln!(s, "| {d}/{br} | {blur} | {} |", row.join(" | "));
        }
    }

    // 3b. SaddlePoint scatter calibration over the whole grid.
    let _ = writeln!(
        s,
        "\n### SaddlePoint scatter calibration over all geometries, cells, contrasts and blurs\n"
    );
    let _ = writeln!(
        s,
        "| noise | z_prec std min | median | max | σ̂₀² prec min | max | σ_pred/σ_emp min | max | accept min |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
    for (n, noise) in NOISES.iter().enumerate().skip(1) {
        let mut zs = Vec::new();
        let mut s0 = Vec::new();
        let mut ratio = Vec::new();
        let mut acc = 1.0f64;
        for fx in 0..r.fixtures.len() {
            let c = r.cell(fx, n, SADDLE);
            let z = c.z.as_ref().unwrap();
            zs.extend(z.z_prec_std);
            s0.push(z.s0sq_prec);
            ratio.push(z.sigma_pred / c.sigma_2d);
            acc = acc.min(c.acceptance);
        }
        for v in [&mut zs, &mut s0, &mut ratio] {
            v.sort_by(f64::total_cmp);
        }
        let _ = writeln!(
            s,
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {acc:.3} |",
            noise.label(),
            zs[0],
            zs[zs.len() / 2],
            zs[zs.len() - 1],
            s0[0],
            s0[s0.len() - 1],
            ratio[0],
            ratio[ratio.len() - 1],
        );
    }

    // 3c. SaddlePoint model-error floor.
    let _ = writeln!(
        s,
        "\n### SaddlePoint model-error floor f (px per axis) for σ̂₀²(Σ + f²I) ≤ max(1, σ̂₀² prec) + 0.05\n"
    );
    let _ = writeln!(
        s,
        "| noise | max, σ_emp < 0.3 px | median, σ_emp < 0.3 px | max, all | worst fixture (σ_emp < 0.3 px) |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|");
    let floor = |f: Option<f64>| f.map_or(">0.05".to_string(), |f| format!("{f}"));
    for (n, noise) in NOISES.iter().enumerate().skip(1) {
        let mut good = Vec::new();
        let mut all = Vec::new();
        for (fx, f) in r.fixtures.iter().enumerate() {
            let c = r.cell(fx, n, SADDLE);
            let need =
                c.z.as_ref()
                    .unwrap()
                    .floor_needed()
                    .unwrap_or(f64::INFINITY);
            all.push(need);
            if c.sigma_2d < 0.3 {
                good.push((need, *f));
            }
        }
        good.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.sort_by(f64::total_cmp);
        let fin = |v: f64| floor(v.is_finite().then_some(v));
        let (worst, wf) = good[good.len() - 1];
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} cell {} {}/{} blur {} |",
            noise.label(),
            fin(worst),
            fin(good[good.len() / 2].0),
            fin(all[all.len() - 1]),
            GEOMETRIES[wf.geometry].0,
            CELLS[wf.cell],
            CONTRASTS[wf.contrast].0,
            CONTRASTS[wf.contrast].1,
            BLURS[wf.blur],
        );
    }

    // 4. Cell size and skew at σₙ = 2.
    let _ = writeln!(
        s,
        "\n### RMS bias / σ (px) at white σₙ = 2, 40/200, by geometry and cell (blur 0 | 1.5)\n"
    );
    let _ = writeln!(
        s,
        "| refiner | geometry | cell 5 | cell 8 | cell 12 | accept (min) | at seed (max) |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|");
    for (ri, name) in REFINERS.iter().enumerate() {
        for (g, (gname, _)) in GEOMETRIES.iter().enumerate() {
            let mut min_acc = 1.0f64;
            let mut max_at_seed = 0.0f64;
            let row: Vec<String> = (0..CELLS.len())
                .map(|ci| {
                    let parts: Vec<String> = [0usize, 2]
                        .iter()
                        .map(|&b| {
                            let c = r.cell(r.find(g, ci, hi, b), 2, ri);
                            min_acc = min_acc.min(c.acceptance);
                            max_at_seed = max_at_seed.max(c.at_seed);
                            format!("{}/{}", fmt(c.rms_bias), fmt(c.sigma_2d))
                        })
                        .collect();
                    parts.join(" \\| ")
                })
                .collect();
            let _ = writeln!(
                s,
                "| {name} | {gname} | {} | {min_acc:.3} | {max_at_seed:.3} |",
                row.join(" | ")
            );
        }
    }

    // 5. Bias structure: linear shrink toward the window centre.
    let _ = writeln!(
        s,
        "\n### Noise-free bias as a linear shrink, cell 8, 40/200: RMS bias → residual after `b = (G − I)·u` (G)\n"
    );
    let _ = writeln!(s, "| refiner | geometry | {} |", blur_header());
    let _ = writeln!(s, "|---|---|{}", "---|".repeat(BLURS.len()));
    for (ri, name) in REFINERS.iter().enumerate().skip(1) {
        for (g, (gname, _)) in GEOMETRIES.iter().enumerate() {
            let row: Vec<String> = (0..BLURS.len())
                .map(|b| {
                    let fx = r.find(g, c8, hi, b);
                    match r.gain(fx, 0, ri) {
                        Some((m, res)) => format!(
                            "{} → {} ([{:.2} {:+.2}; {:+.2} {:.2}])",
                            fmt(r.cell(fx, 0, ri).rms_bias),
                            fmt(res),
                            m[0][0],
                            m[0][1],
                            m[1][0],
                            m[1][1]
                        ),
                        None => "–".into(),
                    }
                })
                .collect();
            let _ = writeln!(s, "| {name} | {gname} | {} |", row.join(" | "));
        }
    }

    // 6. Renderer contribution.
    let _ = writeln!(
        s,
        "\n### Noise-free error: 8×-supersampled vs exact-area board (mean |e| / RMS, px)\n"
    );
    let _ = writeln!(s, "| refiner | cell | blur | supersampled | exact area |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for row in renderer {
        let g = |k: &str, m: &str| row[k][m].as_f64().unwrap();
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} / {} | {} / {} |",
            row["refiner"].as_str().unwrap(),
            row["cell"],
            row["blur"],
            fmt(g("supersampled", "mean_abs_err")),
            fmt(g("supersampled", "rms_bias")),
            fmt(g("exact_area", "mean_abs_err")),
            fmt(g("exact_area", "rms_bias")),
        );
    }
    println!("{s}");
}

fn to_json(r: &Results, renderer: Vec<Value>) -> Value {
    let mut cells = Vec::new();
    for (fi, fx) in r.fixtures.iter().enumerate() {
        let (gname, angle) = GEOMETRIES[fx.geometry];
        let (dark, bright) = CONTRASTS[fx.contrast];
        for (ni, noise) in NOISES.iter().enumerate() {
            for (ri, name) in REFINERS.iter().enumerate() {
                let c = r.cell(fi, ni, ri);
                let phase_bias: Vec<Value> = r.stats[fi]
                    .iter()
                    .map(|per_noise| {
                        let p = &per_noise[ni][ri];
                        if p.n_acc == 0 {
                            Value::Null
                        } else {
                            json!(p.bias)
                        }
                    })
                    .collect();
                cells.push(json!({
                    "refiner": name,
                    "geometry": gname,
                    "axis_angle_deg": angle,
                    "cell": CELLS[fx.cell],
                    "dark": dark,
                    "bright": bright,
                    "blur": BLURS[fx.blur],
                    "noise": noise.json(),
                    "acceptance": c.acceptance,
                    "rms_bias": c.rms_bias,
                    "rms_bias_raw": c.rms_bias_raw,
                    "max_phase_bias": c.max_bias,
                    "sigma": c.sigma,
                    "sigma_2d": c.sigma_2d,
                    "rmse": c.rmse,
                    "mean_abs_err": c.mean_abs_err,
                    "bias_dominates": c.rms_bias > c.sigma_2d,
                    "saddle_prediction": c.z.as_ref().map(|z| json!({
                        "sigma_pred": z.sigma_pred,
                        "z_prec_std": z.z_prec_std,
                        "z_total_rms": z.z_total_rms,
                        "s0sq_prec": z.s0sq_prec,
                        "s0sq_total": z.s0sq_total,
                        "median_bias_over_pred": z.median_bias_over_pred,
                        "s0sq_total_with_floor": FLOORS.iter().zip(z.s0sq_floor).map(|(f, v)| json!({
                            "floor": f,
                            "s0sq": v,
                        })).collect::<Vec<_>>(),
                        "floor_needed": z.floor_needed(),
                    })),
                    "phase_bias": phase_bias,
                    "at_seed_rate": c.at_seed,
                    "phase_gain": r.gain(fi, ni, ri).map(|(g, res)| json!({
                        "g": g,
                        "residual_rms": res,
                    })),
                }));
            }
        }
    }
    json!({
        "meta": {
            "image_size": SIZE,
            "corner": "CENTER + (kx, ky)/PHASES",
            "center": CENTER,
            "phases_per_axis": PHASES,
            "phase_order": "index = ky·PHASES + kx",
            "draws": DRAWS,
            "seed": "truth rounded to the nearest pixel",
            "renderer": "area_chessboard (exact pixel area), then gaussian_blur, then noise; u8 after each",
            "noise_variance": "white: σ² + 1/12; signal: a + b·I (+1/12), σ_pred uses the mean over the 5×5 fit patch",
            "bias": "rms_bias = sqrt(mean_phase |phase mean error|² − σ_2d²/n), i.e. with the draw noise removed",
        },
        "cells": cells,
        "renderer_check": renderer,
    })
}

#[test]
#[ignore]
fn refiner_bias_precision_sweep() {
    let start = std::time::Instant::now();
    let results = run_grid();
    let renderer = renderer_check();
    print_tables(&results, &renderer);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/uq01");
    std::fs::create_dir_all(&dir).unwrap();
    let dir = dir.canonicalize().unwrap();
    let path = dir.join("refiner_accuracy.json");
    let doc = to_json(&results, renderer);
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!(
        "wrote {} in {:.1} s",
        path.display(),
        start.elapsed().as_secs_f64()
    );
}
