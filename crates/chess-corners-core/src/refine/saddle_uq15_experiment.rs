//! UQ-15 investigation harness (temporary; not committed).
#![allow(dead_code, clippy::all)]

use super::*;
use crate::imageview::ImageView;
use chess_corners_testutil::{rotated_area_chessboard, rotated_area_chessboard_f64};

const SIZE: usize = 41;
const C: f32 = 20.0;

pub(super) struct FImg {
    pub side: usize,
    pub data: Vec<f32>,
}

impl Pixels for FImg {
    fn supports(&self, x: i32, y: i32, radius: i32) -> bool {
        let max = self.side as i32 - 1;
        x - radius >= 0 && y - radius >= 0 && x + radius <= max && y + radius <= max
    }
    fn read_patch(&self, x: i32, y: i32, radius: i32, out: &mut [f32]) {
        let side = (2 * radius + 1) as usize;
        for (row, dst) in out.chunks_exact_mut(side).enumerate() {
            let start = ((y - radius) as usize + row) * self.side + (x - radius) as usize;
            dst.copy_from_slice(&self.data[start..start + side]);
        }
    }
}

pub(super) fn blur_f64(img: &mut [f64], size: usize, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let kernel: Vec<f64> = (0..2 * radius + 1)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let s: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|k| k / s).collect();
    let mut tmp = vec![0.0; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0;
            for (ki, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += img[y * size + sx] * k;
            }
            tmp[y * size + x] = acc;
        }
    }
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0;
            for (ki, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += tmp[sy * size + x] * k;
            }
            img[y * size + x] = acc;
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Cond {
    pub cell: usize,
    pub rot_deg: f32,
    pub axis_deg: f32,
    pub blur: f32,
    pub dark: u8,
    pub bright: u8,
}

pub(super) fn phases(n: usize) -> Vec<(f32, f32)> {
    (0..n * n)
        .map(|k| (((k % n) as f32 + 0.5) / n as f32, ((k / n) as f32 + 0.5) / n as f32))
        .collect()
}

/// Noise-free errors of `refiner` over `phases`; `quant` = render u8.
pub(super) fn errors(
    refiner: &mut SaddlePointRefiner,
    cond: Cond,
    phases: &[(f32, f32)],
    quant: bool,
    steps: usize,
) -> (Vec<[f64; 2]>, usize) {
    let mut out = Vec::new();
    let mut nonconv = 0;
    for &(px, py) in phases {
        let truth = (C + px, C + py);
        let mut f = rotated_area_chessboard_f64(
            SIZE,
            cond.cell,
            truth,
            cond.rot_deg.to_radians(),
            cond.axis_deg.to_radians(),
            cond.dark,
            cond.bright,
        );
        let seed = [truth.0.round(), truth.1.round()];
        let centre = [seed[0] as i32, seed[1] as i32];
        let est = if quant {
            let mut u: Vec<u8> = f.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
            chess_corners_testutil::gaussian_blur(&mut u, SIZE, cond.blur);
            let view = ImageView::from_u8_slice(SIZE, SIZE, &u).unwrap();
            refiner.estimate(&view, centre, seed, steps, STEP_TOL)
        } else {
            blur_f64(&mut f, SIZE, cond.blur as f64);
            let img = FImg {
                side: SIZE,
                data: f.iter().map(|&v| v as f32).collect(),
            };
            refiner.estimate(&img, centre, seed, steps, STEP_TOL)
        };
        match est {
            Ok(e) => {
                if steps > 0 && e.covariance.is_none() {
                    nonconv += 1;
                    continue;
                }
                out.push([
                    f64::from(e.xy[0] - truth.0),
                    f64::from(e.xy[1] - truth.1),
                ]);
            }
            Err(_) => nonconv += 1,
        }
    }
    (out, nonconv)
}

pub(super) fn rms(e: &[[f64; 2]]) -> f64 {
    (e.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>() / (2.0 * e.len() as f64)).sqrt()
}

pub(super) fn mean(e: &[[f64; 2]]) -> [f64; 2] {
    let n = e.len() as f64;
    [
        e.iter().map(|v| v[0]).sum::<f64>() / n,
        e.iter().map(|v| v[1]).sum::<f64>() / n,
    ]
}

fn variants() -> Vec<(&'static str, i32, FitDesign)> {
    let d = |w: f64, s: &[f64]| FitDesign {
        weight_sigma: w,
        smooth: s.to_vec(),
        ..FitDesign::default()
    };
    let b3 = [0.25, 0.5, 0.25];
    let b5 = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    vec![
        ("r2", 2, d(0.0, &[])),
        ("r1", 1, d(0.0, &[])),
        ("r3", 3, d(0.0, &[])),
        ("r2 s121", 2, d(0.0, &b3)),
        ("r2 s14641", 2, d(0.0, &b5)),
        ("r2 w1.0", 2, d(1.0, &[])),
        ("r2 w1.5", 2, d(1.5, &[])),
        ("r3 w1.5", 3, d(1.5, &[])),
        ("r3 s121", 3, d(0.0, &b3)),
    ]
}

/// Separable window weight families for the moving-window fit.
#[derive(Clone, Copy, Debug)]
pub(super) enum Weight {
    /// Cubic B-spline `B3(v/s)`, support `|v| < 2s`.
    BSpline(f64),
    /// Hann `cos²(πv/2R)`, support `|v| < R`.
    Hann(f64),
    /// Uniform box `|v| ≤ R` (discontinuous; reference).
    Box(f64),
}

impl Weight {
    pub fn reach(self) -> f64 {
        match self {
            Weight::BSpline(s) => 2.0 * s,
            Weight::Hann(r) | Weight::Box(r) => r,
        }
    }
    pub fn eval(self, v: f64) -> f64 {
        match self {
            Weight::BSpline(s) => {
                let x = (v / s).abs();
                if x < 1.0 {
                    2.0 / 3.0 - x * x + 0.5 * x * x * x
                } else if x < 2.0 {
                    (2.0 - x).powi(3) / 6.0
                } else {
                    0.0
                }
            }
            Weight::Hann(r) => {
                if v.abs() < r {
                    (std::f64::consts::FRAC_PI_2 * v / r).cos().powi(2)
                } else {
                    0.0
                }
            }
            Weight::Box(r) => {
                if v.abs() <= r {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

fn solve6(mut m: [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let p = (col..6).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[p][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, p);
        for r in 0..6 {
            if r != col {
                let f = m[r][col] / m[col][col];
                for k in col..7 {
                    m[r][k] -= f * m[col][k];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| m[i][6] / m[i][i]))
}

/// Weighted quadratic fit on raw pixels around continuous centre `o`.
pub(super) fn mls_coefs(img: &FImg, o: [f64; 2], w: Weight) -> Option<[f64; 6]> {
    let reach = w.reach();
    let lo = |c: f64| (c - reach).ceil() as i64;
    let hi = |c: f64| (c + reach).floor() as i64;
    let mut m = [[0.0f64; 7]; 6];
    for ny in lo(o[1])..=hi(o[1]) {
        let vy = ny as f64 - o[1];
        let wy = w.eval(vy);
        if wy == 0.0 {
            continue;
        }
        for nx in lo(o[0])..=hi(o[0]) {
            let vx = nx as f64 - o[0];
            let ww = wy * w.eval(vx);
            if ww == 0.0 {
                continue;
            }
            if nx < 0 || ny < 0 || nx >= img.side as i64 || ny >= img.side as i64 {
                return None;
            }
            let z = f64::from(img.data[ny as usize * img.side + nx as usize]);
            let phi = [vx * vx, vx * vy, vy * vy, vx, vy, 1.0];
            for r in 0..6 {
                for c in 0..6 {
                    m[r][c] += ww * phi[r] * phi[c];
                }
                m[r][6] += ww * phi[r] * z;
            }
        }
    }
    solve6(m)
}

fn stationary(c: [f64; 6]) -> Option<[f64; 2]> {
    let [a, b, cc, d, e, _] = c;
    let det = 4.0 * a * cc - b * b;
    if !(det < 0.0) {
        return None;
    }
    Some([-(2.0 * cc * d - b * e) / det, (b * d - 2.0 * a * e) / det])
}

/// Moving-window fixed point: returns the converged absolute position.
pub(super) fn mls_estimate(img: &FImg, seed: [f64; 2], w: Weight) -> Option<[f64; 2]> {
    let mut o = seed;
    for _ in 0..100 {
        let u = stationary(mls_coefs(img, o, w)?)?;
        o = [o[0] + u[0], o[1] + u[1]];
        if (o[0] - seed[0]).abs() > 1.5 || (o[1] - seed[1]).abs() > 1.5 {
            return None;
        }
        if u[0].hypot(u[1]) < 1e-7 {
            return Some(o);
        }
    }
    None
}

/// Keys cubic (a = −0.5) weights for fraction t at taps −1, 0, 1, 2.
fn keys(t: f64) -> [f64; 4] {
    let a = -0.5;
    let k = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    };
    [k(t + 1.0), k(t), k(1.0 - t), k(2.0 - t)]
}

/// Lagrange 4-point cubic weights for fraction t at taps −1, 0, 1, 2.
fn lagrange(t: f64) -> [f64; 4] {
    [
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ]
}

/// Re-centred fit with cubic resampling of the 5×5 window.
pub(super) fn cubic_estimate(img: &FImg, seed: [f64; 2], radius: i32, lag: bool) -> Option<[f64; 2]> {
    let op = fit_operator(radius, &FitDesign::default())?;
    let cfg = SaddlePointConfig::default();
    let mut o = seed;
    let side = (2 * radius + 1) as usize;
    for _ in 0..200 {
        let base = [o[0].floor() as i64, o[1].floor() as i64];
        let t = [o[0] - base[0] as f64, o[1] - base[1] as f64];
        let (kx, ky) = if lag { (lagrange(t[0]), lagrange(t[1])) } else { (keys(t[0]), keys(t[1])) };
        let mut coefs = [0.0f32; 5];
        for (k, col) in op.columns.iter().enumerate() {
            let (px, py) = ((k % side) as i64 - radius as i64, (k / side) as i64 - radius as i64);
            let mut v = 0.0;
            for (j, wy) in ky.iter().enumerate() {
                for (i, wx) in kx.iter().enumerate() {
                    let x = base[0] + px + i as i64 - 1;
                    let y = base[1] + py + j as i64 - 1;
                    v += wx * wy * f64::from(img.data[y as usize * img.side + x as usize]);
                }
            }
            for (c, r) in coefs.iter_mut().zip(col) {
                *c += r * (v as f32 - 120.0);
            }
        }
        let s = solve(&cfg, coefs).ok()?;
        o = [o[0] + f64::from(s.u[0]), o[1] + f64::from(s.u[1])];
        if s.u[0].hypot(s.u[1]) < 1e-6 {
            return Some(o);
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Method {
    Current,
    Cubic,
    Lagrange,
    /// The refiner itself with the cubic resampler.
    RefinerCubic,
    Mls(Weight),
}

pub(super) fn image_of(cond: Cond, truth: (f32, f32), quant: bool) -> FImg {
    let mut f = rotated_area_chessboard_f64(
        SIZE,
        cond.cell,
        truth,
        cond.rot_deg.to_radians(),
        cond.axis_deg.to_radians(),
        cond.dark,
        cond.bright,
    );
    let data = if quant {
        let mut u: Vec<u8> = f.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
        chess_corners_testutil::gaussian_blur(&mut u, SIZE, cond.blur);
        u.iter().map(|&v| f32::from(v)).collect()
    } else {
        blur_f64(&mut f, SIZE, cond.blur as f64);
        f.iter().map(|&v| v as f32).collect()
    };
    FImg { side: SIZE, data }
}

pub(super) fn method_estimate(m: Method, img: &FImg, seed: [f32; 2]) -> Option<[f64; 2]> {
    let s = [f64::from(seed[0]), f64::from(seed[1])];
    match m {
        Method::Current => {
            let mut r = SaddlePointRefiner::new(SaddlePointConfig::default());
            let centre = [seed[0] as i32, seed[1] as i32];
            let e = r.estimate(img, centre, seed, MAX_STEPS, STEP_TOL).ok()?;
            e.covariance?;
            Some([f64::from(e.xy[0]), f64::from(e.xy[1])])
        }
        Method::Cubic => cubic_estimate(img, s, 2, false),
        Method::Lagrange => cubic_estimate(img, s, 2, true),
        Method::RefinerCubic => {
            let design = FitDesign { resampler: Resampler::Cubic, ..FitDesign::default() };
            let mut r = SaddlePointRefiner::with_design(SaddlePointConfig::default(), &design);
            let centre = [seed[0] as i32, seed[1] as i32];
            let e = r.estimate(img, centre, seed, MAX_STEPS, STEP_TOL).ok()?;
            e.covariance?;
            Some([f64::from(e.xy[0]), f64::from(e.xy[1])])
        }
        Method::Mls(w) => mls_estimate(img, s, w),
    }
}

#[test]
#[ignore]
fn uq15_methods() {
    let ph = phases(6);
    let angles = [0.0f32, 2.0, 5.0, 8.0, 15.0, 30.0, 45.0];
    let methods = [
        ("current", Method::Current),
        ("keys", Method::Cubic),
        ("box2.5", Method::Mls(Weight::Box(2.5))),
        ("bspl1.0", Method::Mls(Weight::BSpline(1.0))),
        ("bspl1.25", Method::Mls(Weight::BSpline(1.25))),
        ("bspl1.5", Method::Mls(Weight::BSpline(1.5))),
        ("hann2.5", Method::Mls(Weight::Hann(2.5))),
        ("hann3", Method::Mls(Weight::Hann(3.0))),
        ("hann3.5", Method::Mls(Weight::Hann(3.5))),
    ];
    for cell in [8usize, 5] {
        for blur in [0.0f32, 0.8, 1.5, 2.5] {
            println!("cell {cell} blur {blur}: f32 noise-free RMS (mpx) | u8");
            for (name, m) in methods {
                let mut lf = String::new();
                let mut lu = String::new();
                let mut fails = 0;
                for &a in &angles {
                    for (quant, line) in [(false, &mut lf), (true, &mut lu)] {
                        let mut errs = Vec::new();
                        for &(px, py) in &ph {
                            let truth = (C + px, C + py);
                            let img = image_of(cond(cell, a, blur), truth, quant);
                            match method_estimate(m, &img, [truth.0.round(), truth.1.round()]) {
                                Some(xy) => errs.push([
                                    xy[0] - f64::from(truth.0),
                                    xy[1] - f64::from(truth.1),
                                ]),
                                None => fails += 1,
                            }
                        }
                        *line += &format!("{:6.1}", rms(&errs) * 1e3);
                    }
                }
                println!("  {name:9}{lf} |{lu}  fails {fails}");
            }
        }
    }
}

/// Optical blur BEFORE pixel integration: exact sub-pixel areas on a K×
/// grid, Gaussian PSF σ there, then K×K block means. `size` px image.
pub(super) fn render_psf(cond: Cond, truth: (f32, f32), psf: f64, size: usize) -> Vec<f64> {
    const K: usize = 8;
    let fine = size * K;
    let kf = K as f32;
    let off = ((truth.0 + 0.5) * kf - 0.5, (truth.1 + 0.5) * kf - 0.5);
    let mut f = rotated_area_chessboard_f64(
        fine,
        cond.cell * K,
        off,
        cond.rot_deg.to_radians(),
        cond.axis_deg.to_radians(),
        cond.dark,
        cond.bright,
    );
    blur_f64(&mut f, fine, psf * K as f64);
    let mut out = vec![0.0; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut s = 0.0;
            for j in 0..K {
                for i in 0..K {
                    s += f[(y * K + j) * fine + x * K + i];
                }
            }
            out[y * size + x] = s / (K * K) as f64;
        }
    }
    out
}

#[test]
#[ignore]
fn uq15_psf_before_sampling() {
    const S: usize = 27;
    let cc = 13.0f32;
    let ph = phases(5);
    let angles = [0.0f32, 2.0, 5.0, 8.0, 15.0, 30.0, 45.0];
    println!("PSF before sampling, cell 8, f32, current refiner, RMS mpx; post-sampling blur for reference");
    for psf in [0.0f64, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5] {
        let mut line = String::new();
        let mut post = String::new();
        for &a in &angles {
            let c = cond(8, a, 0.0);
            let mut e1 = Vec::new();
            let mut e2 = Vec::new();
            for &(px, py) in &ph {
                let truth = (cc + px, cc + py);
                let seed = [truth.0.round(), truth.1.round()];
                let img = FImg {
                    side: S,
                    data: render_psf(c, truth, psf, S).iter().map(|&v| v as f32).collect(),
                };
                if let Some(xy) = method_estimate(Method::Current, &img, seed) {
                    e1.push([xy[0] - f64::from(truth.0), xy[1] - f64::from(truth.1)]);
                }
                let mut f = rotated_area_chessboard_f64(S, 8, truth, a.to_radians(), 90f32.to_radians(), 40, 200);
                blur_f64(&mut f, S, psf);
                let img = FImg { side: S, data: f.iter().map(|&v| v as f32).collect() };
                if let Some(xy) = method_estimate(Method::Current, &img, seed) {
                    e2.push([xy[0] - f64::from(truth.0), xy[1] - f64::from(truth.1)]);
                }
            }
            line += &format!("{:6.1}", rms(&e1) * 1e3);
            post += &format!("{:6.1}", rms(&e2) * 1e3);
        }
        println!("  psf {psf:4}: {line} | post {post}");
    }
}

#[test]
#[ignore]
fn uq15_psf_methods() {
    const S: usize = 27;
    let cc = 13.0f32;
    let ph = phases(5);
    let angles = [0.0f32, 2.0, 5.0, 8.0, 15.0, 22.0, 30.0, 38.0, 45.0];
    let methods = [
        ("current", Method::Current),
        ("keys", Method::Cubic),
        ("lagrange", Method::Lagrange),
        ("ref-cubic", Method::RefinerCubic),
    ];
    for cell in [8usize] {
        for psf in [0.0f64, 0.25, 0.5, 0.75, 1.0, 1.5] {
            println!("cell {cell} psf {psf}: RMS mpx at {angles:?}");
            let imgs: Vec<Vec<(FImg, (f32, f32))>> = angles
                .iter()
                .map(|&a| {
                    ph.iter()
                        .map(|&(px, py)| {
                            let truth = (cc + px, cc + py);
                            let d = render_psf(cond(cell, a, 0.0), truth, psf, S);
                            (FImg { side: S, data: d.iter().map(|&v| v as f32).collect() }, truth)
                        })
                        .collect()
                })
                .collect();
            for (name, m) in methods {
                let mut line = String::new();
                for set in &imgs {
                    let mut e = Vec::new();
                    for (img, truth) in set {
                        if let Some(xy) = method_estimate(m, img, [truth.0.round(), truth.1.round()]) {
                            e.push([xy[0] - f64::from(truth.0), xy[1] - f64::from(truth.1)]);
                        }
                    }
                    line += &format!("{:6.1}", rms(&e) * 1e3);
                }
                println!("  {name:9}{line}");
            }
        }
    }
}

/// Small sheared cells and precision: bilinear vs cubic resampler.
#[test]
#[ignore]
fn uq15_small_cells_and_precision() {
    const S: usize = 27;
    let cc = 13.0f32;
    let ph = phases(6);
    let designs = [
        ("bilinear", FitDesign { resampler: Resampler::Bilinear, ..FitDesign::default() }),
        ("cubic", FitDesign { resampler: Resampler::Cubic, ..FitDesign::default() }),
    ];
    println!("cell 5 sheared: noise-free RMS mpx (single fit | re-centred), non-converged of {}", ph.len());
    for psf_model in [false, true] {
        for axis in [40.0f32, 50.0, 60.0] {
            for blur in [1.5f64, 2.5] {
                let c = Cond { cell: 5, rot_deg: 0.0, axis_deg: axis, blur: 0.0, dark: 40, bright: 200 };
                let imgs: Vec<(FImg, (f32, f32))> = ph
                    .iter()
                    .map(|&(px, py)| {
                        let truth = (cc + px, cc + py);
                        let d = if psf_model {
                            render_psf(c, truth, blur, S)
                        } else {
                            let mut f = rotated_area_chessboard_f64(S, 5, truth, 0.0, axis.to_radians(), 40, 200);
                            let mut u: Vec<u8> = f.iter().map(|v| v.round() as u8).collect();
                            chess_corners_testutil::gaussian_blur(&mut u, S, blur as f32);
                            f = u.iter().map(|&v| f64::from(v)).collect();
                            f
                        };
                        (FImg { side: S, data: d.iter().map(|&v| v.round() as f32).collect() }, truth)
                    })
                    .collect();
                let mut line = format!("  {} axis {axis} blur {blur}:", if psf_model { "psf " } else { "post" });
                for (name, d) in &designs {
                    let mut r = SaddlePointRefiner::with_design(SaddlePointConfig::default(), d);
                    let (mut single, mut full, mut nc) = (Vec::new(), Vec::new(), 0);
                    for (img, truth) in &imgs {
                        let seed = [truth.0.round(), truth.1.round()];
                        let centre = [seed[0] as i32, seed[1] as i32];
                        let Ok(s1) = r.estimate(img, centre, seed, 0, STEP_TOL) else { continue };
                        single.push([f64::from(s1.xy[0] - truth.0), f64::from(s1.xy[1] - truth.1)]);
                        let e = r.estimate(img, centre, seed, MAX_STEPS, STEP_TOL).unwrap();
                        if e.covariance.is_none() {
                            nc += 1;
                        }
                        full.push([f64::from(e.xy[0] - truth.0), f64::from(e.xy[1] - truth.1)]);
                    }
                    line += &format!(" {name} {:.1} | {:.1} nc {nc};", rms(&single) * 1e3, rms(&full) * 1e3);
                }
                println!("{line}");
            }
        }
    }

    println!("\nprecision: mean predicted σ per axis (px per gray level), PSF model, cell 8");
    for rot in [0.0f32, 8.0, 45.0] {
        for blur in [0.0f64, 0.8, 1.5, 2.5] {
            let mut line = format!("  rot {rot} psf {blur}:");
            for (name, d) in &designs {
                let mut r = SaddlePointRefiner::with_design(SaddlePointConfig::default(), d);
                let mut acc = 0.0;
                let mut n = 0;
                for &(px, py) in &ph {
                    let truth = (cc + px, cc + py);
                    let img = FImg {
                        side: S,
                        data: render_psf(cond(8, rot, 0.0), truth, blur, S).iter().map(|&v| v as f32).collect(),
                    };
                    let seed = [truth.0.round(), truth.1.round()];
                    let centre = [seed[0] as i32, seed[1] as i32];
                    if let Ok(e) = r.estimate(&img, centre, seed, MAX_STEPS, STEP_TOL) {
                        if let Some(c) = e.covariance {
                            acc += 0.5 * (f64::from(c.xx).sqrt() + f64::from(c.yy).sqrt());
                            n += 1;
                        }
                    }
                }
                line += &format!(" {name} {:.5}", acc / n as f64);
            }
            println!("{line}");
        }
    }
}

/// Candidate comparison inside the bilinear-blend structure.
#[test]
#[ignore]
fn uq15_variants() {
    let ph = phases(6);
    let angles = [0.0f32, 2.0, 5.0, 8.0, 15.0, 30.0, 45.0];
    for blur in [0.0f32, 0.8, 1.5, 2.5] {
        println!("blur {blur}, cell 8, f32 noise-free RMS (mpx) | u8");
        for (name, radius, design) in variants() {
            let cfg = SaddlePointConfig {
                radius,
                ..SaddlePointConfig::default()
            };
            let mut r = SaddlePointRefiner::with_design(cfg, &design);
            let mut line = format!("  {name:12}");
            let mut lu = String::new();
            let mut nc = 0;
            for &a in &angles {
                let (ef, n1) = errors(&mut r, cond(8, a, blur), &ph, false, MAX_STEPS);
                let (eu, n2) = errors(&mut r, cond(8, a, blur), &ph, true, MAX_STEPS);
                nc += n1 + n2;
                line += &format!("{:6.1}", rms(&ef) * 1e3);
                lu += &format!("{:6.1}", rms(&eu) * 1e3);
            }
            println!("{line} |{lu}  nonconv {nc}");
        }
    }
}

fn cond(cell: usize, rot: f32, blur: f32) -> Cond {
    Cond {
        cell,
        rot_deg: rot,
        axis_deg: 90.0,
        blur,
        dark: 40,
        bright: 200,
    }
}

/// Step 1: noise-free error vs rotation, f32 vs u8, re-centred vs single fit.
#[test]
#[ignore]
fn uq15_angle_sweep() {
    let ph = phases(8);
    let mut r = SaddlePointRefiner::new(SaddlePointConfig::default());
    println!("cell blur | angle: rms_u8 rms_f32 |mean_f32| nonconv");
    for cell in [8usize, 12] {
        for blur in [0.0f32, 0.5, 0.8, 1.5, 2.5] {
            let mut line = format!("{cell} {blur}:");
            for a in (0..=45).step_by(1) {
                let c = cond(cell, a as f32, blur);
                let (eu, _) = errors(&mut r, c, &ph, true, MAX_STEPS);
                let (ef, nf) = errors(&mut r, c, &ph, false, MAX_STEPS);
                let m = mean(&ef);
                line += &format!(
                    "\n  {a:2} {:.4} {:.4} {:.4} {nf}",
                    rms(&eu),
                    rms(&ef),
                    m[0].hypot(m[1])
                );
            }
            println!("{line}");
        }
    }
}

fn print_map(e: &[[f64; 2]], n: usize) {
    for k in 0..2 {
        println!("  {}", if k == 0 { "ex" } else { "ey" });
        for j in 0..n {
            let row: Vec<String> = (0..n)
                .map(|i| format!("{:6.1}", e[j * n + i][k] * 1e3))
                .collect();
            println!("  {}", row.join(""));
        }
    }
}

#[test]
#[ignore]
fn uq15_fine_phase() {
    let mut r = SaddlePointRefiner::new(SaddlePointConfig::default());
    for rot in [2.0f32, 8.0, 0.0] {
        for blur in [0.0f32, 2.5] {
            println!("rot {rot} blur {blur}");
            for k in 0..=20 {
                let phase = (0.3f32, 0.3 + 0.02 * k as f32);
                let truth = (C + phase.0, C + phase.1);
                let mut f = rotated_area_chessboard_f64(SIZE, 8, truth, rot.to_radians(), 90f32.to_radians(), 40, 200);
                blur_f64(&mut f, SIZE, blur as f64);
                let img = FImg { side: SIZE, data: f.iter().map(|&v| v as f32).collect() };
                let centre = [truth.0.round() as i32, truth.1.round() as i32];
                let seed = centre.map(|c| c as f32);
                let e = r.estimate(&img, centre, seed, 400, 1e-6).unwrap();
                println!(
                    "  ty {:.2}: w frac ({:.3}, {:.3}) err ({:7.4}, {:7.4})",
                    phase.1,
                    e.xy[0] - e.xy[0].floor(),
                    e.xy[1] - e.xy[1].floor(),
                    e.xy[0] - truth.0,
                    e.xy[1] - truth.1
                );
            }
        }
    }
}

/// Oracle: the same fixed-point iteration, but the window samples are the
/// exact pixel-area values at `w + p` (no interpolation).
#[test]
#[ignore]
fn uq15_oracle() {
    let r = SaddlePointRefiner::new(SaddlePointConfig::default());
    let op = r.fit_op.as_ref().unwrap();
    let cfg = SaddlePointConfig::default();
    let ph = phases(4);
    println!("rot | oracle rms (mpx) | bilinear rms (mpx)");
    let mut rr = SaddlePointRefiner::new(cfg);
    for rot in [0.0f32, 2.0, 5.0, 8.0, 15.0, 25.0, 45.0] {
        let mut errs = Vec::new();
        for &(px, py) in &ph {
            let truth = (C + px, C + py);
            let mut w = [truth.0.round(), truth.1.round()];
            for _ in 0..200 {
                // patch side 5 whose pixel (2,2) sits at w.
                let off = (truth.0 - w[0] + 2.0, truth.1 - w[1] + 2.0);
                let patch = rotated_area_chessboard_f64(5, 8, off, rot.to_radians(), 90f32.to_radians(), 40, 200);
                let mut coefs = [0.0f32; 5];
                for (col, z) in op.columns.iter().zip(&patch) {
                    for (c, rv) in coefs.iter_mut().zip(col) {
                        *c += rv * (*z as f32 - 120.0);
                    }
                }
                let s = solve(&cfg, coefs).unwrap();
                w = [w[0] + s.u[0], w[1] + s.u[1]];
                if s.u[0].hypot(s.u[1]) < 1e-6 {
                    break;
                }
            }
            errs.push([f64::from(w[0] - truth.0), f64::from(w[1] - truth.1)]);
        }
        let (eb, _) = errors(&mut rr, cond(8, rot, 0.0), &ph, false, MAX_STEPS);
        println!("{rot:4} | {:.2} | {:.2}", rms(&errs) * 1e3, rms(&eb) * 1e3);
    }
}

/// Does the converged point depend on the starting integer centre?
#[test]
#[ignore]
fn uq15_start_dependence() {
    let mut r = SaddlePointRefiner::new(SaddlePointConfig::default());
    for (rot, phase) in [(2.0f32, (0.3f32, 0.44f32)), (2.0, (0.3, 0.56)), (8.0, (0.31, 0.3))] {
        let truth = (C + phase.0, C + phase.1);
        let f = rotated_area_chessboard_f64(SIZE, 8, truth, rot.to_radians(), 90f32.to_radians(), 40, 200);
        let img = FImg { side: SIZE, data: f.iter().map(|&v| v as f32).collect() };
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)] {
            let centre = [truth.0.round() as i32 + dx, truth.1.round() as i32 + dy];
            let seed = centre.map(|c| c as f32);
            for (steps, tol) in [(MAX_STEPS, STEP_TOL), (400, 1e-6)] {
                let e = r.estimate(&img, centre, seed, steps, tol).unwrap();
                println!(
                    "rot {rot} phase {phase:?} start {centre:?} steps {steps} tol {tol}: err ({:.4}, {:.4}) cov {}",
                    e.xy[0] - truth.0,
                    e.xy[1] - truth.1,
                    e.covariance.is_some()
                );
            }
        }
    }
}

#[test]
#[ignore]
fn uq15_phase_map() {
    let n = 8;
    let ph = phases(n);
    let mut r = SaddlePointRefiner::new(SaddlePointConfig::default());
    for (rot, blur) in [(8.0f32, 0.0f32), (8.0, 2.5), (2.0, 0.0), (45.0, 0.0)] {
        let (e, _) = errors(&mut r, cond(8, rot, blur), &ph, false, MAX_STEPS);
        println!("rot {rot} blur {blur} (rows = phase y, cols = phase x), mpx");
        print_map(&e, n);
    }
}
