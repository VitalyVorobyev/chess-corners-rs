//! Evaluation harness for the orientation σ study (`UQ-06`,
//! `docs/reference/orientation-sigma.md`).
//!
//! For every ground-truth corner that the default detector finds (within
//! 1 px), it runs both orientation methods at the detected position and
//! writes one CSV row per method and axis with the angular error, the
//! shipped per-axis value (`AxisFitResult::fit_std_error_theta{1,2}`), and
//! the ingredients of the candidate σ definitions:
//!
//! - `c_unit`: diagonal of `(JᵀJ)⁻¹` for the axis angle of the fit that
//!   produced the result (ring: 16 bilinear samples; disk: support-disk
//!   pixels), so `sqrt(c_unit · s2)` is the residual-scaled covariance;
//! - `c_pix`: the axis-angle variance per unit *pixel* noise variance
//!   (`(JᵀJ)⁻¹ JᵀDJ (JᵀJ)⁻¹`, `D` = bilinear-weight energy per ring sample;
//!   `D = I` on the disk), so `sqrt(c_pix) · σ_pix` is noise-propagated;
//! - `s2`: residual variance `SSR / (N − 4)`.
//!
//! The shipped value is `sqrt(c_unit · s2)` (candidate `b`); `recon`
//! re-evaluates it from the ingredients as a self-check.
//!
//! The analysis lives in `tools/orientation_sigma/analyze.py`.
//!
//! ```text
//! cargo run --release -p chess-corners --example orientation_sigma_study -- optical out.csv
//! cargo run --release -p chess-corners --example orientation_sigma_study -- files list.tsv out.csv
//! cargo run --release -p chess-corners --example orientation_sigma_study -- cost
//! ```
//!
//! `files` reads a TSV whose lines are `image.png  truth.csv  tag`;
//! `truth.csv` has a header and `x,y,a1,a2` rows (pixel-centre coordinates,
//! true axis directions in radians; `nan` axes skip the corner) and `tag`
//! fills the first eight CSV columns
//! (`gen,cell,blur,rot_deg,shear_deg,noise,shading,seed`).

use chess_corners::{Detector, DetectorConfig};
use chess_corners_core::{
    estimate_noise_sigma_u8, fit_axes_at_point, AxisFitResult, ImageView, OrientationMethod,
};
use chess_corners_testutil::{add_gaussian_noise, optical_chessboard, Lattice};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::time::Instant;

const RING5: [(i32, i32); 16] = [
    (0, -5),
    (2, -5),
    (3, -3),
    (5, -2),
    (5, 0),
    (5, 2),
    (3, 3),
    (2, 5),
    (0, 5),
    (-2, 5),
    (-3, 3),
    (-5, 2),
    (-5, 0),
    (-5, -2),
    (-3, -3),
    (-2, -5),
];
const BETA: f64 = 4.0;
const RADIUS: u32 = 5;
const DISK_RADIUS: f64 = 8.0;
const DISK_WIDTHS: [f64; 4] = [0.35, 0.70, 1.40, 2.80];
const MATCH_PX: f32 = 1.0;
const BORDER: f32 = 12.0;

/// Ingredients of the candidate σ for one fit (both axes).
#[derive(Clone, Copy, Default)]
struct Ingredients {
    c_unit: [f64; 2],
    c_pix: [f64; 2],
    s2: f64,
    /// `sqrt(c_unit·s2)`, a self-check against the shipped value.
    recon: [f64; 2],
    /// Disk only: edge width whose `recon` reproduces the shipped value.
    width: f64,
}

fn line_delta_signed(a: f64, b: f64) -> f64 {
    // Signed undirected difference in (-π/2, π/2].
    let mut d = (a - b).rem_euclid(PI);
    if d > PI / 2.0 {
        d -= PI;
    }
    d
}

fn inv4(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut m = [[0.0f64; 8]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&a[i]);
        m[i][4 + i] = 1.0;
    }
    for i in 0..4 {
        let p = (i..4).max_by(|&r, &s| m[r][i].abs().total_cmp(&m[s][i].abs()))?;
        m.swap(i, p);
        let piv = m[i][i];
        if !piv.is_finite() || piv.abs() < 1e-300 {
            return None;
        }
        for v in m[i].iter_mut() {
            *v /= piv;
        }
        let row = m[i];
        for (k, r) in m.iter_mut().enumerate() {
            if k != i && r[i] != 0.0 {
                let f = r[i];
                for (x, y) in r.iter_mut().zip(row.iter()) {
                    *x -= f * y;
                }
            }
        }
    }
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        out[i].copy_from_slice(&m[i][4..]);
    }
    Some(out)
}

/// Linear LS for `(μ, A)` in `y = μ + A·q`; returns `(μ, A, SSR)`.
fn fit_mu_amp(q: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = q.len() as f64;
    let (sq, sy) = (q.iter().sum::<f64>(), y.iter().sum::<f64>());
    let sqq: f64 = q.iter().map(|v| v * v).sum();
    let sqy: f64 = q.iter().zip(y).map(|(a, b)| a * b).sum();
    let den = sqq - sq * sq / n;
    let amp = if den.abs() > 1e-12 {
        (sqy - sq * sy / n) / den
    } else {
        0.0
    };
    let mu = (sy - amp * sq) / n;
    let ssr = q
        .iter()
        .zip(y)
        .map(|(a, b)| (b - mu - amp * a).powi(2))
        .sum();
    (mu, amp, ssr)
}

/// Accumulate `JᵀJ` and `JᵀDJ` from rows `[1, q, ∂/∂θ₁, ∂/∂θ₂]` and
/// finish the ingredients.
fn finish(rows: &[[f64; 4]], d: &[f64], ssr: f64) -> Option<([f64; 2], [f64; 2], f64)> {
    let mut n = [[0.0; 4]; 4];
    let mut nd = [[0.0; 4]; 4];
    for (j, &dw) in rows.iter().zip(d) {
        for k in 0..4 {
            for l in 0..4 {
                n[k][l] += j[k] * j[l];
                nd[k][l] += dw * j[k] * j[l];
            }
        }
    }
    let ni = inv4(&n)?;
    let mut c_pix = [0.0; 2];
    for (a, slot) in c_pix.iter_mut().enumerate() {
        let r = &ni[2 + a];
        let mut acc = 0.0;
        for k in 0..4 {
            for l in 0..4 {
                acc += r[k] * nd[k][l] * r[l];
            }
        }
        *slot = acc;
    }
    let s2 = ssr / (rows.len() as f64 - 4.0);
    Some(([ni[2][2], ni[3][3]], c_pix, s2))
}

/// Ring-fit ingredients at the reported axes.
fn ring_ingredients(
    view: &ImageView<'_>,
    x: f32,
    y: f32,
    fit: &AxisFitResult,
) -> Option<Ingredients> {
    let (t1, t2) = (fit.theta1 as f64, fit.theta2 as f64);
    let mut q = [0.0; 16];
    let mut s = [0.0; 16];
    let mut d = [0.0; 16];
    let mut parts = [(0.0, 0.0, 0.0, 0.0, 0.0); 16];
    for (i, &(dx, dy)) in RING5.iter().enumerate() {
        let (sx, sy) = (x + dx as f32, y + dy as f32);
        s[i] = view.sample_bilinear(sx, sy) as f64;
        let (tx, ty) = ((sx - sx.floor()) as f64, (sy - sy.floor()) as f64);
        d[i] = ((1.0 - tx).powi(2) + tx * tx) * ((1.0 - ty).powi(2) + ty * ty);
        let phi = (dy as f64).atan2(dx as f64);
        let h1 = (BETA * (phi - t1).sin()).tanh();
        let h2 = (BETA * (phi - t2).sin()).tanh();
        q[i] = h1 * h2;
        parts[i] = (h1, h2, (phi - t1).cos(), (phi - t2).cos(), phi);
    }
    let (_mu, amp, ssr) = fit_mu_amp(&q, &s);
    let rows: Vec<[f64; 4]> = parts
        .iter()
        .zip(q)
        .map(|(&(h1, h2, c1, c2, _), qi)| {
            [
                1.0,
                qi,
                -amp * (1.0 - h1 * h1) * BETA * c1 * h2,
                -amp * h1 * (1.0 - h2 * h2) * BETA * c2,
            ]
        })
        .collect();
    let (c_unit, c_pix, s2) = finish(&rows, &d, ssr)?;
    Some(Ingredients {
        c_unit,
        c_pix,
        s2,
        recon: [(c_unit[0] * s2).sqrt(), (c_unit[1] * s2).sqrt()],
        width: f64::NAN,
    })
}

/// Support-disk pixel offsets from the corner and their values.
type Disk = (Vec<(f64, f64)>, Vec<f64>);

fn disk_pixels(view: &ImageView<'_>, x: f32, y: f32) -> Option<Disk> {
    let (cx, cy) = (x as f64, y as f64);
    let (x0, x1) = (
        (cx - DISK_RADIUS).floor() as i64,
        (cx + DISK_RADIUS).ceil() as i64,
    );
    let (y0, y1) = (
        (cy - DISK_RADIUS).floor() as i64,
        (cy + DISK_RADIUS).ceil() as i64,
    );
    if x0 < 0 || y0 < 0 || x1 >= view.width() as i64 || y1 >= view.height() as i64 {
        return None;
    }
    let (mut p, mut v) = (Vec::new(), Vec::new());
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            let (dx, dy) = (xx as f64 - cx, yy as f64 - cy);
            let rr = dx * dx + dy * dy;
            if rr > DISK_RADIUS * DISK_RADIUS || rr < 1.0 {
                continue;
            }
            p.push((dx, dy));
            v.push(view.data()[yy as usize * view.width() + xx as usize] as f64);
        }
    }
    Some((p, v))
}

/// Disk-fit ingredients at the reported axes for the given edge width.
fn disk_ingredients_at(
    px: &[(f64, f64)],
    vals: &[f64],
    t1: f64,
    t2: f64,
    w: f64,
) -> Option<Ingredients> {
    let (s1, c1) = t1.sin_cos();
    let (s2, c2) = t2.sin_cos();
    let mut q = Vec::with_capacity(px.len());
    let mut parts = Vec::with_capacity(px.len());
    for &(dx, dy) in px {
        let h1 = ((-s1 * dx + c1 * dy) / w).tanh();
        let h2 = ((-s2 * dx + c2 * dy) / w).tanh();
        q.push(h1 * h2);
        // ∂d/∂θ = −cosθ·dx − sinθ·dy.
        parts.push((h1, h2, -c1 * dx - s1 * dy, -c2 * dx - s2 * dy));
    }
    let (_mu, amp, ssr) = fit_mu_amp(&q, vals);
    let rows: Vec<[f64; 4]> = parts
        .iter()
        .zip(&q)
        .map(|(&(h1, h2, e1, e2), &qi)| {
            [
                1.0,
                qi,
                amp * (1.0 - h1 * h1) / w * e1 * h2,
                amp * h1 * (1.0 - h2 * h2) / w * e2,
            ]
        })
        .collect();
    let ones = vec![1.0; rows.len()];
    let (c_unit, c_pix, s2v) = finish(&rows, &ones, ssr)?;
    Some(Ingredients {
        c_unit,
        c_pix,
        s2: s2v,
        recon: [(c_unit[0] * s2v).sqrt(), (c_unit[1] * s2v).sqrt()],
        width: w,
    })
}

/// Disk ingredients: the width whose `recon` reproduces the shipped value,
/// else the width with the smallest residual.
fn disk_ingredients(
    view: &ImageView<'_>,
    x: f32,
    y: f32,
    fit: &AxisFitResult,
) -> Option<Ingredients> {
    let (px, vals) = disk_pixels(view, x, y)?;
    let (t1, t2) = (fit.theta1 as f64, fit.theta2 as f64);
    let all: Vec<Ingredients> = DISK_WIDTHS
        .iter()
        .filter_map(|&w| disk_ingredients_at(&px, &vals, t1, t2, w))
        .collect();
    let shipped = fit.fit_std_error_theta1 as f64;
    all.iter().copied().min_by(|a, b| {
        (a.recon[0] - shipped)
            .abs()
            .total_cmp(&(b.recon[0] - shipped).abs())
            .then(a.s2.total_cmp(&b.s2))
    })
}

struct Truth {
    x: f32,
    y: f32,
    a: [f64; 2],
}

/// Evaluate one image; returns CSV rows (no header) and the match count.
fn evaluate(img: &[u8], w: usize, h: usize, truth: &[Truth], prefix: &str) -> (String, usize) {
    let cfg = DetectorConfig::chess().without_orientation();
    let mut det = Detector::new(cfg).expect("default config is valid");
    let corners = det
        .detect_u8(img, w as u32, h as u32)
        .expect("dimensions match");
    let noise_est = estimate_noise_sigma_u8(img, w, h).map_or(f64::NAN, f64::from);
    let view = ImageView::from_u8_slice(w, h, img).expect("dimensions match");
    let mut out = String::new();
    let mut matched = 0;
    for (id, t) in truth.iter().enumerate() {
        if t.x < BORDER
            || t.y < BORDER
            || t.x > w as f32 - 1.0 - BORDER
            || t.y > h as f32 - 1.0 - BORDER
        {
            continue;
        }
        if !t.a[0].is_finite() || !t.a[1].is_finite() {
            continue;
        }
        let best = corners
            .iter()
            .map(|c| ((c.x - t.x).hypot(c.y - t.y), c))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((dist, c)) = best else { continue };
        if dist > MATCH_PX {
            continue;
        }
        matched += 1;
        let ring = fit_axes_at_point(view, c.x, c.y, RADIUS, OrientationMethod::RingFit);
        let disk = fit_axes_at_point(view, c.x, c.y, RADIUS, OrientationMethod::DiskFit);
        let same = ring.theta1 == disk.theta1 && ring.theta2 == disk.theta2;
        let disk_path = if !same {
            "disk"
        } else if disk.fit_std_error_theta1 > ring.fit_std_error_theta1
            || disk.fit_std_error_theta2 > ring.fit_std_error_theta2
        {
            "inflated"
        } else {
            "ring"
        };
        let ring_ing = ring_ingredients(&view, c.x, c.y, &ring);
        for (method, fit, path) in [("ring", &ring, "ring"), ("disk", &disk, disk_path)] {
            let ing = if path == "disk" {
                disk_ingredients(&view, c.x, c.y, fit)
            } else {
                ring_ing
            };
            let ing = ing.unwrap_or(Ingredients {
                c_unit: [f64::NAN; 2],
                c_pix: [f64::NAN; 2],
                s2: f64::NAN,
                recon: [f64::NAN; 2],
                width: f64::NAN,
            });
            let est = [fit.theta1 as f64, fit.theta2 as f64];
            let sig = [
                fit.fit_std_error_theta1 as f64,
                fit.fit_std_error_theta2 as f64,
            ];
            let direct =
                line_delta_signed(est[0], t.a[0]).abs() + line_delta_signed(est[1], t.a[1]).abs();
            let swapped =
                line_delta_signed(est[0], t.a[1]).abs() + line_delta_signed(est[1], t.a[0]).abs();
            let truth_for = if direct <= swapped {
                [t.a[0], t.a[1]]
            } else {
                [t.a[1], t.a[0]]
            };
            let sep = (est[1] - est[0]).rem_euclid(PI);
            for k in 0..2 {
                let _ = writeln!(
                    out,
                    "{prefix},{id},{:.4},{:.4},{:.4},{:.4},{noise_est:.4},{method},{path},{k},{:.7},{:.7},{:.6e},{:.6e},{:.6e},{:.7},{:.3},{:.3},{:.5},{:.4}",
                    t.x,
                    t.y,
                    c.x,
                    c.y,
                    line_delta_signed(est[k], truth_for[k]),
                    sig[k],
                    ing.c_unit[k],
                    ing.c_pix[k],
                    ing.s2,
                    ing.recon[k],
                    ing.width,
                    fit.amp,
                    fit.rms,
                    sep.min(PI - sep).to_degrees(),
                );
            }
        }
    }
    (out, matched)
}

const HEADER: &str = "gen,cell,blur,rot_deg,shear_deg,noise,shading,seed,id,x_true,y_true,x,y,noise_est,method,path,axis,err,fit_std_error,c_unit,c_pix,s2,recon,width,amp,rms,sep_deg";

fn run_optical(out_path: &str) {
    let cells = [6.0f32, 10.0, 20.0];
    let blurs = [0.5f32, 1.0, 1.5, 2.5];
    let rots = [0.0f32, 15.0, 30.0, 45.0];
    let shears = [90.0f32, 60.0, 40.0];
    let noises = [0.0f32, 1.0, 2.0, 4.0, 8.0];
    let seeds = [1u64, 2];
    let mut jobs = Vec::new();
    for &cell in &cells {
        for &blur in &blurs {
            for &rot in &rots {
                for &shear in &shears {
                    jobs.push((cell, blur, rot, shear));
                }
            }
        }
    }
    let t0 = Instant::now();
    let chunks: Vec<String> = jobs
        .par_iter()
        .map(|&(cell, blur, rot, shear)| {
            let size = ((cell * 10.0) as usize).clamp(96, 200);
            let origin = (size as f32 / 2.0 + 0.37, size as f32 / 2.0 + 0.21);
            let lattice = Lattice::new(cell, origin)
                .rotated(rot.to_radians())
                .with_axis_angle(shear.to_radians());
            let clean = optical_chessboard(size, &lattice, blur, 40, 200);
            let (r, a) = (rot.to_radians() as f64, (rot + shear).to_radians() as f64);
            let (e1, e2) = ((r.cos(), r.sin()), (a.cos(), a.sin()));
            let mut truth = Vec::new();
            let n = (size as f32 / cell) as i32 + 2;
            for k in -n..=n {
                for m in -n..=n {
                    let x = origin.0 as f64 + cell as f64 * (k as f64 * e1.0 + m as f64 * e2.0);
                    let y = origin.1 as f64 + cell as f64 * (k as f64 * e1.1 + m as f64 * e2.1);
                    truth.push(Truth {
                        x: x as f32,
                        y: y as f32,
                        a: [r, a],
                    });
                }
            }
            let mut s = String::new();
            for &noise in &noises {
                for &seed in &seeds {
                    if noise == 0.0 && seed > 1 {
                        continue;
                    }
                    let mut img = clean.clone();
                    add_gaussian_noise(
                        &mut img,
                        noise,
                        seed * 7919 + (cell as u64) * 31 + (rot as u64),
                    );
                    let prefix = format!("optical,{cell},{blur},{rot},{shear},{noise},none,{seed}");
                    let (rows, _) = evaluate(&img, size, size, &truth, &prefix);
                    s.push_str(&rows);
                }
            }
            s
        })
        .collect();
    let mut all = String::from(HEADER);
    all.push('\n');
    for c in chunks {
        all.push_str(&c);
    }
    std::fs::write(out_path, all).expect("write output");
    eprintln!(
        "optical: {} renders in {:.1}s",
        jobs.len(),
        t0.elapsed().as_secs_f64()
    );
}

fn run_files(list: &str, out_path: &str) {
    let text = std::fs::read_to_string(list).expect("read list");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let chunks: Vec<(String, usize, usize)> = lines
        .par_iter()
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let img = image::open(f[0]).expect("open image").to_luma8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let truth: Vec<Truth> = std::fs::read_to_string(f[1])
                .expect("read truth")
                .lines()
                .skip(1)
                .map(|l| {
                    let v: Vec<f64> = l
                        .split(',')
                        .map(|s| s.trim().parse().unwrap_or(f64::NAN))
                        .collect();
                    Truth {
                        x: v[0] as f32,
                        y: v[1] as f32,
                        a: [v[2], v[3]],
                    }
                })
                .collect();
            let prefix = f[2].to_string();
            let (rows, m) = evaluate(img.as_raw(), w, h, &truth, &prefix);
            (rows, m, truth.len())
        })
        .collect();
    let mut all = String::from(HEADER);
    all.push('\n');
    let (mut m, mut t) = (0, 0);
    for (c, mm, tt) in chunks {
        all.push_str(&c);
        m += mm;
        t += tt;
    }
    std::fs::write(out_path, all).expect("write output");
    eprintln!(
        "files: {} images, {m} of {t} truth corners matched",
        lines.len()
    );
}

/// Per-corner timing of the shipped fits and of the candidate
/// ingredients, on noisy optical boards (single thread, release).
fn run_cost() {
    let mut cases = Vec::new();
    for (cell, blur, shear) in [
        (10.0f32, 1.0f32, 90.0f32),
        (10.0, 1.5, 60.0),
        (20.0, 2.5, 60.0),
    ] {
        let size = 200usize;
        let origin = (100.37f32, 100.21f32);
        let lat = Lattice::new(cell, origin)
            .rotated(15f32.to_radians())
            .with_axis_angle(shear.to_radians());
        let mut img = optical_chessboard(size, &lat, blur, 40, 200);
        add_gaussian_noise(&mut img, 2.0, 5);
        let mut pts = Vec::new();
        let (r, a) = (15f32.to_radians(), (15.0 + shear).to_radians());
        for k in -6..=6 {
            for m in -6..=6 {
                let x = origin.0 + cell * (k as f32 * r.cos() + m as f32 * a.cos());
                let y = origin.1 + cell * (k as f32 * r.sin() + m as f32 * a.sin());
                if x > 14.0 && y > 14.0 && x < 185.0 && y < 185.0 {
                    pts.push((x + 0.05, y - 0.04));
                }
            }
        }
        cases.push((format!("cell {cell} blur {blur} shear {shear}"), img, pts));
    }
    for (name, img, pts) in &cases {
        let view = ImageView::from_u8_slice(200, 200, img).unwrap();
        let time = |f: &dyn Fn() -> f64| {
            let reps = 20;
            let mut acc = 0.0;
            let t = Instant::now();
            for _ in 0..reps {
                acc += f();
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as f64 / (reps * pts.len()) as f64
        };
        let ring_fits: Vec<AxisFitResult> = pts
            .iter()
            .map(|&(x, y)| fit_axes_at_point(view, x, y, RADIUS, OrientationMethod::RingFit))
            .collect();
        let disk_fits: Vec<AxisFitResult> = pts
            .iter()
            .map(|&(x, y)| fit_axes_at_point(view, x, y, RADIUS, OrientationMethod::DiskFit))
            .collect();
        let n_disk = disk_fits
            .iter()
            .zip(&ring_fits)
            .filter(|(d, r)| d.theta1 != r.theta1 || d.theta2 != r.theta2)
            .count();
        let t_ring = time(&|| {
            pts.iter()
                .map(|&(x, y)| {
                    fit_axes_at_point(view, x, y, RADIUS, OrientationMethod::RingFit)
                        .fit_std_error_theta1 as f64
                })
                .sum()
        });
        let t_disk = time(&|| {
            pts.iter()
                .map(|&(x, y)| {
                    fit_axes_at_point(view, x, y, RADIUS, OrientationMethod::DiskFit)
                        .fit_std_error_theta1 as f64
                })
                .sum()
        });
        let t_ring_ing = time(&|| {
            pts.iter()
                .zip(&ring_fits)
                .map(|(&(x, y), f)| ring_ingredients(&view, x, y, f).map_or(0.0, |i| i.c_unit[0]))
                .sum()
        });
        // One Jacobian pass at a known width (what a production disk fit
        // would add: it already holds the disk pixels and the width).
        let disks: Vec<_> = pts
            .iter()
            .map(|&(x, y)| disk_pixels(&view, x, y).unwrap())
            .collect();
        let t_disk_ing = time(&|| {
            disks
                .iter()
                .zip(&disk_fits)
                .map(|((p, v), f)| {
                    disk_ingredients_at(p, v, f.theta1 as f64, f.theta2 as f64, 1.4)
                        .map_or(0.0, |i| i.c_unit[0])
                })
                .sum()
        });
        println!(
            "{name}: {} corners ({n_disk} disk-accepted) | RingFit {t_ring:.0} ns | DiskFit {t_disk:.0} ns | ring ingredients (f64, from scratch) {t_ring_ing:.0} ns | disk Jacobian pass (f64) {t_disk_ing:.0} ns",
            pts.len()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("optical") => run_optical(&args[2]),
        Some("files") => run_files(&args[2], &args[3]),
        Some("cost") => run_cost(),
        _ => eprintln!("usage: orientation_sigma_study optical OUT | files LIST OUT | cost"),
    }
}
