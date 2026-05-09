//! Full-disk two-axis orientation estimator.
//!
//! This module ports the benchmark `disk_sector_py` prototype into the
//! core crate as an opt-in method. It keeps the detected center fixed,
//! samples the image disk around that center, searches candidate
//! crossing lines, and only publishes the disk result when the
//! full-area model is confident. Otherwise it returns the sigma-LUT
//! fallback unchanged.

use core::f32::consts::PI;

use super::{baseline, sigma_correction};

const WIDTHS_PX: [f32; 4] = [0.35, 0.70, 1.40, 2.80];
const SUPPORT_SCALE: f32 = 1.6;
const MAX_SUPPORT_RADIUS: f32 = 8.0;
const MAX_SIDE: usize = 34;
const MAX_SUPPORT: usize = MAX_SIDE * MAX_SIDE;
const HIST_BINS: usize = 72;
const MAX_CANDIDATES: usize = 64;
const TOP_FITS: usize = 4;

const INNER_RADIUS: f32 = 1.0;
const MIN_SUPPORT: usize = 64;
const MIN_CONTRAST: f32 = 10.0;
const MIN_CORR: f32 = 0.74;
const ACCEPT_REL_MARGIN: f32 = 0.03;
const ACCEPT_REL_RATIO: f32 = 0.92;
const WEAK_DISAGREE_RAD: f32 = 12.0_f32.to_radians();
const INFLATED_SIGMA_RAD: f32 = 10.0_f32.to_radians();

#[derive(Clone, Copy, Debug)]
struct Fit {
    theta1: f32,
    theta2: f32,
    width: f32,
    amp: f32,
    rms: f32,
    rel_rms: f32,
    corr: f32,
    edge_score: f32,
    objective: f32,
}

#[derive(Clone)]
struct DiskData {
    n: usize,
    xs: [f32; MAX_SUPPORT],
    ys: [f32; MAX_SUPPORT],
    vals: [f32; MAX_SUPPORT],
    vals_centered: [f32; MAX_SUPPORT],
    grad_angles: [f32; MAX_SUPPORT],
    grad_weights: [f32; MAX_SUPPORT],
    mean: f32,
    val_energy: f32,
}

impl DiskData {
    fn new() -> Self {
        Self {
            n: 0,
            xs: [0.0; MAX_SUPPORT],
            ys: [0.0; MAX_SUPPORT],
            vals: [0.0; MAX_SUPPORT],
            vals_centered: [0.0; MAX_SUPPORT],
            grad_angles: [0.0; MAX_SUPPORT],
            grad_weights: [0.0; MAX_SUPPORT],
            mean: 0.0,
            val_energy: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct CandidateSet {
    n: usize,
    angles: [f32; MAX_CANDIDATES],
}

impl CandidateSet {
    fn new() -> Self {
        Self {
            n: 0,
            angles: [0.0; MAX_CANDIDATES],
        }
    }

    fn push_unique(&mut self, angle: f32, min_sep: f32) {
        if self.n >= MAX_CANDIDATES {
            return;
        }
        let a = wrap_pi(angle);
        for &existing in &self.angles[..self.n] {
            if line_delta(a, existing) <= min_sep {
                return;
            }
        }
        self.angles[self.n] = a;
        self.n += 1;
    }
}

/// Full-disk estimator used by the public image-side dispatcher.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit(
    img: &[u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    radius: u32,
    samples: &[f32; 16],
    ring_phi: &[f32; 16],
) -> baseline::TwoAxisFit {
    let fallback = sigma_correction::fit_lut(samples, ring_phi);
    let Some(data) = extract_disk(img, w, h, cx, cy, radius) else {
        return fallback;
    };
    let Some(fit) = best_disk_fit(&data, fallback.theta1, fallback.theta2) else {
        return fallback;
    };

    let base_rel = fallback.rms.max(0.0) / fallback.amp.max(1.0);
    let disagreement = pair_disagreement(fit.theta1, fit.theta2, fallback.theta1, fallback.theta2);
    let sep = line_separation(fit.theta1, fit.theta2);

    let accept_by_margin = fit.rel_rms <= base_rel - ACCEPT_REL_MARGIN;
    let accept_by_ratio = fit.rel_rms <= base_rel * ACCEPT_REL_RATIO;
    let strong_nonorthogonal = sep < 55.0_f32.to_radians();
    let sharp_orthogonal = sep >= 75.0_f32.to_radians() && fit.width <= 0.70;
    let edge_disagreement =
        disagreement > WEAK_DISAGREE_RAD && fit.edge_score >= 0.18 && fit.rel_rms <= 0.45;
    let near_orthogonal_blur = sep >= 65.0_f32.to_radians()
        && fit.width >= 2.80
        && disagreement <= WEAK_DISAGREE_RAD
        && fit.rel_rms > base_rel - 0.08;
    let residual_accept = (accept_by_margin || accept_by_ratio) && !near_orthogonal_blur;

    let accepted = fit.amp >= MIN_CONTRAST
        && fit.corr >= MIN_CORR
        && fit.edge_score >= 0.035
        && fit.rel_rms.is_finite()
        && (residual_accept || edge_disagreement || strong_nonorthogonal || sharp_orthogonal);

    if accepted {
        let sigma = sigma_for_fit(&fit);
        baseline::TwoAxisFit {
            amp: fit.amp,
            theta1: fit.theta1,
            theta2: fit.theta2,
            sigma_theta1: sigma,
            sigma_theta2: sigma,
            rms: fit.rms,
        }
    } else if disagreement > WEAK_DISAGREE_RAD {
        baseline::TwoAxisFit {
            sigma_theta1: fallback.sigma_theta1.max(INFLATED_SIGMA_RAD),
            sigma_theta2: fallback.sigma_theta2.max(INFLATED_SIGMA_RAD),
            ..fallback
        }
    } else {
        fallback
    }
}

fn extract_disk(img: &[u8], w: usize, h: usize, cx: f32, cy: f32, radius: u32) -> Option<DiskData> {
    if w == 0 || h == 0 || img.len() < w.saturating_mul(h) || !cx.is_finite() || !cy.is_finite() {
        return None;
    }

    let support_radius = (radius as f32 * SUPPORT_SCALE).min(MAX_SUPPORT_RADIUS);
    if support_radius < 1.0 {
        return None;
    }

    let x0 = (cx - support_radius).floor() as i32;
    let x1 = (cx + support_radius).ceil() as i32;
    let y0 = (cy - support_radius).floor() as i32;
    let y1 = (cy + support_radius).ceil() as i32;
    if x0 < 0 || y0 < 0 || x1 >= w as i32 || y1 >= h as i32 {
        return None;
    }
    if (x1 - x0 + 1) as usize > MAX_SIDE || (y1 - y0 + 1) as usize > MAX_SIDE {
        return None;
    }

    let mut data = DiskData::new();
    let r2 = support_radius * support_radius;
    let inner2 = INNER_RADIUS * INNER_RADIUS;
    let mut sum = 0.0f32;

    for yy in y0..=y1 {
        let dy = yy as f32 - cy;
        for xx in x0..=x1 {
            let dx = xx as f32 - cx;
            let rr = dx * dx + dy * dy;
            if rr > r2 || rr < inner2 {
                continue;
            }
            if data.n >= MAX_SUPPORT {
                return None;
            }
            let idx = yy as usize * w + xx as usize;
            let val = img[idx] as f32;
            let (gx, gy) = sobel_at(img, w, h, xx, yy);
            let weight = (gx * gx + gy * gy).sqrt();
            data.xs[data.n] = dx;
            data.ys[data.n] = dy;
            data.vals[data.n] = val;
            data.grad_weights[data.n] = weight;
            data.grad_angles[data.n] = wrap_pi(gy.atan2(gx) + PI * 0.5);
            sum += val;
            data.n += 1;
        }
    }

    if data.n < MIN_SUPPORT {
        return None;
    }

    data.mean = sum / data.n as f32;
    let mut val_energy = 0.0f32;
    for i in 0..data.n {
        let centered = data.vals[i] - data.mean;
        data.vals_centered[i] = centered;
        val_energy += centered * centered;
    }
    data.val_energy = val_energy;
    Some(data)
}

fn sobel_at(img: &[u8], w: usize, h: usize, x: i32, y: i32) -> (f32, f32) {
    let p = |xx: i32, yy: i32| -> f32 {
        let xc = xx.clamp(0, w.saturating_sub(1) as i32) as usize;
        let yc = yy.clamp(0, h.saturating_sub(1) as i32) as usize;
        img[yc * w + xc] as f32
    };

    let tl = p(x - 1, y - 1);
    let tc = p(x, y - 1);
    let tr = p(x + 1, y - 1);
    let ml = p(x - 1, y);
    let mr = p(x + 1, y);
    let bl = p(x - 1, y + 1);
    let bc = p(x, y + 1);
    let br = p(x + 1, y + 1);

    let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    (gx, gy)
}

fn histogram_candidates(data: &DiskData, base0: f32, base1: f32) -> CandidateSet {
    let mut candidates = CandidateSet::new();
    let mut hist = [0.0f32; HIST_BINS];
    for i in 0..data.n {
        let weight = data.grad_weights[i];
        if weight <= 0.0 {
            continue;
        }
        let idx = ((data.grad_angles[i] / PI) * HIST_BINS as f32).floor() as usize % HIST_BINS;
        hist[idx] += weight;
    }

    let mut smooth = [0.0f32; HIST_BINS];
    for i in 0..HIST_BINS {
        let prev = if i == 0 { HIST_BINS - 1 } else { i - 1 };
        let next = if i + 1 == HIST_BINS { 0 } else { i + 1 };
        smooth[i] = 0.25 * hist[prev] + 0.5 * hist[i] + 0.25 * hist[next];
    }

    let mut max_v = 0.0f32;
    for &v in &smooth {
        max_v = max_v.max(v);
    }
    if max_v > 0.0 {
        let mut used = [false; HIST_BINS];
        for _ in 0..8 {
            let mut best_i = 0usize;
            let mut best_v = -1.0f32;
            for i in 0..HIST_BINS {
                if !used[i] && smooth[i] > best_v {
                    best_i = i;
                    best_v = smooth[i];
                }
            }
            if best_v < max_v * 0.12 {
                break;
            }
            used[best_i] = true;
            let angle = (best_i as f32 + 0.5) * PI / HIST_BINS as f32;
            candidates.push_unique(angle, 4.0_f32.to_radians());
        }
    }

    for seed in [base0, base1] {
        for off_deg in [-8.0f32, -4.0, 0.0, 4.0, 8.0] {
            candidates.push_unique(seed + off_deg.to_radians(), 1.0_f32.to_radians());
        }
    }

    let mut deg = 0;
    while deg < 180 {
        candidates.push_unique((deg as f32).to_radians(), 1.0_f32.to_radians());
        deg += 30;
    }

    candidates
}

fn best_disk_fit(data: &DiskData, base0: f32, base1: f32) -> Option<Fit> {
    let candidates = histogram_candidates(data, wrap_pi(base0), wrap_pi(base1));
    let mut top: [Option<Fit>; TOP_FITS] = [None; TOP_FITS];

    for i in 0..candidates.n {
        let a0 = candidates.angles[i];
        for j in i + 1..candidates.n {
            let a1 = candidates.angles[j];
            if !valid_pair(a0, a1) {
                continue;
            }
            let edge_score = edge_pair_score(data, a0, a1);
            for width in WIDTHS_PX {
                if let Some(fit) = score_pair(data, a0, a1, width, edge_score) {
                    insert_top(&mut top, fit);
                }
            }
        }
    }

    let mut best = top[0]?;
    for seed in top.iter().flatten().copied() {
        let refined = refine_fit(data, seed);
        if fit_is_better(refined, best) {
            best = refined;
        }
    }
    Some(best)
}

fn insert_top(top: &mut [Option<Fit>; TOP_FITS], fit: Fit) {
    let mut pos = TOP_FITS;
    for (i, existing) in top.iter().enumerate() {
        match existing {
            Some(v) if fit_is_better(fit, *v) => {
                pos = i;
                break;
            }
            None => {
                pos = i;
                break;
            }
            _ => {}
        }
    }
    if pos == TOP_FITS {
        return;
    }
    for i in (pos + 1..TOP_FITS).rev() {
        top[i] = top[i - 1];
    }
    top[pos] = Some(fit);
}

fn refine_fit(data: &DiskData, seed: Fit) -> Fit {
    let mut best = seed;
    for step_deg in [2.0f32, 1.0, 0.5, 0.25] {
        let step = step_deg.to_radians();
        for d0 in [-step, 0.0, step] {
            for d1 in [-step, 0.0, step] {
                if d0 == 0.0 && d1 == 0.0 {
                    continue;
                }
                if let Some(fit) =
                    score_pair(data, best.theta1 + d0, best.theta2 + d1, best.width, -1.0)
                {
                    if fit_is_better(fit, best) {
                        best = fit;
                    }
                }
            }
        }
    }
    best
}

fn score_pair(
    data: &DiskData,
    theta0: f32,
    theta1: f32,
    width: f32,
    edge_score: f32,
) -> Option<Fit> {
    if !valid_pair(theta0, theta1) {
        return None;
    }
    if data.val_energy <= 1e-9 {
        return None;
    }

    let (s0, c0) = theta0.sin_cos();
    let (s1, c1) = theta1.sin_cos();
    let mut sum_q = 0.0f32;
    let mut sum_q2 = 0.0f32;
    let mut dot = 0.0f32;

    for i in 0..data.n {
        let d0 = -s0 * data.xs[i] + c0 * data.ys[i];
        let d1 = -s1 * data.xs[i] + c1 * data.ys[i];
        let q = (d0 / width).tanh() * (d1 / width).tanh();
        sum_q += q;
        sum_q2 += q * q;
        dot += q * data.vals_centered[i];
    }

    let n_inv = 1.0 / data.n as f32;
    let q_mean = sum_q * n_inv;
    let denom = sum_q2 - sum_q * sum_q * n_inv;
    if denom <= 1e-9 {
        return None;
    }

    let amp = dot / denom;
    let mu = data.mean - amp * q_mean;
    let mut ssr = 0.0f32;

    for i in 0..data.n {
        let d0 = -s0 * data.xs[i] + c0 * data.ys[i];
        let d1 = -s1 * data.xs[i] + c1 * data.ys[i];
        let q = (d0 / width).tanh() * (d1 / width).tanh();
        let residual = data.vals[i] - (mu + amp * q);
        ssr += residual * residual;
    }

    let rms = (ssr * n_inv).sqrt();
    let corr = dot.abs() / (denom * data.val_energy).sqrt();
    let rel_rms = rms / amp.abs().max(1.0);
    let edge = if edge_score >= 0.0 {
        edge_score
    } else {
        edge_pair_score(data, theta0, theta1)
    };
    let objective = rel_rms - 1.25 * edge;
    let (theta1_c, theta2_c, amp_c) = baseline::canonicalize(theta0, theta1, amp);

    Some(Fit {
        theta1: theta1_c,
        theta2: theta2_c,
        width,
        amp: amp_c,
        rms,
        rel_rms,
        corr,
        edge_score: edge,
        objective,
    })
}

fn edge_pair_score(data: &DiskData, theta0: f32, theta1: f32) -> f32 {
    let mut total = 0.0f32;
    for i in 0..data.n {
        total += data.grad_weights[i];
    }
    if total <= 1e-9 {
        return 0.0;
    }

    let sigma = 4.0_f32.to_radians();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    for i in 0..data.n {
        let w = data.grad_weights[i];
        let d0 = signed_line_delta(data.grad_angles[i], theta0);
        let d1 = signed_line_delta(data.grad_angles[i], theta1);
        s0 += w * (-0.5 * (d0 / sigma) * (d0 / sigma)).exp();
        s1 += w * (-0.5 * (d1 / sigma) * (d1 / sigma)).exp();
    }

    s0 /= total;
    s1 /= total;
    let balance = 2.0 * s0.min(s1) / (s0 + s1).max(1e-9);
    (s0 + s1) * balance
}

fn fit_is_better(a: Fit, b: Fit) -> bool {
    if a.objective < b.objective - 1e-9 {
        return true;
    }
    if (a.objective - b.objective).abs() > 1e-9 {
        return false;
    }
    if a.rel_rms < b.rel_rms - 1e-9 {
        return true;
    }
    if (a.rel_rms - b.rel_rms).abs() > 1e-9 {
        return false;
    }
    if a.edge_score > b.edge_score + 1e-9 {
        return true;
    }
    if (a.edge_score - b.edge_score).abs() > 1e-9 {
        return false;
    }
    if a.width < b.width - 1e-9 {
        return true;
    }
    if (a.width - b.width).abs() > 1e-9 {
        return false;
    }
    if a.theta1 < b.theta1 - 1e-9 {
        return true;
    }
    if (a.theta1 - b.theta1).abs() > 1e-9 {
        return false;
    }
    a.theta2 < b.theta2
}

fn sigma_for_fit(fit: &Fit) -> f32 {
    let sep_deg = line_separation(fit.theta1, fit.theta2).to_degrees();
    let floor_deg = if sep_deg >= 55.0 { 1.5 } else { 3.0 };
    let model_deg = floor_deg + (8.0 * fit.rel_rms).min(6.0);
    (model_deg * 0.55).to_radians()
}

fn valid_pair(a0: f32, a1: f32) -> bool {
    let sep = line_separation(a0, a1);
    (12.0_f32.to_radians()..=89.5_f32.to_radians()).contains(&sep)
}

fn line_separation(a0: f32, a1: f32) -> f32 {
    line_delta(a0, a1)
}

fn pair_disagreement(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    let direct = line_delta(a0, b0).max(line_delta(a1, b1));
    let swapped = line_delta(a0, b1).max(line_delta(a1, b0));
    direct.min(swapped)
}

fn line_delta(a: f32, b: f32) -> f32 {
    signed_line_delta(a, b).abs()
}

fn signed_line_delta(a: f32, b: f32) -> f32 {
    (a - b + PI * 0.5).rem_euclid(PI) - PI * 0.5
}

fn wrap_pi(a: f32) -> f32 {
    a.rem_euclid(PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{ring_angles, sample_ring};
    use crate::ring::ring_offsets;

    fn synthetic_corner(size: usize, theta0: f32, theta1: f32, width: f32) -> Vec<u8> {
        let cx = (size / 2) as f32;
        let cy = cx;
        let (s0, c0) = theta0.sin_cos();
        let (s1, c1) = theta1.sin_cos();
        let mut img = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d0 = -s0 * dx + c0 * dy;
                let d1 = -s1 * dx + c1 * dy;
                let q = (d0 / width).tanh() * (d1 / width).tanh();
                let val = 128.0 + 80.0 * q;
                img[y * size + x] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    fn pair_err(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
        pair_disagreement(a0, a1, b0, b1)
    }

    #[test]
    fn recovers_projective_disk_axes() {
        let size = 41usize;
        let cx = 20.0;
        let cy = 20.0;
        let target0 = 20.0_f32.to_radians();
        let target1 = 50.0_f32.to_radians();
        let img = synthetic_corner(size, target0, target1, 0.7);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, cx, cy, ring);

        let fit = fit(&img, size, size, cx, cy, 5, &samples, &phi);
        assert!(
            pair_err(fit.theta1, fit.theta2, target0, target1) < 3.0_f32.to_radians(),
            "fit={fit:?}"
        );
        assert!(fit.sigma_theta1.is_finite());
        assert!(fit.sigma_theta2.is_finite());
    }

    #[test]
    fn falls_back_near_border() {
        let size = 41usize;
        let img = synthetic_corner(size, 0.2, 1.2, 1.0);
        let ring = ring_offsets(5);
        let phi = ring_angles(ring);
        let samples = sample_ring(&img, size, size, 4.0, 4.0, ring);
        let fallback = sigma_correction::fit_lut(&samples, &phi);
        let fit = fit(&img, size, size, 4.0, 4.0, 5, &samples, &phi);

        assert!((fit.theta1 - fallback.theta1).abs() < 1e-6);
        assert!((fit.theta2 - fallback.theta2).abs() < 1e-6);
        assert!((fit.rms - fallback.rms).abs() < 1e-6);
    }
}
