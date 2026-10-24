//! Peak detection over a Radon response map.
//!
//! Applies threshold / NMS / min-cluster rejection on the
//! working-resolution response from
//! [`super::response::radon_response_u8`], keeps the peaks where the
//! image is a saddle, then a 3-point peak fit on the blurred response
//! to get subpixel offsets. Output coordinates are divided by
//! `image_upsample` so they sit in the **input pixel frame**.

use super::primitives::{fit_peak_curvature_grad, fit_peak_frac, fit_peak_frac_grad};
use super::response::{RadonDetectorParams, RadonResponseView};
use crate::detect::{count_positive_neighbors, is_local_max, Corner};
use crate::imageview::ImageView;
use crate::refine::PositionCovariance;
use crate::uncertainty::{
    edge_width, knot, CovarianceModel, PixelGradients, PixelTile, SystematicModel,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Stage 1 of Radon detection: threshold + NMS + cluster-filter +
/// saddle test + 3-point Gaussian peak-fit on the response map.
///
/// The saddle test drops peaks where the image around the peak is not
/// a chessboard X-junction. The `(max − min)²` ray response also peaks,
/// at about a third of a corner's response, beside straight edges that
/// run along a ray direction (0°, 45°, 90°, 135°) — on boards rotated
/// near 45° in particular. Around an X-junction the image is a saddle:
/// its curvature has opposite signs along the two diagonals between the
/// grid axes. Beside an edge it bends across the edge only. The test
/// fits a quadratic to the working image over the ray support and keeps
/// the peak when the smaller principal curvature is at least
/// `MIN_SADDLE_RATIO` of the larger one and of the opposite sign.
///
/// The 3-point Gaussian peak-fit is a **response-map** subpixel
/// operation (not image-domain refinement), so it stays inside this
/// stage. The output is already subpixel in the input-image frame
/// (positions are divided by `image_upsample`). Image-domain
/// refinement against the input pixels is the caller's choice and
/// lives in [`crate::refine_corners_on_image`].
pub fn detect_peaks_from_radon(
    resp: &RadonResponseView<'_>,
    params: &RadonDetectorParams,
) -> Vec<Corner> {
    let w = resp.width();
    let h = resp.height();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let data = resp.data();

    let mut max_r = f32::NEG_INFINITY;
    for &v in data {
        if v > max_r {
            max_r = v;
        }
    }
    if !max_r.is_finite() {
        return Vec::new();
    }

    let thr = relative_floor(params.threshold_rel, max_r);

    let nms_r = params.nms_radius as i32;
    let ray_r = params.ray_radius_clamped() as i32;
    // We already zeroed pixels < ray_r from the border; leave 1 extra
    // pixel for the 3-point peak fit.
    let border = (ray_r + nms_r + 1).max(0) as usize;
    if w <= 2 * border || h <= 2 * border {
        return Vec::new();
    }

    let inv_up = 1.0 / (params.image_upsample_clamped() as f32);
    let min_cluster = params.min_cluster_size;
    let peak_fit = params.peak_fit;
    // The saddle fit covers the ray support, which `border` keeps inside
    // the map.
    let saddle = SaddleTest::new(ray_r as usize);

    #[inline(always)]
    fn at(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
        data[y * w + x]
    }

    let process_row = |y: usize, sink: &mut Vec<Corner>| {
        for x in border..(w - border) {
            let v = at(data, w, x, y);
            if v <= thr {
                continue;
            }
            if !is_local_max(data, w, h, x, y, nms_r, v) {
                continue;
            }
            if count_positive_neighbors(data, w, h, x, y, nms_r) < min_cluster {
                continue;
            }
            if !saddle.accepts(resp.image, w, x, y) {
                continue;
            }

            let r_c = v;
            let r_xm = at(data, w, x - 1, y);
            let r_xp = at(data, w, x + 1, y);
            let r_ym = at(data, w, x, y - 1);
            let r_yp = at(data, w, x, y + 1);
            let fx = fit_peak_frac(r_xm, r_c, r_xp, peak_fit);
            let fy = fit_peak_frac(r_ym, r_c, r_yp, peak_fit);

            let gx = (x as f32 + fx) * inv_up;
            let gy = (y as f32 + fy) * inv_up;
            sink.push(Corner::new(gx, gy, v));
        }
    };

    #[cfg(feature = "rayon")]
    {
        let row_results: Vec<Vec<Corner>> = (border..(h - border))
            .into_par_iter()
            .map(|y| {
                let mut sink = Vec::new();
                process_row(y, &mut sink);
                sink
            })
            .collect();
        let total: usize = row_results.iter().map(|r| r.len()).sum();
        let mut out = Vec::with_capacity(total);
        for row in row_results {
            out.extend(row);
        }
        out
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut out = Vec::new();
        for y in border..(h - border) {
            process_row(y, &mut out);
        }
        out
    }
}

/// Acceptance floor of a Radon peak: `threshold_rel` of the frame's
/// strongest response `max_r`, never negative. A peak is kept when its
/// response exceeds the floor.
#[inline]
fn relative_floor(threshold_rel: f32, max_r: f32) -> f32 {
    (threshold_rel * max_r).max(0.0)
}

/// Keep the corners whose strength exceeds `threshold_rel` of the
/// strongest one: the frame threshold of [`detect_peaks_from_radon`],
/// applied to peaks gathered from several response patches of one frame.
pub(crate) fn retain_above_relative_threshold(corners: &mut Vec<Corner>, threshold_rel: f32) {
    let max_r = corners
        .iter()
        .map(|c| c.strength)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_r.is_finite() {
        return;
    }
    let thr = relative_floor(threshold_rel, max_r);
    corners.retain(|c| c.strength > thr);
}

/// Smallest ratio of the two principal curvatures (smaller / larger
/// magnitude, opposite signs) for which [`SaddleTest`] accepts a peak.
///
/// A saddle `(n₁·p)(n₂·p)` whose zero lines are `θ` apart has ratio
/// `(1 − |cos θ|) / (1 + |cos θ|)`: 1 for a square board, 0.33 at 60°.
const MIN_SADDLE_RATIO: f32 = 0.2;

/// Smallest difference of the two principal curvatures, times the
/// squared window radius, for which [`SaddleTest`] takes the curvature
/// as resolved: one gray level. Below it the ratio is quantization noise
/// (a straight edge through the window centre fits no curvature at all).
const MIN_SADDLE_SPREAD: f32 = 1.0;

/// Saddle test on the working image: fits `c + a·x² + b·xy + c'·y²`
/// (plus linear terms, which drop out) by least squares over the disk of
/// radius `k` around a pixel and accepts it when the fitted Hessian
/// `[2a b; b 2c']` has eigenvalues of opposite signs whose magnitudes
/// are within [`MIN_SADDLE_RATIO`] of each other.
///
/// A disk, not a square, keeps the fit rotation invariant: on a square
/// window a thin line along a diagonal projects onto the `xy` term alone
/// and would pass as a saddle.
struct SaddleTest {
    k: usize,
    /// Mean `r²` over the disk.
    mean_r2: f32,
    /// `1 / Σ (r² − mean_r2)²`, `1 / Σ (x² − y²)²`, `1 / Σ x²y²`.
    inv_norms: [f32; 3],
}

impl SaddleTest {
    fn new(k: usize) -> Self {
        let disk = || Self::offsets(k).map(|(x, y)| (x as f32, y as f32));
        let n = disk().count() as f32;
        let mean_r2 = disk().map(|(x, y)| x * x + y * y).sum::<f32>() / n;
        let mut norms = [0.0f32; 3];
        for (x, y) in disk() {
            let u = x * x + y * y - mean_r2;
            let v = x * x - y * y;
            norms[0] += u * u;
            norms[1] += v * v;
            norms[2] += x * x * y * y;
        }
        Self {
            k,
            mean_r2,
            inv_norms: norms.map(|v| 1.0 / v),
        }
    }

    /// Offsets `(dx, dy)` of the disk `dx² + dy² ≤ k²`, widened to the
    /// 3×3 block at `k = 1`, whose disk has no diagonal pixel to determine
    /// the `xy` term.
    fn offsets(k: usize) -> impl Iterator<Item = (isize, isize)> {
        let ki = k as isize;
        let r2 = (ki * ki).max(2);
        (-ki..=ki).flat_map(move |dy| {
            (-ki..=ki)
                .filter(move |dx| dx * dx + dy * dy <= r2)
                .map(move |dx| (dx, dy))
        })
    }

    /// Whether `image` (row stride `w`) is a saddle around `(x, y)`; the
    /// disk must lie inside the image.
    fn accepts(&self, image: &[u8], w: usize, x: usize, y: usize) -> bool {
        let k = self.k;
        // On the 4-fold symmetric disk, `1`, `r² − mean_r2`, `x² − y²`
        // and `xy` are mutually orthogonal (and orthogonal to `x`, `y`),
        // so each coefficient is one projection.
        let (mut su, mut sv, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
        for (ox, oy) in Self::offsets(k) {
            let px = x.wrapping_add_signed(ox);
            let py = y.wrapping_add_signed(oy);
            let i = f32::from(image[py * w + px]);
            let (dx, dy) = (ox as f32, oy as f32);
            su += (dx * dx + dy * dy - self.mean_r2) * i;
            sv += (dx * dx - dy * dy) * i;
            sxy += dx * dy * i;
        }
        // I ≈ p·(x² + y²) + q·(x² − y²) + b·xy: Hxx = 2(p + q),
        // Hyy = 2(p − q), Hxy = b.
        let p = su * self.inv_norms[0];
        let q = sv * self.inv_norms[1];
        let b = sxy * self.inv_norms[2];
        // Eigenvalues (tr ± spread)/2 with tr = 4p and
        // spread = √((Hxx − Hyy)² + 4Hxy²). Opposite signs with
        // min(|λ|)/max(|λ|) ≥ ρ  ⇔  |tr| ≤ spread·(1 − ρ)/(1 + ρ).
        let tr = 4.0 * p;
        let spread = (16.0 * q * q + 4.0 * b * b).sqrt();
        let kk = (k * k) as f32;
        spread * kk >= MIN_SADDLE_SPREAD
            && tr.abs() * (1.0 + MIN_SADDLE_RATIO) <= spread * (1.0 - MIN_SADDLE_RATIO)
    }
}

/// Systematic error of the response-map peak fit by edge width (see
/// [`SystematicModel`]); the fit window is one working-resolution pixel.
/// The pull toward the peak pixel grows with blur (gain 0.9–1 on sharp
/// edges, 0.64–0.76 at edge widths of 1.4–1.7 px), most on boards rotated
/// near 45°. Fitted on camera-formed synthetic boards from two generators
/// (cells 6–12, rotations 0–45°, shear, lens blur σ 0.3–3) by
/// `tools/refiner_robustness/fit_systematic.py`.
const SYSTEMATIC: SystematicModel = SystematicModel {
    knots: &[
        knot(1.09, 0.918, 0.0359), // rms 0.0379, n 6289
        knot(1.14, 0.949, 0.0300), // rms 0.0309, n 8171
        knot(1.18, 0.940, 0.0295), // rms 0.0308, n 8666
        knot(1.24, 0.927, 0.0314), // rms 0.0331, n 8029
        knot(1.30, 0.863, 0.0374), // rms 0.0428, n 10301
        knot(1.35, 0.932, 0.0358), // rms 0.0372, n 8203
        knot(1.40, 0.764, 0.0571), // rms 0.0666, n 6104
        knot(1.47, 0.716, 0.0653), // rms 0.0772, n 6308
        knot(1.58, 0.716, 0.0678), // rms 0.0793, n 6400
        knot(1.68, 0.641, 0.0803), // rms 0.0966, n 8313
    ],
    // Real stereo calibration set: `tools/refiner_robustness/real_validation/fit_floor.py`.
    floor: 0.045,
};

/// Ray directions in the order the response kernel scans them
/// (horizontal, vertical, the two diagonals); ties go to the first.
const RAYS: [[i32; 2]; 4] = [[1, 0], [0, 1], [1, 1], [1, -1]];

/// Scratch for [`peak_fit_covariance`], held by
/// [`RadonBuffers`](super::RadonBuffers) and reused across peaks and frames.
#[derive(Debug, Default)]
pub(crate) struct PeakFitScratch {
    /// Working-resolution samples around the peak, row-major.
    work: Vec<f32>,
    /// Gradient per working sample of the quantity being propagated.
    work_grad: Vec<[f32; 2]>,
    /// `(max − min, argmax, argmin)` of the ray sums per response sample.
    rays: Vec<(f32, u8, u8)>,
    /// Unblurred response per response sample.
    response: Vec<f32>,
    grads: PixelGradients,
    tile: PixelTile,
}

/// Geometry of one peak's support.
#[derive(Clone, Copy)]
struct PeakSupport {
    peak: [i32; 2],
    up: i32,
    ray_r: i32,
    blur_r: i32,
    resp_k: i32,
    work_k: i32,
}

impl PeakSupport {
    fn rs(&self) -> usize {
        (2 * self.resp_k + 1) as usize
    }
    fn ws(&self) -> usize {
        (2 * self.work_k + 1) as usize
    }
}

/// Propagate the gradient of a quantity with respect to the five blurred
/// fit samples (`samples`: offset from the peak pixel and the 2-vector of
/// derivatives) back to the input pixels; returns `Σₚ gₚgₚᵀ` over the input
/// pixels and, separately, over the working samples (for their 8-bit
/// rounding when the input is upsampled).
fn backprop(
    sup: PeakSupport,
    samples: &[([i32; 2], [f32; 2]); 5],
    rays: &[(f32, u8, u8)],
    work_grad: &mut Vec<[f32; 2]>,
    grads: &mut PixelGradients,
) -> Option<(PositionCovariance, PositionCovariance)> {
    let (rs, ws) = (sup.rs(), sup.ws());
    let (blur_r, ray_r, work_k, resp_k) = (sup.blur_r, sup.ray_r, sup.work_k, sup.resp_k);
    let blur_n = ((2 * blur_r + 1) * (2 * blur_r + 1)) as f32;
    work_grad.clear();
    work_grad.resize(ws * ws, [0.0; 2]);
    for (c, d) in samples {
        let d = d.map(|v| v / blur_n);
        for by in -blur_r..=blur_r {
            for bx in -blur_r..=blur_r {
                let (qx, qy) = (c[0] + bx, c[1] + by);
                let (diff, imax, imin) = rays[(qy + resp_k) as usize * rs + (qx + resp_k) as usize];
                // ∂R/∂U = 2(S_max − S_min) on the max ray, minus on the min ray.
                let f = [2.0 * diff * d[0], 2.0 * diff * d[1]];
                if f == [0.0, 0.0] {
                    continue;
                }
                for (ray, sign) in [(imax, 1.0f32), (imin, -1.0)] {
                    let dir = RAYS[ray as usize];
                    for k in -ray_r..=ray_r {
                        let (sx, sy) = (qx + k * dir[0], qy + k * dir[1]);
                        let g =
                            &mut work_grad[(sy + work_k) as usize * ws + (sx + work_k) as usize];
                        g[0] += sign * f[0];
                        g[1] += sign * f[1];
                    }
                }
            }
        }
    }
    // Back through the upsampling onto input pixels.
    let (peak, up) = (sup.peak, sup.up);
    let lo = [peak[0] - work_k, peak[1] - work_k].map(|v| v.div_euclid(up));
    let side = ((2 * work_k) / up + 2) as usize;
    grads.reset(lo, side);
    let (mut qxx, mut qxy, mut qyy) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..ws as i32 {
        for i in 0..ws as i32 {
            let g = work_grad[j as usize * ws + i as usize];
            if g == [0.0, 0.0] {
                continue;
            }
            let (gx, gy) = (f64::from(g[0]), f64::from(g[1]));
            qxx += gx * gx;
            qxy += gx * gy;
            qyy += gy * gy;
            let (wx, wy) = (peak[0] - work_k + i, peak[1] - work_k + j);
            if up == 1 {
                grads.add(wx, wy, g);
                continue;
            }
            let (x0, y0) = (wx.div_euclid(2), wy.div_euclid(2));
            let tx = if wx.rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            let ty = if wy.rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            for (px, py, wgt) in [
                (x0, y0, (1.0 - tx) * (1.0 - ty)),
                (x0 + 1, y0, tx * (1.0 - ty)),
                (x0, y0 + 1, (1.0 - tx) * ty),
                (x0 + 1, y0 + 1, tx * ty),
            ] {
                if wgt != 0.0 {
                    grads.add(px, py, [wgt * g[0], wgt * g[1]]);
                }
            }
        }
    }
    let working = PositionCovariance::new(qxx as f32, qxy as f32, qyy as f32);
    Some((grads.covariance()?, working))
}

/// Covariance model of a Radon peak at `xy` (input pixels, as
/// [`detect_peaks_from_radon`] returns it for `image`), or `None` when the
/// 3-point fit does not respond to the response (a flat or clamped fit on
/// either axis).
///
/// The peak is `(X + f)/up` with the 3-point fit `f` of the blurred response
/// `R̄` at the working-resolution pixel `X`. Every stage is differentiated:
/// the fit with respect to its samples, the box blur, the response
/// `R = (S_max − S_min)²` with the arg-max and arg-min rays held fixed, the
/// ray sums, and the bilinear 2× upsampling of the input, so the gradient is
/// accumulated over the input pixels. The upsampled samples are rounded to
/// 8 bits; that rounding enters as independent noise of variance 1/12 per
/// working sample, in the systematic term. The fit divides by the curvature
/// `D` of `R̄` (or `ln R̄`) across the peak; where the response is flat, the
/// noise in `D` is not small next to `D`, and the noise term grows by the
/// second-order factor `1 + 3·var(D)/D²` per axis.
pub(crate) fn peak_fit_covariance(
    image: &ImageView<'_>,
    xy: [f32; 2],
    params: &RadonDetectorParams,
    scratch: &mut PeakFitScratch,
) -> Option<CovarianceModel> {
    let up = params.image_upsample_clamped() as i32;
    let upf = up as f32;
    // Working-resolution peak pixel; the fit offset is in (−½, ½).
    let peak = xy.map(|v| (v * upf).round());
    let frac = [xy[0] * upf - peak[0], xy[1] * upf - peak[1]];
    if !(frac[0].abs() < 0.5 && frac[1].abs() < 0.5) {
        return None;
    }
    let blur_r = params.response_blur_radius as i32;
    let ray_r = params.ray_radius_clamped() as i32;
    let sup = PeakSupport {
        peak: peak.map(|v| v as i32),
        up,
        ray_r,
        blur_r,
        resp_k: 1 + blur_r,
        work_k: 1 + blur_r + ray_r,
    };
    let (rs, ws, resp_k, work_k) = (sup.rs(), sup.ws(), sup.resp_k, sup.work_k);
    let PeakFitScratch {
        work,
        work_grad,
        rays,
        response,
        grads,
        tile,
    } = scratch;
    work.clear();
    for j in -work_k..=work_k {
        for i in -work_k..=work_k {
            let (wx, wy) = (sup.peak[0] + i, sup.peak[1] + j);
            let v = if up == 1 {
                image.sample(wx, wy)
            } else {
                image
                    .sample_bilinear(wx as f32 / upf, wy as f32 / upf)
                    .round()
                    .clamp(0.0, 255.0)
            };
            work.push(v);
        }
    }
    let w_at =
        |work: &[f32], dx: i32, dy: i32| work[(dy + work_k) as usize * ws + (dx + work_k) as usize];

    rays.clear();
    response.clear();
    for qy in -resp_k..=resp_k {
        for qx in -resp_k..=resp_k {
            let mut sums = [0.0f32; 4];
            for (s, d) in sums.iter_mut().zip(RAYS) {
                for k in -ray_r..=ray_r {
                    *s += w_at(work, qx + k * d[0], qy + k * d[1]);
                }
            }
            let (mut imax, mut imin) = (0usize, 0usize);
            for a in 1..4 {
                if sums[a] > sums[imax] {
                    imax = a;
                }
                if sums[a] < sums[imin] {
                    imin = a;
                }
            }
            let d = sums[imax] - sums[imin];
            rays.push((d, imax as u8, imin as u8));
            response.push(d * d);
        }
    }
    let r_at = |dx: i32, dy: i32| response[(dy + resp_k) as usize * rs + (dx + resp_k) as usize];
    let blur_n = ((2 * blur_r + 1) * (2 * blur_r + 1)) as f32;
    let blurred = |cx: i32, cy: i32| {
        let mut acc = 0.0f32;
        for dy in -blur_r..=blur_r {
            for dx in -blur_r..=blur_r {
                acc += r_at(cx + dx, cy + dy);
            }
        }
        acc / blur_n
    };
    let (bc, bxm, bxp, bym, byp) = (
        blurred(0, 0),
        blurred(-1, 0),
        blurred(1, 0),
        blurred(0, -1),
        blurred(0, 1),
    );
    let mode = params.peak_fit;
    let gx = fit_peak_frac_grad(bxm, bc, bxp, mode)?;
    let gy = fit_peak_frac_grad(bym, bc, byp, mode)?;
    let samples = |x: [f32; 3], y: [f32; 3], s: f32| {
        [
            ([-1, 0], [s * x[0], 0.0]),
            ([1, 0], [s * x[2], 0.0]),
            ([0, -1], [0.0, s * y[0]]),
            ([0, 1], [0.0, s * y[2]]),
            ([0, 0], [s * x[1], s * y[1]]),
        ]
    };
    // Position: ∂x̂ = ∂f / up.
    let (noise, working) = backprop(sup, &samples(gx, gy, 1.0 / upf), rays, work_grad, grads)?;
    // Curvature of the fit on each axis, for the second-order growth.
    let (dx, cx) = fit_peak_curvature_grad(bxm, bc, bxp, mode);
    let (dy, cy) = fit_peak_curvature_grad(bym, bc, byp, mode);
    let (curv, _) = backprop(sup, &samples(cx, cy, 1.0), rays, work_grad, grads)?;
    let growth = [(curv.xx, dx), (curv.yy, dy)].map(|(v, d)| (3.0 * v / (d * d)).sqrt());

    let offset = frac.map(|f| f / upf);
    let centre = xy.map(|v| v.round() as i32);
    let mut systematic = SYSTEMATIC.covariance(offset, edge_width(image, centre, tile), 1.0 / upf);
    if up > 1 {
        let q = 1.0 / 12.0;
        systematic = PositionCovariance::new(
            systematic.xx + q * working.xx,
            systematic.xy + q * working.xy,
            systematic.yy + q * working.yy,
        );
    }
    Some(CovarianceModel {
        noise,
        systematic,
        growth,
    })
}

/// Attach the Radon peak-fit covariance to `corner`, a peak of
/// [`detect_peaks_from_radon`] on `image` that keeps its peak-fit position,
/// using `buffers`' scratch. Exposed for the facade and the benches.
#[doc(hidden)]
pub fn attach_radon_peak_covariance(
    corner: &mut Corner,
    image: ImageView<'_>,
    params: &RadonDetectorParams,
    buffers: &mut super::RadonBuffers,
) {
    let model = peak_fit_covariance(
        &image,
        [corner.x, corner.y],
        params,
        &mut buffers.peak_scratch,
    );
    corner.set_model(model);
}

#[cfg(test)]
mod tests {
    use super::super::primitives::PeakFitMode;
    use super::super::response::{radon_response_u8, RadonBuffers, RadonDetectorParams};
    use super::super::test_fixtures::synthetic_chessboard_aa;
    use super::*;
    use crate::refine::test_fixtures::tanh_saddle;

    /// The Radon peak position for working-resolution peak pixel `peak` of a
    /// float image: bilinear upsampling (unrounded), ray sums, `(max −
    /// min)²`, box blur, 3-point fit. The reference for the analytic
    /// gradient.
    fn peak_f64(
        img: &[f64],
        side: usize,
        params: &RadonDetectorParams,
        peak: [i32; 2],
    ) -> [f64; 2] {
        let up = params.image_upsample_clamped() as i32;
        let ray_r = params.ray_radius_clamped() as i32;
        let br = params.response_blur_radius as i32;
        let px = |x: i32, y: i32| img[y as usize * side + x as usize];
        let work = |wx: i32, wy: i32| {
            if up == 1 {
                return px(wx, wy);
            }
            let (x0, y0) = (wx.div_euclid(2), wy.div_euclid(2));
            let (tx, ty) = (
                0.5 * f64::from(wx.rem_euclid(2)),
                0.5 * f64::from(wy.rem_euclid(2)),
            );
            let a = px(x0, y0) * (1.0 - tx) + px(x0 + 1, y0) * tx;
            let b = px(x0, y0 + 1) * (1.0 - tx) + px(x0 + 1, y0 + 1) * tx;
            a * (1.0 - ty) + b * ty
        };
        let resp = |qx: i32, qy: i32| {
            let sums = RAYS.map(|d| {
                (-ray_r..=ray_r)
                    .map(|k| work(qx + k * d[0], qy + k * d[1]))
                    .sum::<f64>()
            });
            let mx = sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mn = sums.iter().copied().fold(f64::INFINITY, f64::min);
            (mx - mn) * (mx - mn)
        };
        let blurred = |cx: i32, cy: i32| {
            let mut acc = 0.0;
            for dy in -br..=br {
                for dx in -br..=br {
                    acc += resp(cx + dx, cy + dy);
                }
            }
            acc / f64::from((2 * br + 1) * (2 * br + 1))
        };
        let fit = |m: f64, c: f64, p: f64| {
            let (m, c, p) = match params.peak_fit {
                PeakFitMode::Gaussian => (m.ln(), c.ln(), p.ln()),
                _ => (m, c, p),
            };
            0.5 * (m - p) / (m - 2.0 * c + p)
        };
        let (x, y) = (peak[0], peak[1]);
        let c = blurred(x, y);
        let fx = fit(blurred(x - 1, y), c, blurred(x + 1, y));
        let fy = fit(blurred(x, y - 1), c, blurred(x, y + 1));
        [
            (f64::from(x) + fx) / f64::from(up),
            (f64::from(y) + fy) / f64::from(up),
        ]
    }

    /// `peak_fit_covariance` is `Σₚ (∂x̂/∂Iₚ)(∂x̂/∂Iₚ)ᵀ` over the input
    /// pixels: central differences (±0.01 gray level) of a float
    /// re-implementation of the peak position (no rounding of the
    /// upsampled samples, which the analytic gradient treats as the
    /// identity), for both peak-fit modes and both upsampling factors.
    #[test]
    fn peak_covariance_matches_finite_differences() {
        const SIDE: usize = 29;
        for (up, mode, angles, offset) in [
            (
                2u32,
                PeakFitMode::Gaussian,
                [0.3f32, 1.4f32],
                [0.3f32, -0.2f32],
            ),
            (2, PeakFitMode::Parabolic, [-0.2, 1.05], [-0.35, 0.15]),
            (1, PeakFitMode::Gaussian, [0.1, 1.6], [0.2, 0.4]),
        ] {
            let params = RadonDetectorParams {
                image_upsample: up,
                peak_fit: mode,
                ..RadonDetectorParams::default()
            };
            let img: Vec<u8> = tanh_saddle(SIDE, [14, 14], offset, angles, 1.2, 6.0)
                .iter()
                .map(|v| v.round() as u8)
                .collect();
            let mut buffers = RadonBuffers::new();
            let resp = radon_response_u8(&img, SIDE, SIDE, &params, &mut buffers);
            let corner = detect_peaks_from_radon(&resp, &params)
                .into_iter()
                .min_by(|a, b| {
                    let d = |c: &Corner| (c.x - 14.0 - offset[0]).hypot(c.y - 14.0 - offset[1]);
                    d(a).total_cmp(&d(b))
                })
                .expect("a peak near the saddle");
            let view = ImageView::from_u8_slice(SIDE, SIDE, &img).unwrap();
            let cov = peak_fit_covariance(
                &view,
                [corner.x, corner.y],
                &params,
                &mut PeakFitScratch::default(),
            )
            .expect("the fit responds");
            let cov = cov.noise;

            let peak = [corner.x, corner.y].map(|v| (v * up as f32).round() as i32);
            let mut f: Vec<f64> = img.iter().map(|&v| f64::from(v)).collect();
            let base = peak_f64(&f, SIDE, &params, peak);
            // The detector rounds the upsampled samples; the reference does not.
            assert!(
                (base[0] - f64::from(corner.x)).abs() < 1e-2
                    && (base[1] - f64::from(corner.y)).abs() < 1e-2,
                "reference {base:?} vs detector ({}, {})",
                corner.x,
                corner.y
            );
            let eps = 0.01f64;
            let mut fd = [0.0f64; 3];
            for k in 0..f.len() {
                let v = f[k];
                f[k] = v + eps;
                let plus = peak_f64(&f, SIDE, &params, peak);
                f[k] = v - eps;
                let minus = peak_f64(&f, SIDE, &params, peak);
                f[k] = v;
                let g = [0, 1].map(|a| (plus[a] - minus[a]) / (2.0 * eps));
                fd[0] += g[0] * g[0];
                fd[1] += g[0] * g[1];
                fd[2] += g[1] * g[1];
            }
            let scale = (fd[0] * fd[2]).sqrt();
            // Exact for up = 1; with 2× upsampling the analytic gradient is
            // taken at the rounded samples, the reference at unrounded ones.
            // Measured: up to 2e-5 at up = 1, 2.3e-2 at up = 2.
            let tol = if up == 1 { 1e-4 } else { 5e-2 };
            assert!(
                fd[1].abs() > 0.01 * scale,
                "fixture must have xy ≠ 0: {fd:?}"
            );
            for (name, g, f) in [
                ("xx", cov.xx, fd[0]),
                ("xy", cov.xy, fd[1]),
                ("yy", cov.yy, fd[2]),
            ] {
                assert!(
                    (f64::from(g) - f).abs() <= tol * scale,
                    "up {up} {mode:?} {name}: analytic {g:e} vs finite-difference {f:e}"
                );
            }
        }
    }

    #[test]
    fn detector_recovers_interior_corner_without_seed() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);

        let params = RadonDetectorParams {
            image_upsample: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_peaks_from_radon(&resp, &params);

        assert!(
            !corners.is_empty(),
            "expected at least one corner, got none"
        );

        let expected_x =
            offset.0 + (((SIZE as f32 / 2.0 - offset.0) / CELL as f32).round() * CELL as f32);
        let expected_y =
            offset.1 + (((SIZE as f32 / 2.0 - offset.1) / CELL as f32).round() * CELL as f32);

        let (best, err) = corners
            .iter()
            .map(|c| {
                let dx = c.x - expected_x;
                let dy = c.y - expected_y;
                (c, (dx * dx + dy * dy).sqrt())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty");
        assert!(
            err < 0.2,
            "closest corner err={:.4} at ({:.3}, {:.3}), expected ({:.3}, {:.3}); {} corners found",
            err,
            best.x,
            best.y,
            expected_x,
            expected_y,
            corners.len(),
        );
    }

    /// `size × size` image of the half-plane `n·(p − c) > 0` (bright) with
    /// unit normal `n` at angle `deg`, each pixel the mean of 8×8
    /// sub-samples.
    fn half_plane(size: usize, deg: f32, c: (f32, f32)) -> Vec<u8> {
        let (s, co) = deg.to_radians().sin_cos();
        let mut img = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut acc = 0.0f32;
                for sy in 0..8 {
                    for sx in 0..8 {
                        let px = x as f32 + (sx as f32 + 0.5) / 8.0 - 0.5 - c.0;
                        let py = y as f32 + (sy as f32 + 0.5) / 8.0 - 0.5 - c.1;
                        acc += if co * px + s * py > 0.0 { 200.0 } else { 40.0 };
                    }
                }
                img[y * size + x] = (acc / 64.0).round() as u8;
            }
        }
        img
    }

    /// Whether the saddle test accepts the centre of `img` (`size × size`,
    /// odd) on the working image the detector would see.
    fn centre_is_saddle(img: &[u8], size: usize) -> bool {
        let params = RadonDetectorParams::default();
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(img, size, size, &params, &mut buffers);
        let saddle = SaddleTest::new(params.ray_radius_clamped() as usize);
        let c = size - 1; // working-resolution centre at upsample 2
        saddle.accepts(resp.image, resp.width(), c, c)
    }

    /// The saddle test accepts an X-junction at any rotation and at a 60°
    /// shear, and rejects a straight edge beside the centre and a thin
    /// line through it, along and between the ray directions.
    #[test]
    fn saddle_test_separates_junctions_from_edges() {
        use chess_corners_testutil::{optical_chessboard, Lattice};
        const SIZE: usize = 21;
        let c = (SIZE / 2) as f32;
        for deg in [0.0f32, 10.0, 22.5, 30.0, 45.0] {
            for axes in [90.0f32, 60.0] {
                let lattice = Lattice::new(12.0, (c, c))
                    .rotated(deg.to_radians())
                    .with_axis_angle(axes.to_radians());
                let img = optical_chessboard(SIZE, &lattice, 0.7, 40, 200);
                assert!(
                    centre_is_saddle(&img, SIZE),
                    "X-junction at {deg}°, axes {axes}°"
                );
            }
        }
        for deg in [0.0f32, 22.5, 45.0, 67.5, 90.0, 135.0] {
            let (s, co) = deg.to_radians().sin_cos();
            for offset in [0.25f32, 0.5, 1.0, 1.5, 2.0] {
                let img = half_plane(SIZE, deg, (c - offset * co, c - offset * s));
                assert!(
                    !centre_is_saddle(&img, SIZE),
                    "edge with normal {deg}° at {offset} px"
                );
            }
        }
        // A thin dark line through the centre along each ray direction.
        for (dx, dy) in [(1i32, 0i32), (0, 1), (1, 1), (1, -1)] {
            let mut img = vec![180u8; SIZE * SIZE];
            for t in -(SIZE as i32)..SIZE as i32 {
                let (x, y) = (c as i32 + t * dx, c as i32 + t * dy);
                if (0..SIZE as i32).contains(&x) && (0..SIZE as i32).contains(&y) {
                    img[y as usize * SIZE + x as usize] = 60;
                }
            }
            assert!(!centre_is_saddle(&img, SIZE), "line along ({dx}, {dy})");
        }
    }

    /// Every ray radius gives a determined fit, down to `ray_radius = 1`.
    #[test]
    fn saddle_fit_is_determined_at_every_radius() {
        for k in 1..=8 {
            let t = SaddleTest::new(k);
            assert!(
                t.inv_norms.iter().all(|v| v.is_finite() && *v > 0.0),
                "k = {k}"
            );
        }
    }

    /// A sharp board rotated 45°: the edges between the corners run along
    /// the diagonal rays, where the ray response forms ridges at about a
    /// third of the corner response. Every detected peak is a corner, and
    /// every interior corner is detected.
    #[test]
    fn no_edge_peaks_on_sharp_45_degree_board() {
        use chess_corners_testutil::{optical_chessboard, Lattice};
        const SIZE: usize = 120;
        let lattice = Lattice::new(12.0, (60.3, 60.6)).rotated(45f32.to_radians());
        let img = optical_chessboard(SIZE, &lattice, 0.5, 40, 200);
        let params = RadonDetectorParams::default();
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_peaks_from_radon(&resp, &params);
        assert!(!corners.is_empty());
        for c in &corners {
            let (x, y) = lattice.nearest_corner(c.x, c.y);
            let err = (x - c.x).hypot(y - c.y);
            assert!(
                err < 0.5,
                "peak ({}, {}) is {err} px from a corner",
                c.x,
                c.y
            );
        }
        // Interior corners: lattice points at least 8 px inside the frame.
        let mut expected = 0;
        for k in -10i32..=10 {
            for m in -10i32..=10 {
                let (s, co) = 45f32.to_radians().sin_cos();
                let x = 60.3 + 12.0 * (k as f32 * co - m as f32 * s);
                let y = 60.6 + 12.0 * (k as f32 * s + m as f32 * co);
                if (8.0..SIZE as f32 - 9.0).contains(&x) && (8.0..SIZE as f32 - 9.0).contains(&y) {
                    expected += 1;
                    assert!(
                        corners.iter().any(|c| (c.x - x).hypot(c.y - y) < 0.5),
                        "corner ({x}, {y}) missed"
                    );
                }
            }
        }
        assert!(expected > 40, "{expected}");
    }

    /// The frame threshold over gathered peaks keeps exactly the peaks
    /// above `threshold_rel` of the strongest one.
    #[test]
    fn frame_threshold_is_relative_to_the_strongest_peak() {
        let mut corners: Vec<Corner> = [100.0f32, 29.0, 31.0, 5.0, 60.0]
            .iter()
            .enumerate()
            .map(|(i, &s)| Corner::new(i as f32, 0.0, s))
            .collect();
        retain_above_relative_threshold(&mut corners, 0.3);
        let kept: Vec<f32> = corners.iter().map(|c| c.strength).collect();
        assert_eq!(kept, vec![100.0, 31.0, 60.0]);
        let mut none: Vec<Corner> = Vec::new();
        retain_above_relative_threshold(&mut none, 0.3);
        assert!(none.is_empty());
    }

    #[test]
    fn detector_upsample_1_still_produces_corners() {
        const SIZE: usize = 65;
        const CELL: usize = 8;
        let offset = (32.35, 32.8);
        let img = synthetic_chessboard_aa(SIZE, CELL, offset, 30, 230);
        let params = RadonDetectorParams {
            image_upsample: 1,
            ray_radius: 2,
            nms_radius: 2,
            ..RadonDetectorParams::default()
        };
        let mut buffers = RadonBuffers::new();
        let resp = radon_response_u8(&img, SIZE, SIZE, &params, &mut buffers);
        let corners = detect_peaks_from_radon(&resp, &params);
        assert!(!corners.is_empty(), "upsample=1 produced no corners");
    }
}
