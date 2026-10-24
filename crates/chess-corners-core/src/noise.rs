//! Per-image estimate of the pixel-noise level.
//!
//! A 4×4 window is expanded in the orthogonal polynomials on four points,
//! `s = (1, 1, 1, 1)`, `l = (−3, −1, 1, 3)`, `e = (1, −1, −1, 1)`,
//! `d = (1, −3, 3, −1)` (degrees 0–3), giving 16 components `(i, j)`
//! (degree `i` along x, `j` along y). The estimator measures the noise with
//! `d ⊗ d`, the product of third differences: it cancels every intensity
//! that is at most quadratic along either axis — planar shading, straight
//! edges parallel to an image axis, most of a smoothly blurred pattern —
//! and for white noise of standard deviation `σ` its output has standard
//! deviation `20σ`.
//!
//! What it does not cancel — sharp edges at any angle, corners — also
//! drives the other non-smooth components, so a window is used only when
//! those are small against the current noise level. The gate takes
//! every component of total degree 3 or more except the operator itself:
//! `(1,2)`, `(2,1)`, `(2,2)`, `(1,3)`, `(3,1)`, `(2,3)`, `(3,2)`, `(3,0)`,
//! `(0,3)`. A straight edge along the window diagonal projects only onto
//! `(i, i)` components, so `(2,2)` is what catches it; smooth shading lives
//! in the low-degree components and drives none of them.
//!
//! Each gate component is scaled to the operator's white-noise variance,
//! so a noise-only window's gate statistic is χ²₉, and the gate keeps
//! windows below a fixed multiple of the current noise variance (never
//! below the 8-bit rounding variance). Under Gaussian noise the components
//! are uncorrelated with the operator output, so the gate does not bias it.
//!
//! The noise level is a fixed point — the gate it sets yields it — reached
//! from the quietest sixteenth of the windows, so it starts below any
//! structure, and every estimate is a 4σ-truncated mean of the operator's
//! square. The threshold is applied exactly to an in-memory sample, so the
//! result does not jump with the gate setting.
//!
//! Sensor noise grows with brightness, so windows are split into four
//! intensity bands, each reaching its own fixed point from its own quietest
//! windows. A band holds structure rather than noise — and is dropped — if
//! its variance exceeds eight times the image-wide one or three times the
//! median over bands (noise changes smoothly with brightness; edge-only
//! intensities between flat dark and bright areas stand out). The result
//! is the admitted-count weighted mean of the remaining bands.
//!
//! Windows whose mean lies within [`RAIL_MARGIN`] gray levels of 0 or 255
//! are skipped: clipping there removes noise. The window mean is the
//! `(0, 0)` component, orthogonal to the operator, so this selection does
//! not bias the statistic (rejecting by pixel value would).
//!
//! The windows are a fixed sample of about [`WINDOWS`] on a staggered grid
//! (four times as many, once, when too few are free of edges); everything
//! after sampling works on the sample alone.

/// About this many 4×4 windows are sampled per image.
const WINDOWS: usize = 8192;
/// Windows whose mean is within this many gray levels of 0 or 255 are
/// skipped (clipped noise).
const RAIL_MARGIN: i32 = 12;
/// Gate components `(degree along x, degree along y)`.
const GATE_TERMS: [(usize, usize); 9] = [
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 0),
    (0, 3),
];
/// Keep windows whose gate statistic (χ²₉ for noise) is below this.
const GATE: f64 = 20.0;
/// Truncation of the operator's second moment, in its standard deviations.
const TRUNC: f64 = 4.0;
/// A band whose noise variance exceeds this many times the image-wide one
/// is taken for structure.
const MAX_BAND_RATIO: f64 = 8.0;
/// A band whose noise variance exceeds this many times the median over
/// bands is taken for structure.
const MAX_BAND_OUTLIER: f64 = 3.0;
/// A band needs this many admitted windows to contribute.
const MIN_BAND_SAMPLES: usize = 128;
/// Below this many admitted windows over all bands the estimate is not
/// reported (the relative standard error of σ̂ is about `1/sqrt(2n)`, 3%
/// here).
const MIN_SAMPLES: usize = 512;
/// A fixed point counts as settled when the variance moves by less than
/// this fraction.
const SETTLED: f64 = 1e-3;
/// Fixed-point rounds (the gate settles in a handful).
const ROUNDS: usize = 12;
/// Intensity bands, by window mean: `[0, 64)`, `[64, 128)`, ….
const BANDS: usize = 4;
/// Standard deviation of the operator output per unit pixel noise σ
/// (`sqrt(Σ wᵢ²) = sqrt(20 · 20)`).
const OPERATOR_GAIN: f64 = 20.0;

/// Estimate the standard deviation of the pixel noise in an 8-bit
/// grayscale image, in gray levels.
///
/// Measures the noise as it appears in the 8-bit image, so the estimate
/// includes the rounding to integers (`1/12` gray level² when the noise
/// spans several levels). It is designed for images dominated by strong
/// edges, such as calibration boards: edges, corners and smooth shading are
/// excluded from the statistic, and so are areas within 12 gray levels of
/// black or white, where clipping removes noise. Under signal-dependent
/// noise the estimate is the root of the mean noise variance over the
/// image's structure-free areas.
///
/// For white noise it is the noise standard deviation. For spatially
/// correlated noise (demosaiced, compressed or resampled images) it is the
/// noise as the estimator's operator, a product of third differences, sees
/// it — usually below the per-pixel σ, which that operator partly cancels:
/// a consistent per-image scale rather than the per-pixel σ. Adding white
/// noise of variance `σₐ²` raises its square by `σₐ²`.
///
/// Accuracy on synthetic boards with white noise (cells 5–12, any rotation
/// or shear, blur up to 2.5 px): within about 4% for σ ≥ 2 gray levels and
/// 5% for σ ≥ 1 (12% on images with very few structure-free areas). Below
/// 1 gray level, residual pattern in strongly blurred images is comparable
/// to the noise and reads up to about 20% high. Images
/// with almost no structure-free area — small, sharp or slightly blurred
/// cells filling the whole frame — give no estimate at low noise and can
/// read high (up to 2×) at very high noise.
///
/// Deterministic and cheap: it samples about 8 000 4×4 windows on a
/// regular grid, so the cost does not grow with the image size (under
/// 0.1 ms per frame); only an image where too few of them are free of edges
/// is sampled once more, four times as densely.
///
/// Returns `None` when the image is smaller than 4×4 or too few windows
/// (fewer than 512) are free of edges.
///
/// # Panics
///
/// Panics if `img.len() != w * h`.
pub fn estimate_noise_sigma_u8(img: &[u8], w: usize, h: usize) -> Option<f32> {
    assert_eq!(
        img.len(),
        w * h,
        "estimate_noise_sigma_u8: img.len() ({}) must equal w*h ({w} * {h} = {})",
        img.len(),
        w * h,
    );
    if w < 4 || h < 4 {
        return None;
    }
    // An image with too few structure-free windows in the sample is
    // sampled once more, four times as densely.
    let (sample, step) = Sample::collect(img, w, h, WINDOWS);
    noise_sigma(&sample).or_else(|| {
        (step > 1)
            .then(|| Sample::collect(img, w, h, 4 * WINDOWS).0)
            .and_then(|dense| noise_sigma(&dense))
    })
}

/// The noise σ of a sample (see [`estimate_noise_sigma_u8`]).
fn noise_sigma(sample: &Sample) -> Option<f32> {
    if sample.len() < MIN_SAMPLES {
        return None;
    }
    let image_var = sample.image_variance();
    // Per band: the fixed point of its own gate and its admitted count.
    let bands: Vec<(usize, f64)> = (0..BANDS)
        .filter_map(|b| sample.band(b).variance(image_var))
        .collect();
    // Noise variance changes smoothly with brightness; a band far above the
    // others holds structure (edge-only intensities between the flat ones).
    let mut vars: Vec<f64> = bands.iter().map(|&(_, v)| v).collect();
    vars.sort_by(f64::total_cmp);
    let median = match vars.len() {
        0 => return None,
        n if n % 2 == 1 => vars[n / 2],
        n => 0.5 * (vars[n / 2 - 1] + vars[n / 2]),
    };
    let (mut n, mut nvar) = (0usize, 0f64);
    for &(nb, var) in &bands {
        if var <= MAX_BAND_OUTLIER * median.max(ROUNDING_OP_VAR) {
            n += nb;
            nvar += nb as f64 * var;
        }
    }
    (n >= MIN_SAMPLES).then(|| ((nvar / n as f64).sqrt() / OPERATOR_GAIN) as f32)
}

/// Windows per block of [`Band::admitted`]'s sum.
const BLOCK: usize = 256;
/// Blocks [`Band::admitted`] sums side by side.
const LANES: usize = 8;

/// The sampled windows of one intensity band, structure of arrays: gate
/// statistic (in units of the operator's noise variance) and squared
/// operator output per window.
#[derive(Clone, Copy)]
struct Band<'a> {
    gate: &'a [f32],
    op: &'a [f32],
}

impl Band<'_> {
    /// Truncated mean of the operator's square over the windows whose gate
    /// statistic is below [`GATE`] times `gate_var`, and their count.
    fn admitted(&self, gate_var: f64, op_var: f64) -> (usize, f64) {
        let threshold = (GATE * gate_var.max(ROUNDING_OP_VAR)) as f32;
        let cut = (TRUNC * TRUNC * op_var.max(ROUNDING_OP_VAR)) as f32;
        let kept = |g: f32, o: f32| {
            let keep = (g < threshold) & (o <= cut);
            (if keep { o } else { 0.0 }, u32::from(keep))
        };
        let (mut sum, mut n) = (0f64, 0usize);
        // Branch-free blocks of `BLOCK` windows summed in f32, accumulated
        // in f64 in block order. `LANES` blocks run side by side, each in
        // its own order, so the sum does not wait on one chain of additions.
        let (gate, op) = (self.gate, self.op);
        let full = gate.len() / BLOCK;
        let mut first = 0;
        while first + LANES <= full {
            let (mut block, mut count) = ([0f32; LANES], [0u32; LANES]);
            let gates: [&[f32]; LANES] =
                core::array::from_fn(|l| &gate[(first + l) * BLOCK..(first + l + 1) * BLOCK]);
            let ops: [&[f32]; LANES] =
                core::array::from_fn(|l| &op[(first + l) * BLOCK..(first + l + 1) * BLOCK]);
            for i in 0..BLOCK {
                for l in 0..LANES {
                    let (v, c) = kept(gates[l][i], ops[l][i]);
                    block[l] += v;
                    count[l] += c;
                }
            }
            for l in 0..LANES {
                sum += f64::from(block[l]);
                n += count[l] as usize;
            }
            first += LANES;
        }
        let rest = first * BLOCK;
        for (gates, ops) in gate[rest..].chunks(BLOCK).zip(op[rest..].chunks(BLOCK)) {
            let (mut block, mut count) = (0f32, 0u32);
            for (&g, &o) in gates.iter().zip(ops) {
                let (v, c) = kept(g, o);
                block += v;
                count += c;
            }
            sum += f64::from(block);
            n += count as usize;
        }
        (n, sum)
    }

    /// Operator noise variance of the band, with its admitted window
    /// count: the fixed point of the band's own gate, starting from the
    /// quietest sixteenth of its windows. `None` when too few windows are
    /// admitted or the variance exceeds [`MAX_BAND_RATIO`] times the
    /// image-wide one (structure).
    fn variance(&self, image_var: f64) -> Option<(usize, f64)> {
        if self.gate.len() < MIN_BAND_SAMPLES {
            return None;
        }
        let quiet = quantile(self.gate, 1.0 / 16.0);
        let mut var = mean_quiet(self.op, self.gate, quiet);
        let mut n_prev = usize::MAX;
        for _ in 0..ROUNDS {
            let (n, sum) = self.admitted(var, var);
            if n == 0 {
                return None;
            }
            let next = sum / n as f64 / TRUNC_SECOND_MOMENT;
            let settled = n == n_prev || (next - var).abs() <= SETTLED * var;
            (var, n_prev) = (next, n);
            if settled {
                break;
            }
        }
        let structure = var > MAX_BAND_RATIO * image_var.max(ROUNDING_OP_VAR);
        (n_prev >= MIN_BAND_SAMPLES && !structure).then_some((n_prev, var))
    }
}

/// The sampled windows, by intensity band: the bands one after the other,
/// each in sampling order.
struct Sample {
    gate: Vec<f32>,
    op: Vec<f32>,
    /// Band `b` is windows `start[b]..start[b + 1]`.
    start: [usize; BANDS + 1],
}

impl Sample {
    fn len(&self) -> usize {
        self.op.len()
    }

    /// The windows of band `b`.
    fn band(&self, b: usize) -> Band<'_> {
        let range = self.start[b]..self.start[b + 1];
        Band {
            gate: &self.gate[range.clone()],
            op: &self.op[range],
        }
    }

    /// Sample about `windows` 4×4 windows on a grid of step
    /// `⌈√(w·h / windows)⌉`, each grid row offset by a golden-ratio fraction
    /// of the step so a coarse grid does not alias with a periodic pattern
    /// such as a board's cells. Returns the sample and the step.
    fn collect(img: &[u8], w: usize, h: usize, windows: usize) -> (Self, usize) {
        let step = (((w * h) as f64 / windows as f64).sqrt().ceil() as usize).max(1);
        // Grid row `j` has its first window at `x0` and `count` windows.
        let grid_row = |j: usize| {
            let x0 = ((j as f64 * 0.618_034).fract() * step as f64) as usize;
            (x0, (w - 3).saturating_sub(x0).div_ceil(step))
        };
        let total: usize = (0..(h - 3).div_ceil(step)).map(|j| grid_row(j).1).sum();
        // Every window in sampling order: pixel sum, gate, operator output.
        let (mut sums, mut gates, mut ops) = (vec![0; total], vec![0; total], vec![0; total]);
        let mut at = 0;
        for (j, y) in (0..h - 3).step_by(step).enumerate() {
            let (x0, count) = grid_row(j);
            let rows = &img[y * w..(y + 4) * w];
            let mut k = 0;
            while k + BATCH <= count {
                let xs = core::array::from_fn(|l| x0 + (k + l) * step);
                let [s, g, o] = window_batch(rows, w, &xs);
                sums[at..at + BATCH].copy_from_slice(&s);
                gates[at..at + BATCH].copy_from_slice(&g);
                ops[at..at + BATCH].copy_from_slice(&o);
                (k, at) = (k + BATCH, at + BATCH);
            }
            for k in k..count {
                [sums[at], gates[at], ops[at]] = window(rows, w, x0 + k * step);
                at += 1;
            }
        }
        // Band of a window by its mean (window mean / 64 = sum / 1024), or
        // `BANDS` near the rails, where clipping removes noise: those are
        // skipped.
        let band = |sum: i32| {
            if (16 * RAIL_MARGIN..=16 * (255 - RAIL_MARGIN)).contains(&sum) {
                (sum >> 10) as usize
            } else {
                BANDS
            }
        };
        let mut count = [0usize; BANDS + 1];
        for &sum in &sums {
            let b = band(sum);
            for (slot, n) in count.iter_mut().enumerate() {
                *n += usize::from(b == slot);
            }
        }
        let mut start = [0usize; BANDS + 1];
        for b in 0..BANDS {
            start[b + 1] = start[b] + count[b];
        }
        let kept = start[BANDS];
        // Grouped by band, each in sampling order; windows near the rails
        // land in one spare slot past the end.
        let (mut gate, mut op) = (vec![0f32; kept + 1], vec![0f32; kept + 1]);
        let [mut c0, mut c1, mut c2, mut c3] = [start[0], start[1], start[2], start[3]];
        for ((&sum, &g), &o) in sums.iter().zip(&gates).zip(&ops) {
            let b = band(sum);
            let k = match b {
                0 => c0,
                1 => c1,
                2 => c2,
                3 => c3,
                _ => kept,
            };
            let o = o as f32;
            (gate[k], op[k]) = (g as f32, o * o);
            c0 += usize::from(b == 0);
            c1 += usize::from(b == 1);
            c2 += usize::from(b == 2);
            c3 += usize::from(b == 3);
        }
        gate.truncate(kept);
        op.truncate(kept);
        (Self { gate, op, start }, step)
    }

    /// Image-wide operator noise variance: the fixed point where the gate
    /// it sets yields it, starting from the quietest sixteenth of the
    /// windows (below any structure).
    fn image_variance(&self) -> f64 {
        let quiet = quantile(&self.gate, 1.0 / 16.0);
        let mut var = mean_quiet(&self.op, &self.gate, quiet);
        let mut n_prev = usize::MAX;
        for _ in 0..ROUNDS {
            let (n, sum) = (0..BANDS)
                .map(|b| self.band(b).admitted(var, var))
                .fold((0, 0.0), |(n, s), (bn, bs)| (n + bn, s + bs));
            if n == 0 {
                break;
            }
            let next = sum / n as f64 / TRUNC_SECOND_MOMENT;
            let settled = n == n_prev || (next - var).abs() <= SETTLED * var;
            (var, n_prev) = (next, n);
            if settled {
                break;
            }
        }
        var
    }
}

/// The 4×4 window with top-left pixel `(x, 0)` of the four image rows
/// `rows` (stride `w`): its pixel sum, gate statistic and operator output
/// `(3, 3)`.
#[inline(always)]
fn window(rows: &[u8], w: usize, x: usize) -> [i32; 3] {
    window_stats(core::array::from_fn(|k| {
        let start = k * w + x;
        let px: [u8; 4] = rows[start..start + 4]
            .try_into()
            .expect("window lies inside the image");
        px.map(i32::from)
    }))
}

/// Pixel sum, gate statistic and operator output `(3, 3)` of the 4×4
/// window `px` (`px[y][x]`).
#[inline(always)]
fn window_stats(px: [[i32; 4]; 4]) -> [i32; 3] {
    // Polynomials down the columns, then across: `comp[i][j]` is component
    // `(i, j)` (degree `i` along x, `j` along y).
    let comp = poly4_down(transpose(poly4_down(px)));
    // On 8-bit pixels the weighted sum stays below 7.4·10⁸: exact in `i32`.
    let mut gate = 0;
    for (c, wt) in comp.iter().zip(&GATE_WEIGHT) {
        for j in 0..4 {
            gate += c[j] * c[j] * wt[j];
        }
    }
    [comp[0][0], gate, comp[3][3]]
}

/// Windows [`window_batch`] evaluates side by side.
const BATCH: usize = 8;
/// One value per window of a batch.
type Lanes = [i32; BATCH];

/// The [`BATCH`] 4×4 windows with top-left pixels `(xs[l], 0)` of the four
/// image rows `rows` (stride `w`): per window the pixel sum, the gate
/// statistic and the operator output `(3, 3)`.
#[inline(always)]
fn window_batch(rows: &[u8], w: usize, xs: &[usize; BATCH]) -> [Lanes; 3] {
    // Four pixels per window and row, packed little-endian.
    let packed: [[u32; BATCH]; 4] = core::array::from_fn(|k| {
        core::array::from_fn(|l| {
            let start = k * w + xs[l];
            u32::from_le_bytes(
                rows[start..start + 4]
                    .try_into()
                    .expect("window lies inside the image"),
            )
        })
    });
    let mut out = [[0i32; BATCH]; 3];
    // One window per iteration, the same arithmetic in every lane.
    for l in 0..BATCH {
        let px = core::array::from_fn(|k| {
            core::array::from_fn(|c| ((packed[k][l] >> (8 * c)) & 0xff) as i32)
        });
        let [sum, gate, op] = window_stats(px);
        (out[0][l], out[1][l], out[2][l]) = (sum, gate, op);
    }
    out
}

/// Operator variance of the 8-bit rounding noise (`400 / 12`): the gate is
/// never set tighter than this.
const ROUNDING_OP_VAR: f64 = 400.0 / 12.0;

/// `E[z² | |z| < TRUNC]` for a standard normal `z`: the truncated second
/// moment is divided by it.
const TRUNC_SECOND_MOMENT: f64 = 0.998_929;
/// Squared norms of the orthogonal polynomials `s`, `l`, `e`, `d`
/// ([`poly4_down`]).
const POLY_NORM2: [i32; 4] = [4, 20, 4, 20];
/// Weight of component `(i, j)` in the gate statistic, `[i][j]`: the
/// [`GATE_TERMS`] scaled to the operator's white-noise variance
/// (`‖d‖⁴ = 400`), `400 / (‖Pᵢ‖²‖Pⱼ‖²)` (1, 5 or 25); 0 elsewhere.
const GATE_WEIGHT: [[i32; 4]; 4] = {
    let mut wt = [[0; 4]; 4];
    let mut k = 0;
    while k < GATE_TERMS.len() {
        let (i, j) = GATE_TERMS[k];
        wt[i][j] = 400 / (POLY_NORM2[i] * POLY_NORM2[j]);
        k += 1;
    }
    wt
};

/// The four orthogonal polynomials `[s, l, e, d]` applied down each column
/// of four rows: `out[p][x]` is polynomial `p` over `rows[0..4][x]`.
#[inline(always)]
fn poly4_down([v0, v1, v2, v3]: [[i32; 4]; 4]) -> [[i32; 4]; 4] {
    let lanes = |f: fn(i32, i32, i32, i32) -> i32| -> [i32; 4] {
        core::array::from_fn(|x| f(v0[x], v1[x], v2[x], v3[x]))
    };
    [
        lanes(|a, b, c, d| a + b + c + d),
        lanes(|a, b, c, d| 3 * (d - a) + c - b),
        lanes(|a, b, c, d| a - b - c + d),
        lanes(|a, b, c, d| a - 3 * b + 3 * c - d),
    ]
}

/// Transpose of a 4×4 array.
#[inline(always)]
fn transpose(m: [[i32; 4]; 4]) -> [[i32; 4]; 4] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

/// Mean of `op[k]` over the windows whose `gate[k] <= quiet` (0 if none),
/// summed in window order.
fn mean_quiet(op: &[f32], gate: &[f32], quiet: f32) -> f64 {
    let (mut sum, mut n) = (0f64, 0usize);
    let mut picked = [0f32; BLOCK];
    for (ops, gates) in op.chunks(BLOCK).zip(gate.chunks(BLOCK)) {
        // Branch-free selection: write every value, keep the quiet ones.
        let mut m = 0;
        for (&o, &g) in ops.iter().zip(gates) {
            picked[m] = o;
            m += usize::from(g <= quiet);
        }
        for &v in &picked[..m] {
            sum += f64::from(v);
        }
        n += m;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// The `p`-quantile of a sub-sample of at most 255 evenly spaced values.
fn quantile(values: &[f32], p: f64) -> f32 {
    let stride = values.len().div_ceil(255).max(1);
    let mut sub: Vec<f32> = values.iter().step_by(stride).copied().collect();
    let at = ((sub.len() as f64 * p) as usize).min(sub.len() - 1);
    *sub.select_nth_unstable_by(at, f32::total_cmp).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_or_flat_images() {
        assert_eq!(estimate_noise_sigma_u8(&[0; 4], 2, 2), None);
        // Too few samples.
        assert_eq!(estimate_noise_sigma_u8(&[7; 100], 10, 10), None);
        // Flat, noise-free: zero.
        let flat = vec![100u8; 64 * 64];
        let s = estimate_noise_sigma_u8(&flat, 64, 64).expect("estimate");
        assert!(s < 0.01, "{s}");
        // Saturated: no usable window.
        let white = vec![255u8; 64 * 64];
        assert_eq!(estimate_noise_sigma_u8(&white, 64, 64), None);
    }

    #[test]
    #[should_panic(expected = "estimate_noise_sigma_u8: img.len()")]
    fn panics_on_dimension_mismatch() {
        let _ = estimate_noise_sigma_u8(&[0; 10], 4, 4);
    }
}
