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
    let bands: Vec<(usize, f64)> = sample
        .bands
        .iter()
        .filter_map(|band| band.variance(image_var))
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

/// The sampled windows of one intensity band, structure of arrays: gate
/// statistic (in units of the operator's noise variance) and squared
/// operator output per window.
#[derive(Default)]
struct Band {
    gate: Vec<f32>,
    op: Vec<f32>,
}

impl Band {
    /// Truncated mean of the operator's square over the windows whose gate
    /// statistic is below [`GATE`] times `gate_var`, and their count.
    fn admitted(&self, gate_var: f64, op_var: f64) -> (usize, f64) {
        let threshold = (GATE * gate_var.max(ROUNDING_OP_VAR)) as f32;
        let cut = (TRUNC * TRUNC * op_var.max(ROUNDING_OP_VAR)) as f32;
        let (mut sum, mut n) = (0f64, 0usize);
        // Branch-free blocks summed in f32, accumulated in f64.
        for (gates, ops) in self.gate.chunks(256).zip(self.op.chunks(256)) {
            let (mut block, mut count) = (0f32, 0u32);
            for (&g, &o) in gates.iter().zip(ops) {
                let keep = (g < threshold) & (o <= cut);
                block += if keep { o } else { 0.0 };
                count += u32::from(keep);
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
        let quiet = quantile(&self.gate, 1.0 / 16.0);
        let mut var = mean_where(&self.op, |k| self.gate[k] <= quiet);
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

/// The sampled windows, by intensity band.
struct Sample {
    bands: [Band; BANDS],
}

impl Sample {
    fn len(&self) -> usize {
        self.bands.iter().map(|b| b.op.len()).sum()
    }

    /// Sample about `windows` 4×4 windows on a grid of step
    /// `⌈√(w·h / windows)⌉`, each grid row offset by a golden-ratio fraction
    /// of the step so a coarse grid does not alias with a periodic pattern
    /// such as a board's cells. Returns the sample and the step.
    fn collect(img: &[u8], w: usize, h: usize, windows: usize) -> (Self, usize) {
        let step = (((w * h) as f64 / windows as f64).sqrt().ceil() as usize).max(1);
        let mut sample = Self {
            bands: Default::default(),
        };
        for (j, y) in (0..h - 3).step_by(step).enumerate() {
            let x0 = ((j as f64 * 0.618_034).fract() * step as f64) as usize;
            for x in (x0..w - 3).step_by(step) {
                sample.record(img, w, x, y);
            }
        }
        (sample, step)
    }

    /// Add the 4×4 window with top-left pixel `(x, y)`.
    #[inline]
    fn record(&mut self, img: &[u8], w: usize, x: usize, y: usize) {
        let row = |k: usize| -> [i32; 4] {
            let start = (y + k) * w + x;
            let px: [u8; 4] = img[start..start + 4]
                .try_into()
                .expect("window lies inside the image");
            px.map(i32::from)
        };
        // Polynomials across each row, then down: `comp[j][i]` is component
        // `(i, j)` (degree `i` along x, `j` along y).
        let comp = poly4_rows(poly4_rows([row(0), row(1), row(2), row(3)]));
        // Window mean, via the sum `(0, 0)`: skip windows near the rails,
        // where clipping removes noise.
        let sum = comp[0][0];
        if !(16 * RAIL_MARGIN..=16 * (255 - RAIL_MARGIN)).contains(&sum) {
            return;
        }
        // Each component scaled to the operator's white-noise variance
        // (`‖d‖⁴ = 400`): `400 / (‖Pᵢ‖²‖Pⱼ‖²)`.
        let sq = |i: usize, j: usize| {
            let c = i64::from(comp[j][i]);
            c * c * i64::from(400 / (POLY_NORM2[i] * POLY_NORM2[j]))
        };
        let gate: i64 = GATE_TERMS.iter().map(|&(i, j)| sq(i, j)).sum();
        let r = comp[3][3] as f32;
        // Window mean / 64 = window sum / 1024.
        let band = &mut self.bands[(sum >> 10) as usize];
        band.gate.push(gate as f32);
        band.op.push(r * r);
    }

    /// Image-wide operator noise variance: the fixed point where the gate
    /// it sets yields it, starting from the quietest sixteenth of the
    /// windows (below any structure).
    fn image_variance(&self) -> f64 {
        let all_gates: Vec<f32> = self
            .bands
            .iter()
            .flat_map(|b| b.gate.iter().copied())
            .collect();
        let quiet = quantile(&all_gates, 1.0 / 16.0);
        let (mut sum, mut n) = (0f64, 0usize);
        for band in &self.bands {
            for (&g, &o) in band.gate.iter().zip(&band.op) {
                if g <= quiet {
                    sum += f64::from(o);
                    n += 1;
                }
            }
        }
        let mut var = sum / n.max(1) as f64;
        let mut n_prev = usize::MAX;
        for _ in 0..ROUNDS {
            let (n, sum) = self
                .bands
                .iter()
                .map(|b| b.admitted(var, var))
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

/// Operator variance of the 8-bit rounding noise (`400 / 12`): the gate is
/// never set tighter than this.
const ROUNDING_OP_VAR: f64 = 400.0 / 12.0;

/// `E[z² | |z| < TRUNC]` for a standard normal `z`: the truncated second
/// moment is divided by it.
const TRUNC_SECOND_MOMENT: f64 = 0.998_929;
/// Squared norms of the orthogonal polynomials `s`, `l`, `e`, `d`
/// ([`poly4`]).
const POLY_NORM2: [i32; 4] = [4, 20, 4, 20];

/// `poly4` applied across each of four rows, the results stacked by
/// polynomial: `out[p][r]` is polynomial `p` across row `r`.
#[inline]
fn poly4_rows(rows: [[i32; 4]; 4]) -> [[i32; 4]; 4] {
    transpose(rows.map(poly4))
}

/// Transpose of a 4×4 array.
#[inline]
fn transpose(m: [[i32; 4]; 4]) -> [[i32; 4]; 4] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

/// The four orthogonal polynomials applied to four values: `[s, l, e, d]`.
#[inline]
fn poly4([v0, v1, v2, v3]: [i32; 4]) -> [i32; 4] {
    [
        v0 + v1 + v2 + v3,
        3 * (v3 - v0) + v2 - v1,
        v0 - v1 - v2 + v3,
        v0 - 3 * v1 + 3 * v2 - v3,
    ]
}

/// Mean of `values[k]` over the `k` that `keep` selects (0 if none).
fn mean_where(values: &[f32], keep: impl Fn(usize) -> bool) -> f64 {
    let (mut sum, mut n) = (0f64, 0usize);
    for (k, &v) in values.iter().enumerate() {
        if keep(k) {
            sum += f64::from(v);
            n += 1;
        }
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
