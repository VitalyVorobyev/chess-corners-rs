//! Per-image estimate of the white pixel-noise level.
//!
//! The estimator applies the 4×4 operator `d ⊗ d`, `d = (1, −3, 3, −1)`
//! (the product of third differences along x and y). It cancels every
//! intensity that is at most quadratic along either axis — planar shading,
//! straight edges parallel to an image axis, most of a smoothly blurred
//! pattern — and for white noise of standard deviation `σ` its output has
//! standard deviation `20σ`. What it does not cancel — sharp or oblique
//! edges, corners — is removed by a gate: a window is used only when its
//! other third-order content is small against the current noise estimate.
//! The gate measures the window with `d ⊗ s`, `s ⊗ d`, `d ⊗ l` and `l ⊗ d`
//! (`s = (1, 1, 1, 1)`, `l = (−3, −1, 1, 3)`): a sharp edge or a corner
//! drives at least one of them, while a smooth gradient drives none. The
//! scale is a truncated second moment around a median-based start, so rare
//! structure that survives the gate does not inflate it.
//!
//! `d`, `l`, `s` are mutually orthogonal, so under Gaussian noise the four
//! gate outputs are independent of the operator output: the gate removes
//! structure without biasing the noise statistic. The gate threshold and
//! the noise scale are iterated to a fixed point over the whole image.
//!
//! Sensor noise grows with brightness, and a single gate admits the
//! noisier windows less often. The variance is therefore measured per
//! intensity band on the windows the gate admits (still unbiased within a
//! band), and the bands are averaged with weights `admitted / P(admit)`,
//! where `P(admit)` is the χ²₄ probability that a noise-only window of the
//! band's level passes the gate. A band whose level is far above the gate
//! (`P(admit)` below ¼) holds structure, not noise, and is dropped.
//!
//! The windows are sampled on a regular grid, sparse first and denser only
//! when too few pass the gate. Each sampling fills a histogram of (band,
//! gate statistic, operator output); everything else works on the
//! histogram alone.

/// Intensity bands, by window mean: `[0, 64)`, `[64, 128)`, ….
const BANDS: usize = 4;
/// Width of an `|operator output|` bin, in gray levels.
const R_WIDTH: u32 = 4;
/// Bins of `|operator output|` (the last collects everything beyond).
const R_BINS: usize = 256;
/// Bins of the gate statistic `q`: bin 0 holds `q = 0`, bin `b ≥ 1` holds
/// `q ∈ [2^(b−1), 2^b)`, and the last collects `q ≥ 2^22` (beyond the gate
/// for any σ below 36 gray levels).
const Q_BINS: usize = 24;
/// 4×4 windows sampled at first; images whose gate admits fewer than
/// [`ENOUGH_SAMPLES`] of them are resampled four times as densely, up to
/// [`MAX_WINDOWS`].
const FIRST_WINDOWS: usize = 1 << 12;
/// Upper bound on the sampled windows (every window of a 256×256 image).
const MAX_WINDOWS: usize = 1 << 16;
/// Admitted windows that make the estimate good to about 2%.
const ENOUGH_SAMPLES: u64 = 1024;
/// A band needs this many gated windows to contribute.
const MIN_BAND_SAMPLES: u64 = 128;
/// A band contributes only when a noise-only window at its level passes
/// the gate with at least this probability.
const MIN_ADMIT: f64 = 0.25;
/// Below this many gated windows over all bands the estimate is not
/// reported (the relative standard error of σ̂ is about `1/sqrt(2n)`, 3%
/// here).
const MIN_SAMPLES: u64 = 512;
/// Gate statistic `q = 5·(a² + b²) + c₁² + c₂²` with `a = d ⊗ s`,
/// `b = s ⊗ d` (variance `80σ²` each) and `c₁ = d ⊗ l`, `c₂ = l ⊗ d`
/// (variance `400σ²` each): `q / (400σ²)` is χ²₄ under white noise.
const GATE_GAIN2: f64 = 400.0;
/// Keep windows with `q < GATE · 400σ²`; with the power-of-two bins this
/// keeps 59–91% of noise-only windows.
const GATE: f64 = 8.0;
/// Truncation of the second moment, in units of the median-based scale.
const TRUNC: f64 = 4.0;
/// `E[z² | |z| < TRUNC]` for a standard normal `z`: the truncated second
/// moment is divided by it.
const TRUNC_SECOND_MOMENT: f64 = 0.998_929;
/// Median of `|z|` for a standard normal `z`.
const HALF_NORMAL_MEDIAN: f64 = 0.674_49;
/// Standard deviation of the operator output per unit pixel noise σ
/// (`sqrt(Σ wᵢ²) = sqrt(20 · 20)`).
const OPERATOR_GAIN: f64 = 20.0;

/// Window counts of one intensity band, by gate bin and `|r|` bin (made
/// cumulative over gate bins once filled).
type BandHistogram = [[u32; R_BINS]; Q_BINS];

/// Estimate the standard deviation of white pixel noise in an 8-bit
/// grayscale image, in gray levels.
///
/// Measures the noise as it appears in the 8-bit image, so the estimate
/// includes the rounding to integers (`1/12` gray level² when the noise
/// spans several levels). It is designed for images dominated by strong
/// edges, such as calibration boards: edges, corners and smooth shading are
/// excluded from the statistic, and so are saturated pixels. Under
/// signal-dependent noise the estimate is the root of the mean noise
/// variance over the image's structure-free areas. Spatially correlated
/// noise (demosaicing, compression, resampling) is partly cancelled by the
/// estimator and is under-reported.
///
/// An image with no structure-free area — small, sharp, oblique cells
/// filling the whole frame — gives no estimate at low noise and an
/// over-estimate at high noise, where the gate can no longer tell such
/// edges from noise.
///
/// Deterministic and cheap: it samples about 4 000 4×4 windows on a
/// regular grid, so the cost does not grow with the image size, and
/// samples more densely (up to about 65 000 windows) only when too few of
/// them are free of edges. It allocates one fixed 96 KiB histogram per
/// call.
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
    // Sample sparsely first; if too few windows pass the gate, sample
    // four times as densely, up to `MAX_WINDOWS` or every window.
    let mut hist = vec![[[0u32; R_BINS]; Q_BINS]; BANDS];
    let grid_step =
        |windows: usize| ((((w * h) as f64 / windows as f64).sqrt().ceil()) as usize).max(1);
    let min_step = grid_step(MAX_WINDOWS);
    let mut windows = FIRST_WINDOWS;
    loop {
        let step = grid_step(windows).max(min_step);
        fill_histogram(&mut hist, img, w, h, step);
        let estimate = noise_from_histogram(&hist);
        let enough = matches!(estimate, Some((_, n)) if n >= ENOUGH_SAMPLES);
        if enough || step == min_step {
            return estimate.map(|(sigma, _)| sigma);
        }
        windows *= 4;
    }
}

/// Noise σ (gray levels) from a filled histogram, with the number of
/// windows behind it.
fn noise_from_histogram(hist: &[BandHistogram]) -> Option<(f32, u64)> {
    let qmax = gate_fixed_point(hist)?;
    // Gate upper bound: windows with q < 2^qmax pass (q = 0 for qmax = 0).
    let gate_q = if qmax == 0 {
        0.5
    } else {
        (1u64 << qmax) as f64
    };
    let (mut weight, mut wvar, mut admitted) = (0f64, 0f64, 0u64);
    for band in hist {
        let Some((n, var_r)) = scale(&pooled(band, qmax)) else {
            continue;
        };
        // `q / (400σ²)` is χ²₄ for a noise-only window, and `var_r = 400σ²`.
        let admit = chi2_4_cdf(gate_q / var_r);
        if admit < MIN_ADMIT {
            continue;
        }
        let wb = n as f64 / admit;
        weight += wb;
        wvar += wb * var_r;
        admitted += n;
    }
    (admitted >= MIN_SAMPLES).then(|| (((wvar / weight).sqrt() / OPERATOR_GAIN) as f32, admitted))
}

/// Add the 4×4 window with top-left pixel `(x, y)` to the histogram.
#[inline]
fn record(hist: &mut [BandHistogram], img: &[u8], w: usize, x: usize, y: usize) {
    let px: [[u8; 4]; 4] = core::array::from_fn(|k| {
        let start = (y + k) * w + x;
        img[start..start + 4]
            .try_into()
            .expect("window lies inside the image")
    });
    // Saturated pixels carry no noise.
    if px.iter().flatten().any(|&p| p == 0 || p == u8::MAX) {
        return;
    }
    // Down each column: `d`, `s` and `l` over the four rows.
    let col = |c: usize| {
        let [p0, p1, p2, p3] = px.map(|row| i32::from(row[c]));
        (
            p0 - 3 * p1 + 3 * p2 - p3,
            p0 + p1 + p2 + p3,
            3 * (p3 - p0) + p2 - p1,
        )
    };
    let [(d0, s0, l0), (d1, s1, l1), (d2, s2, l2), (d3, s3, l3)] = [col(0), col(1), col(2), col(3)];
    // Across the columns, `x ⊗ y` kernels: the operator `d ⊗ d`, and the
    // gate's `d ⊗ s`, `s ⊗ d`, `d ⊗ l`, `l ⊗ d`.
    let r = d0 - 3 * d1 + 3 * d2 - d3;
    let ds = s0 - 3 * s1 + 3 * s2 - s3;
    let sd = d0 + d1 + d2 + d3;
    let dl = l0 - 3 * l1 + 3 * l2 - l3;
    let ld = 3 * (d3 - d0) + d2 - d1;
    let q = 5 * ((ds * ds) as u32 + (sd * sd) as u32) + (dl * dl) as u32 + (ld * ld) as u32;
    let qbin = ((u32::BITS - q.leading_zeros()) as usize).min(Q_BINS - 1);
    let rbin = ((r.unsigned_abs() / R_WIDTH) as usize).min(R_BINS - 1);
    // Window mean / 64 = window sum / 1024.
    let band = ((s0 + s1 + s2 + s3) >> 10) as usize;
    hist[band][qbin][rbin] += 1;
}

/// Refill the `(band, gate bin, |operator output| bin)` histogram from the
/// 4×4 windows on a grid of the given step. Each grid row is offset by a
/// golden-ratio fraction of the step, so a coarse grid does not alias with
/// a periodic pattern such as a board's cells.
fn fill_histogram(hist: &mut [BandHistogram], img: &[u8], w: usize, h: usize, step: usize) {
    for band in hist.iter_mut() {
        for row in band.iter_mut() {
            row.fill(0);
        }
    }
    for (j, y) in (0..h - 3).step_by(step).enumerate() {
        let x0 = ((j as f64 * 0.618_034).fract() * step as f64) as usize;
        for x in (x0..w - 3).step_by(step) {
            record(hist, img, w, x, y);
        }
    }
    // Cumulative along the gate axis: `hist[band][q]` then counts the
    // windows of gate bins `0..=q`.
    for band in hist.iter_mut() {
        for q in 1..Q_BINS {
            let (lo, hi) = band.split_at_mut(q);
            for (upper, &lower) in hi[0].iter_mut().zip(&lo[q - 1]) {
                *upper += lower;
            }
        }
    }
}

/// `|r|` counts of one band over gate bins `0..=qmax`.
fn pooled(band: &BandHistogram, qmax: usize) -> [u64; R_BINS] {
    band[qmax].map(u64::from)
}

/// `|r|` counts over all bands and gate bins `0..=qmax`.
fn pooled_all(hist: &[BandHistogram], qmax: usize) -> [u64; R_BINS] {
    let mut out = [0u64; R_BINS];
    for band in hist {
        for (o, &n) in out.iter_mut().zip(&band[qmax]) {
            *o += u64::from(n);
        }
    }
    out
}

/// The gate bin at which the gate and the robust noise scale agree, over
/// the whole image: the gate set by σ̂ yields σ̂. `None` when that gate
/// admits fewer than [`MIN_SAMPLES`] windows.
fn gate_fixed_point(hist: &[BandHistogram]) -> Option<usize> {
    // Start from the least-structured windows (the lowest gate bins holding
    // at least 1/16 of all windows) and move the gate to the bin the
    // estimate implies until it settles. Starting low converges to the
    // smallest self-consistent gate, which admits the least structure.
    let admitted = |q: usize| -> u64 {
        hist.iter()
            .map(|band| band[q].iter().map(|&n| u64::from(n)).sum::<u64>())
            .sum()
    };
    let start_count = (admitted(Q_BINS - 1) / 16).max(MIN_SAMPLES);
    let qmax = (0..Q_BINS)
        .find(|&q| admitted(q) >= start_count)
        .unwrap_or(Q_BINS - 1);
    let mut qmax = qmax;
    let mut sigma_r = median_scale(&pooled_all(hist, qmax), MIN_SAMPLES)?;
    // Statistical jitter could make the gate alternate between adjacent
    // bins, so the rounds are capped (either bin is then a valid gate).
    for _ in 0..16 {
        let next = gate_bin(sigma_r / OPERATOR_GAIN);
        if next == qmax {
            break;
        }
        qmax = next;
        sigma_r = median_scale(&pooled_all(hist, qmax), MIN_SAMPLES)?;
    }
    Some(qmax)
}

/// Variance of the operator output from gated counts, with the number of
/// windows behind it: a truncated second moment around the median-based
/// scale. `None` below [`MIN_BAND_SAMPLES`] windows.
fn scale(counts: &[u64; R_BINS]) -> Option<(u64, f64)> {
    let sigma_r = median_scale(counts, MIN_BAND_SAMPLES)?;
    // Bin `k` holds |r| ∈ {4k, …, 4k + 3}: its mean square is taken as that
    // of its mean plus the within-bin variance 5/4.
    let cut = TRUNC * sigma_r;
    let w = f64::from(R_WIDTH);
    let mut n = 0u64;
    let mut s2 = 0f64;
    for (k, &c) in counts.iter().enumerate().take(R_BINS - 1) {
        let mean = w * k as f64 + 1.5;
        if mean > cut {
            break;
        }
        n += c;
        s2 += c as f64 * (mean * mean + 1.25);
    }
    (n >= MIN_BAND_SAMPLES).then(|| (n, s2 / n as f64 / TRUNC_SECOND_MOMENT))
}

/// CDF of the χ² distribution with 4 degrees of freedom.
fn chi2_4_cdf(x: f64) -> f64 {
    let h = 0.5 * x;
    1.0 - (-h).exp() * (1.0 + h)
}

/// Largest gate bin whose upper bound lies below the gate for pixel noise
/// `sigma`.
fn gate_bin(sigma: f64) -> usize {
    let limit = GATE * GATE_GAIN2 * sigma * sigma;
    // Bin b ≥ 1 covers [2^(b−1), 2^b): keep it when 2^b ≤ limit.
    let mut b = 0;
    while b + 2 < Q_BINS && ((1u64 << (b + 1)) as f64) <= limit {
        b += 1;
    }
    b
}

/// Standard deviation of the operator output from the median of `|r|`.
/// Bin `k` holds the integers `{4k, …, 4k + 3}`, taken as the interval
/// `[4k − ½, 4k + 3.5)` (`[0, 3.5)` for `k = 0`), and the median is
/// interpolated inside it. `None` below `min_samples` windows.
fn median_scale(counts: &[u64; R_BINS], min_samples: u64) -> Option<f64> {
    let total: u64 = counts.iter().sum();
    if total < min_samples {
        return None;
    }
    let half = total as f64 / 2.0;
    let w = f64::from(R_WIDTH);
    let mut below = 0f64;
    for (k, &c) in counts.iter().enumerate() {
        let c = c as f64;
        if below + c >= half {
            let (lo, width) = if k == 0 {
                (0.0, w - 0.5)
            } else {
                (w * k as f64 - 0.5, w)
            };
            return Some((lo + width * (half - below) / c) / HALF_NORMAL_MEDIAN);
        }
        below += c;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_or_flat_images() {
        assert_eq!(estimate_noise_sigma_u8(&[0; 4], 2, 2), None);
        // Too few samples.
        assert_eq!(estimate_noise_sigma_u8(&[7; 100], 10, 10), None);
        // Flat, noise-free: close to zero (the 4-level bins of the operator
        // output resolve σ to about 0.1 gray level).
        let flat = vec![100u8; 64 * 64];
        let s = estimate_noise_sigma_u8(&flat, 64, 64).expect("estimate");
        assert!(s < 0.1, "{s}");
        // Saturated: no usable window.
        let white = vec![255u8; 64 * 64];
        assert_eq!(estimate_noise_sigma_u8(&white, 64, 64), None);
    }

    #[test]
    #[should_panic(expected = "estimate_noise_sigma_u8: img.len()")]
    fn panics_on_dimension_mismatch() {
        let _ = estimate_noise_sigma_u8(&[0; 10], 4, 4);
    }

    #[test]
    fn gate_bin_is_monotone() {
        assert_eq!(gate_bin(0.0), 0);
        let mut prev = 0;
        for i in 1..200 {
            let b = gate_bin(f64::from(i) * 0.1);
            assert!(b >= prev);
            prev = b;
        }
        assert_eq!(gate_bin(1e6), Q_BINS - 2);
    }
}
