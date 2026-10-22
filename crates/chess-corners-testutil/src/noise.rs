//! Deterministic image degradations: separable Gaussian blur and seeded
//! additive Gaussian noise (white or signal-dependent). All operate in
//! place on a row-major 8-bit image.

/// Separable Gaussian blur with a 3σ radius and edge clamping, applied in
/// place. A non-positive `sigma` is a no-op, so callers can sweep
/// `sigma = 0.0` (the "clean" condition) without special-casing it.
pub fn gaussian_blur(img: &mut [u8], size: usize, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let klen = 2 * radius + 1;
    let mut kernel = vec![0f32; klen];
    let mut sum = 0f32;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        *k = (-(x * x) / (2.0 * sigma * sigma)).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    let mut tmp = vec![0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += img[y * size + sx] as f32 * k;
            }
            tmp[y * size + x] = acc;
        }
    }
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, size as i32 - 1) as usize;
                acc += tmp[sy * size + x] * k;
            }
            img[y * size + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Seeded standard-normal source: a 64-bit LCG feeding a Box–Muller
/// transform (cosine branch only, two uniforms per draw), so a given seed
/// reproduces the same sequence on every platform.
struct GaussianSource {
    state: u64,
}

impl GaussianSource {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    /// Uniform in the open interval (0, 1).
    fn uniform(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The high 32 bits: dividing by `u32::MAX` needs the full range, or
        // the uniform would be confined to a fraction of (0, 1).
        let bits = (self.state >> 32) as u32;
        (bits as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    }

    /// One standard-normal draw.
    fn next(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
    }
}

/// Add seeded zero-mean Gaussian noise of standard deviation `sigma` (gray
/// levels) to every pixel in place, then round and clamp to `[0, 255]`. A
/// non-positive `sigma` is a no-op.
///
/// Fully reproducible for a given `seed`. The rounding adds a further
/// `1/12` gray-level² of (non-Gaussian) quantization variance.
pub fn add_gaussian_noise(img: &mut [u8], sigma: f32, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let mut rng = GaussianSource::new(seed);
    for px in img.iter_mut() {
        let v = *px as f32 + rng.next() * sigma;
        *px = v.round().clamp(0.0, 255.0) as u8;
    }
}

/// Add seeded zero-mean, signal-dependent Gaussian noise in place: pixel
/// `I` receives noise of variance `a + b·I` (gray levels²), the usual
/// read-noise-plus-shot-noise sensor model. The variance is taken from
/// the pixel's value *before* the noise, clamped at zero, and the result is
/// rounded and clamped to `[0, 255]` like [`add_gaussian_noise`].
///
/// Fully reproducible for a given `seed`; `a = σ², b = 0` draws the same
/// noise as `add_gaussian_noise(img, σ, seed)`.
pub fn add_signal_dependent_noise(img: &mut [u8], a: f32, b: f32, seed: u64) {
    let mut rng = GaussianSource::new(seed);
    for px in img.iter_mut() {
        let sigma = (a + b * *px as f32).max(0.0).sqrt();
        let v = *px as f32 + rng.next() * sigma;
        *px = v.round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(img: &[u8]) -> u64 {
        img.iter().fold(0xcbf29ce484222325u64, |h, &v| {
            (h ^ v as u64).wrapping_mul(0x100000001b3)
        })
    }

    /// Pins the white-noise sequence: tests and perf fixtures across the
    /// workspace were captured against it.
    #[test]
    fn gaussian_noise_sequence_is_pinned() {
        let mut img = vec![128u8; 4096];
        add_gaussian_noise(&mut img, 7.5, 42);
        assert_eq!(fnv1a(&img), 0xc2284820e7c1e0ee);
    }

    #[test]
    fn signal_dependent_matches_white_when_b_is_zero() {
        let mut white = vec![100u8; 1024];
        let mut sd = white.clone();
        add_gaussian_noise(&mut white, 3.0, 7);
        add_signal_dependent_noise(&mut sd, 9.0, 0.0, 7);
        assert_eq!(white, sd);
    }

    /// Empirical variance at two intensities matches `a + b·I` plus the
    /// `1/12` rounding variance. 20 000 samples per level give the sample
    /// variance a relative standard error of `sqrt(2/20000) = 1%`; the
    /// ±5% band is five standard errors.
    #[test]
    fn signal_dependent_variance_follows_intensity() {
        const N: usize = 20_000;
        let (a, b) = (1.0f32, 0.04f32);
        for (level, seed) in [(50u8, 11u64), (200, 12)] {
            let mut img = vec![level; N];
            add_signal_dependent_noise(&mut img, a, b, seed);
            let mean = img.iter().map(|&v| v as f64).sum::<f64>() / N as f64;
            let var = img.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (N - 1) as f64;
            let want = f64::from(a + b * level as f32) + 1.0 / 12.0;
            assert!(
                (mean - level as f64).abs() < 0.1,
                "level {level}: mean {mean}"
            );
            assert!(
                (var / want - 1.0).abs() < 0.05,
                "level {level}: variance {var}, expected {want}"
            );
        }
    }
}
