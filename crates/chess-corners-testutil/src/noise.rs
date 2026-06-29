//! Deterministic image degradations: separable Gaussian blur and seeded
//! additive Gaussian noise. Both operate in place on a `size × size`
//! row-major 8-bit image.

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

/// Add seeded zero-mean Gaussian noise of standard deviation `sigma` (gray
/// levels) to every pixel in place. A non-positive `sigma` is a no-op.
///
/// Uses a 64-bit LCG seeded by `seed` feeding a Box–Muller transform, so the
/// output is fully reproducible for a given `seed`.
pub fn add_gaussian_noise(img: &mut [u8], sigma: f32, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let mut uniform = || -> f32 { (next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0) };
    for px in img.iter_mut() {
        let u1 = uniform();
        let u2 = uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let v = *px as f32 + z * sigma;
        *px = v.round().clamp(0.0, 255.0) as u8;
    }
}
