"""Simple photometric and geometric augmentations."""

from __future__ import annotations

import numpy as np


def apply_photometric(
    patch: np.ndarray,
    rng: np.random.Generator,
    contrast_range: tuple[float, float],
    brightness_range: tuple[float, float],
    gamma_range: tuple[float, float],
) -> np.ndarray:
    contrast = float(rng.uniform(*contrast_range))
    brightness = float(rng.uniform(*brightness_range))
    gamma = float(rng.uniform(*gamma_range))

    out = np.clip(patch, 0.0, 1.0)
    out = out**gamma
    out = (out - 0.5) * contrast + 0.5
    out = out + (brightness / 255.0)
    return np.clip(out, 0.0, 1.0)


def apply_noise(
    patch: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
) -> np.ndarray:
    if noise_sigma <= 0.0:
        return patch
    sigma = noise_sigma / 255.0
    noise = rng.normal(0.0, sigma, size=patch.shape).astype(np.float32)
    out = patch + noise
    return np.clip(out, 0.0, 1.0)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def apply_blur(patch: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma <= 0.0:
        return patch
    kernel = gaussian_kernel1d(blur_sigma)
    radius = kernel.size // 2
    height, width = patch.shape

    padded = np.pad(patch, ((radius, radius), (radius, radius)), mode="reflect")
    tmp = np.empty_like(patch)
    for row in range(height):
        tmp[row, :] = np.convolve(padded[row + radius, :], kernel, mode="valid")

    padded_tmp = np.pad(tmp, ((radius, radius), (radius, radius)), mode="reflect")
    out = np.empty_like(patch)
    for col in range(width):
        out[:, col] = np.convolve(padded_tmp[:, col + radius], kernel, mode="valid")
    return out
