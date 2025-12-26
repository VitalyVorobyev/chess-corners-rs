"""Negative patch generation for ML refiner datasets."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from synth import render_corner


_GRID_CACHE: dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _grid(patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if patch_size not in _GRID_CACHE:
        _GRID_CACHE[patch_size] = render_corner.make_patch_grid(patch_size)
    return _GRID_CACHE[patch_size]


def _pick_type(rng: np.random.Generator, cfg: Dict[str, Any]) -> str:
    types = cfg.get("types") or ["flat", "edge", "stripe", "blob", "noise", "near_corner"]
    if not isinstance(types, (list, tuple)):
        raise ValueError("neg.types must be a list")
    types = [str(t) for t in types]
    if not types:
        return "noise"

    near_prob = float(cfg.get("near_corner_prob", 0.0))
    if "near_corner" in types and near_prob > 0.0:
        if rng.uniform() < near_prob:
            return "near_corner"
        rest = [t for t in types if t != "near_corner"]
        if rest:
            return str(rng.choice(rest))
        return "near_corner"

    return str(rng.choice(types))


def _apply_range(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    low = float(rng.uniform(0.05, 0.4))
    high = float(rng.uniform(0.6, 0.95))
    if low > high:
        low, high = high, low
    return low + (high - low) * patch


def _flat(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = float(rng.uniform(0.1, 0.9))
    norm = max(1.0, float(np.max(np.abs(x))))
    gx = float(rng.uniform(-0.2, 0.2))
    gy = float(rng.uniform(-0.2, 0.2))
    return np.clip(base + gx * (x / norm) + gy * (y / norm), 0.0, 1.0)


def _edge(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    offset = float(rng.uniform(-0.3, 0.3)) * float(np.max(np.abs(x)))
    softness = float(rng.uniform(0.3, 1.0))
    dist = np.cos(theta) * x + np.sin(theta) * y - offset
    blend = 0.5 * (1.0 + np.tanh(dist / softness))
    patch = _apply_range(blend, rng)
    return np.clip(patch, 0.0, 1.0)


def _stripe(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    freq = float(rng.uniform(1.0, 4.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    axis = np.cos(theta) * x + np.sin(theta) * y
    size = float(np.max(np.abs(x)) * 2.0)
    wave = np.sin(2.0 * np.pi * freq * axis / max(size, 1.0) + phase)
    base = float(rng.uniform(0.4, 0.6))
    amp = float(rng.uniform(0.2, 0.4))
    patch = base + amp * wave
    return np.clip(patch, 0.0, 1.0)


def _blob(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = float(rng.uniform(0.2, 0.8))
    patch = np.full_like(x, base, dtype=np.float32)
    num = int(rng.integers(1, 4))
    limit = float(np.max(np.abs(x)))
    for _ in range(num):
        cx = float(rng.uniform(-limit, limit))
        cy = float(rng.uniform(-limit, limit))
        sigma = float(rng.uniform(1.2, 4.0))
        amp = float(rng.uniform(-0.5, 0.5))
        patch += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
    return np.clip(patch, 0.0, 1.0)


def _noise(rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    patch = rng.normal(0.5, 0.25, size=shape).astype(np.float32)
    return np.clip(patch, 0.0, 1.0)


def _near_corner(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    subtype = str(rng.choice(["offset", "t_junction", "quadrants"]))
    softness = float(rng.uniform(0.4, 0.9))

    if subtype == "offset":
        shift = float(rng.uniform(1.5, 4.0))
        dx = shift * float(rng.choice([-1.0, 1.0]))
        dy = shift * float(rng.choice([-1.0, 1.0]))
        val = np.tanh((x - dx) / softness) * np.tanh((y - dy) / softness)
        patch = 0.5 + 0.5 * val
        return np.clip(_apply_range(patch, rng), 0.0, 1.0)

    if subtype == "t_junction":
        edge_x = np.tanh(x / softness)
        edge_y = np.tanh(y / softness)
        mask = 0.5 * (1.0 + np.tanh(x / softness))
        val = edge_x + 0.5 * mask * edge_y
        val = np.clip(val, -1.0, 1.0)
        patch = 0.5 + 0.5 * val
        return np.clip(_apply_range(patch, rng), 0.0, 1.0)

    sx = 0.5 * (1.0 + np.tanh(x / softness))
    sy = 0.5 * (1.0 + np.tanh(y / softness))
    w00 = (1.0 - sx) * (1.0 - sy)
    w10 = sx * (1.0 - sy)
    w01 = (1.0 - sx) * sy
    w11 = sx * sy

    if rng.uniform() < 0.5:
        dark = float(rng.uniform(0.1, 0.3))
        bright = float(rng.uniform(0.7, 0.9))
        intensities = [bright, bright, bright, dark]
    else:
        dark = float(rng.uniform(0.1, 0.3))
        bright = float(rng.uniform(0.7, 0.9))
        intensities = [dark, dark, dark, bright]

    patch = (
        intensities[0] * w00
        + intensities[1] * w10
        + intensities[2] * w01
        + intensities[3] * w11
    )
    return np.clip(patch, 0.0, 1.0)


def generate_negative_patch(
    rng: np.random.Generator, patch_size: int, cfg: Dict[str, Any]
) -> np.ndarray:
    x, y = _grid(patch_size)
    neg_type = _pick_type(rng, cfg)

    if neg_type == "flat":
        patch = _flat(rng, x, y)
    elif neg_type == "edge":
        patch = _edge(rng, x, y)
    elif neg_type == "stripe":
        patch = _stripe(rng, x, y)
    elif neg_type == "blob":
        patch = _blob(rng, x, y)
    elif neg_type == "noise":
        patch = _noise(rng, x.shape)
    elif neg_type == "near_corner":
        patch = _near_corner(rng, x, y)
    else:
        patch = _noise(rng, x.shape)

    return np.clip(patch, 0.0, 1.0).astype(np.float32)
