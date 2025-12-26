"""Homography sampling and helpers for synthetic corner warps."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _as_range(cfg: dict, key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    if key not in cfg:
        return float(default[0]), float(default[1])
    value = cfg[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a 2-element list")
    return float(value[0]), float(value[1])


def _sample_range(
    rng: np.random.Generator, cfg: dict, key: str, default: Tuple[float, float]
) -> float:
    lo, hi = _as_range(cfg, key, default)
    return float(rng.uniform(lo, hi))


def _rotation(theta: float) -> np.ndarray:
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    return np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _shear(k: float) -> np.ndarray:
    return np.array([[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def _projective(p1: float, p2: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [p1, p2, 1.0]], dtype=np.float32)


def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    x = xy[..., 0]
    y = xy[..., 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return np.stack((u, v), axis=-1)


def invert_homography(H: np.ndarray) -> np.ndarray:
    return np.linalg.inv(H).astype(np.float32)


def _normalize_homography(H: np.ndarray) -> np.ndarray:
    if H[2, 2] == 0:
        return H
    return (H / H[2, 2]).astype(np.float32)


def _center_homography(H: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
    center_xy = np.array(center, dtype=np.float32).reshape(1, 2)
    uv = apply_homography(H, center_xy)[0]
    shift = _translation(-float(uv[0]), -float(uv[1]))
    return _normalize_homography(shift @ H)


def _local_jacobian(Hinv: np.ndarray, u: float, v: float, eps: float = 1e-2) -> np.ndarray:
    base = apply_homography(Hinv, np.array([u, v], dtype=np.float32))
    du = apply_homography(Hinv, np.array([u + eps, v], dtype=np.float32))
    dv = apply_homography(Hinv, np.array([u, v + eps], dtype=np.float32))
    dx_du = (du[0] - base[0]) / eps
    dy_du = (du[1] - base[1]) / eps
    dx_dv = (dv[0] - base[0]) / eps
    dy_dv = (dv[1] - base[1]) / eps
    return np.array([[dx_du, dx_dv], [dy_du, dy_dv]], dtype=np.float32)


def _singular_values_2x2(J: np.ndarray) -> Tuple[float, float]:
    a, b = float(J[0, 0]), float(J[0, 1])
    c, d = float(J[1, 0]), float(J[1, 1])
    t = a * a + b * b + c * c + d * d
    det = (a * d - b * c) ** 2
    disc = max(t * t - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    s1 = np.sqrt(max(0.5 * (t + root), 0.0))
    s2 = np.sqrt(max(0.5 * (t - root), 0.0))
    if s1 >= s2:
        return s1, s2
    return s2, s1


def _check_homography(
    H: np.ndarray,
    patch_size: int,
    min_scale: float,
    max_scale: float,
    max_skew: float,
) -> bool:
    try:
        Hinv = invert_homography(H)
    except np.linalg.LinAlgError:
        return False

    half = (patch_size - 1) / 2.0
    points: Iterable[Tuple[float, float]] = (
        (0.0, 0.0),
        (-half, -half),
        (half, -half),
        (-half, half),
        (half, half),
    )

    for u, v in points:
        J = _local_jacobian(Hinv, u, v)
        smax, smin = _singular_values_2x2(J)
        if smin < min_scale or smax > max_scale:
            return False
        if smin <= 0.0:
            return False
        if smax / smin > max_skew:
            return False
    return True


def sample_homography(
    rng: np.random.Generator,
    cfg: dict,
    patch_size: int,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return np.eye(3, dtype=np.float32)

    attempts = int(cfg.get("max_attempts", 50))
    min_scale = float(cfg.get("min_local_scale", 0.4))
    max_scale = float(cfg.get("max_local_scale", 2.5))
    max_skew = float(cfg.get("max_skew", 1.2))

    for _ in range(max(1, attempts)):
        theta = _sample_range(rng, cfg, "rotation", (0.0, 0.0))
        shear = _sample_range(rng, cfg, "shear_range", (0.0, 0.0))
        scale_x = _sample_range(rng, cfg, "scale_x", (1.0, 1.0))
        scale_y = _sample_range(rng, cfg, "scale_y", (1.0, 1.0))
        tx = _sample_range(rng, cfg, "tx_range", (0.0, 0.0))
        ty = _sample_range(rng, cfg, "ty_range", (0.0, 0.0))
        p1 = _sample_range(rng, cfg, "p_range", (0.0, 0.0))
        p2 = _sample_range(rng, cfg, "p_range", (0.0, 0.0))

        A = _rotation(theta) @ _shear(shear) @ _scale(scale_x, scale_y)
        H = _projective(p1, p2) @ _translation(tx, ty) @ A
        H = _normalize_homography(H)
        H = _center_homography(H, center)

        if _check_homography(H, patch_size, min_scale, max_scale, max_skew):
            return H

    return np.eye(3, dtype=np.float32)
