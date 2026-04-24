"""Render and sample an ideal chess corner pattern."""

from __future__ import annotations

import numpy as np


def make_render_grid(super_res: int, extent: float) -> tuple[np.ndarray, np.ndarray]:
    size = int(round(2.0 * extent * super_res)) + 1
    coords = np.linspace(-extent, extent, size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coords, coords)
    return x_grid, y_grid


def render_ideal_corner_from_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta: float,
    scale: float,
    edge_softness: float,
) -> np.ndarray:
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    x_r = cos_t * x_grid - sin_t * y_grid
    y_r = sin_t * x_grid + cos_t * y_grid
    denom = np.float32(max(edge_softness * scale, 1e-6))
    x_s = x_r / denom
    y_s = y_r / denom
    img = 0.5 + 0.5 * np.tanh(x_s) * np.tanh(y_s)
    return img.astype(np.float32)


def render_ideal_corner(
    super_res: int,
    extent: float,
    theta: float,
    scale: float,
    edge_softness: float = 0.15,
) -> np.ndarray:
    x_grid, y_grid = make_render_grid(super_res, extent)
    return render_ideal_corner_from_grid(x_grid, y_grid, theta, scale, edge_softness)


def render_hard_cells_from_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta: float,
    cell_size_px: float,
) -> np.ndarray:
    """Render a periodic chessboard with hard cell boundaries.

    Cells are ``cell_size_px`` wide in patch-pixel units. The corner of
    interest sits at the grid origin; cell parity alternates so the
    pattern has a proper chessboard intersection at (0, 0). Rotation
    ``theta`` (radians) orients the cell grid.

    The output is normalised to ``{0.0, 1.0}``. Anti-aliasing at the
    cell edges is produced downstream when the caller bilinearly
    samples this high-resolution image at the 21×21 patch grid — the
    same mechanism the tanh renderer relies on.
    """
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    x_r = cos_t * x_grid - sin_t * y_grid
    y_r = sin_t * x_grid + cos_t * y_grid
    cell = np.float32(max(cell_size_px, 1e-6))
    cx = np.floor(x_r / cell).astype(np.int32)
    cy = np.floor(y_r / cell).astype(np.int32)
    parity = (cx + cy) & 1
    return parity.astype(np.float32)


def render_corner_pattern_from_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta: float,
    scale: float,
    edge_softness: float,
    cell_size_px: float,
    render_mode: str,
) -> np.ndarray:
    """Dispatch rendering by mode.

    - ``tanh`` (legacy, v1-v4): smooth infinite saddle.
      Depends on ``scale`` and ``edge_softness``. Ignores ``cell_size_px``.
    - ``hard_cells`` (v5+): periodic chessboard with hard cell
      boundaries and anti-aliasing via the downstream bilinear
      sampler. Depends on ``cell_size_px``. Ignores ``scale`` and
      ``edge_softness``.
    """
    if render_mode == "tanh":
        return render_ideal_corner_from_grid(x_grid, y_grid, theta, scale, edge_softness)
    if render_mode == "hard_cells":
        return render_hard_cells_from_grid(x_grid, y_grid, theta, cell_size_px)
    raise ValueError(f"unknown render_mode {render_mode!r}")


def make_patch_grid(patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    center = (patch_size - 1) / 2.0
    coords = np.arange(patch_size, dtype=np.float32) - center
    x_grid, y_grid = np.meshgrid(coords, coords)
    return x_grid, y_grid


def bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = img.shape
    u = np.clip(u, 0.0, width - 1.001)
    v = np.clip(v, 0.0, height - 1.001)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u0 = np.clip(u0, 0, width - 2)
    v0 = np.clip(v0, 0, height - 2)
    u1 = u0 + 1
    v1 = v0 + 1
    du = u - u0
    dv = v - v0
    i00 = img[v0, u0]
    i01 = img[v0, u1]
    i10 = img[v1, u0]
    i11 = img[v1, u1]
    return (
        (1.0 - du) * (1.0 - dv) * i00
        + du * (1.0 - dv) * i01
        + (1.0 - du) * dv * i10
        + du * dv * i11
    )


def sample_patch_from_image(
    img: np.ndarray,
    extent: float,
    super_res: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    xs = x_grid - dx
    ys = y_grid - dy
    return sample_from_image(img, extent, super_res, xs, ys)


def sample_from_image(
    img: np.ndarray,
    extent: float,
    super_res: int,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    u = (xs + extent) * super_res
    v = (ys + extent) * super_res
    return bilinear_sample(img, u, v).astype(np.float32)
