#!/usr/bin/env python3
"""Render single-corner chessboard crops under projective warp with detector overlays.

Generates:

- ``book/src/img/readme_warp_moderate.png`` — single chessboard corner under
  a moderate projective tilt (yaw 30°, pitch 20°).
- ``book/src/img/readme_warp_strong.png`` — single corner under a much
  stronger tilt (yaw 55°, pitch 40°).

Each PNG shows one detected corner with its two measured grid axes drawn on
top. The axes are **not** assumed orthogonal — under projective warp they
tilt independently, and the overlay shows the detector recovering that tilt.
Gaussian blur is applied after the warp so the image looks like a slightly
defocused camera capture rather than a synthetic checkerboard.

Run from the workspace root, after building the wheel into ``.venv``::

    (cd crates/chess-corners-py && maturin develop --release)
    .venv/bin/python tools/render_readme_warp_overlays.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import chess_corners as cc

WORKSPACE = Path(__file__).resolve().parents[1]
BOOK_IMG = WORKSPACE / "book" / "src" / "img"

# Texture: 9x6 inner-corner chessboard rendered at 60 pixels per square.
INNER_X, INNER_Y = 9, 6
SQUARE_PX = 60
TEX_W = (INNER_X + 1) * SQUARE_PX
TEX_H = (INNER_Y + 1) * SQUARE_PX

IMG_W, IMG_H = 1024, 720
FOCAL = 1100.0
PRINCIPAL = (IMG_W / 2.0, IMG_H / 2.0)

# Crop window around the chosen corner, output PNG size.
CROP = 220

AXIS0_COLOR = "#ff5a5a"
AXIS1_COLOR = "#37c46a"


def chessboard_texture() -> np.ndarray:
    """Black/white chessboard, no anti-aliasing (the warp + blur add it)."""
    tex = np.zeros((TEX_H, TEX_W), dtype=np.uint8)
    for iy in range(INNER_Y + 1):
        for ix in range(INNER_X + 1):
            if (ix + iy) % 2 == 1:
                y0, y1 = iy * SQUARE_PX, (iy + 1) * SQUARE_PX
                x0, x1 = ix * SQUARE_PX, (ix + 1) * SQUARE_PX
                tex[y0:y1, x0:x1] = 235
    return tex


def rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def texture_to_image_homography(yaw: float, pitch: float, roll: float, z: float) -> np.ndarray:
    """Homography mapping texture pixels (u, v) -> image pixels (x, y).

    Texture is placed on the Z=0 plane recentered on the optical axis;
    one texture pixel = one world unit; camera looks down +Z.
    """
    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)
    t = np.array([-TEX_W / 2.0, -TEX_H / 2.0, z], dtype=np.float64)
    K = np.array(
        [[FOCAL, 0, PRINCIPAL[0]], [0, FOCAL, PRINCIPAL[1]], [0, 0, 1]], dtype=np.float64
    )
    M = np.column_stack([R[:, 0], R[:, 1], t])
    return K @ M


def bilinear_sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = tex.shape
    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)

    valid = (u0 >= 0) & (u0 + 1 < W) & (v0 >= 0) & (v0 + 1 < H)
    out = np.zeros(uv.shape[0], dtype=np.float32)

    u0c = np.clip(u0, 0, W - 2)
    v0c = np.clip(v0, 0, H - 2)
    a = tex[v0c, u0c].astype(np.float32)
    b = tex[v0c, u0c + 1].astype(np.float32)
    c = tex[v0c + 1, u0c].astype(np.float32)
    d = tex[v0c + 1, u0c + 1].astype(np.float32)
    val = (1 - du) * (1 - dv) * a + du * (1 - dv) * b + (1 - du) * dv * c + du * dv * d
    out[valid] = val[valid]
    return out


def warp_perspective(tex: np.ndarray, H: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    Hinv = np.linalg.inv(H)
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    pts = np.column_stack(
        [xx.ravel().astype(np.float64), yy.ravel().astype(np.float64), np.ones(xx.size)]
    )
    src = (Hinv @ pts.T).T
    w = src[:, 2:3]
    uv = src[:, :2] / np.where(np.abs(w) < 1e-12, 1.0, w)
    sampled = bilinear_sample(tex, uv).reshape(img_h, img_w)

    mask = (
        (uv[:, 0] >= 0) & (uv[:, 0] < tex.shape[1])
        & (uv[:, 1] >= 0) & (uv[:, 1] < tex.shape[0])
    ).reshape(img_h, img_w)
    img = np.where(mask, sampled, 170.0)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def add_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via PIL — same kernel a real lens MTF would round off."""
    pil = Image.fromarray(img)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.uint8)


def add_sensor_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Light additive Gaussian noise to mimic photon / read noise."""
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


@dataclass
class WarpCase:
    name: str
    yaw: float
    pitch: float
    roll: float
    z: float
    blur_sigma: float
    noise_sigma: float


CASES = [
    WarpCase(name="moderate", yaw=30.0, pitch=20.0, roll=8.0,  z=900.0, blur_sigma=1.1, noise_sigma=2.0),
    WarpCase(name="strong",   yaw=55.0, pitch=40.0, roll=15.0, z=900.0, blur_sigma=1.3, noise_sigma=2.5),
]


def detect(img: np.ndarray) -> np.ndarray:
    cfg = cc.DetectorConfig.chess_multiscale()
    cfg.strategy.chess.refiner = cc.ChessRefiner.forstner()
    cfg.orientation_method = cc.OrientationMethod.DISK_FIT
    detector = cc.Detector(cfg)
    return detector.detect(np.ascontiguousarray(img))


def pick_corner(corners: np.ndarray, img_h: int, img_w: int, margin: int) -> np.ndarray | None:
    """Pick the corner with the most non-orthogonal axis pair, away from edges."""
    if corners.shape[0] == 0:
        return None

    def margin_ok(row: np.ndarray) -> bool:
        return margin < row[0] < img_w - margin and margin < row[1] < img_h - margin

    def non_ortho(row: np.ndarray) -> float:
        delta = float(row[7] - row[5]) - np.pi / 2.0
        wrapped = (delta + np.pi / 2.0) % np.pi - np.pi / 2.0
        return abs(wrapped)

    candidates = [row for row in corners if margin_ok(row)]
    if not candidates:
        return None
    candidates.sort(key=non_ortho, reverse=True)
    return candidates[0]


def draw_axis(ax, cx: float, cy: float, theta: float, length: float, **kwargs) -> None:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    ax.plot(
        [cx - length * c, cx + length * c],
        [cy - length * s, cy + length * s],
        **kwargs,
    )


def render_case(case: WarpCase, rng: np.random.Generator) -> None:
    tex = chessboard_texture()
    H = texture_to_image_homography(case.yaw, case.pitch, case.roll, case.z)
    img = warp_perspective(tex, H, IMG_W, IMG_H)
    img = add_blur(img, case.blur_sigma)
    img = add_sensor_noise(img, case.noise_sigma, rng)

    corners = detect(img)
    picked = pick_corner(corners, IMG_H, IMG_W, margin=CROP // 2 + 8)
    if picked is None:
        raise RuntimeError(f"no corner picked for case {case.name}")

    cx, cy = float(picked[0]), float(picked[1])
    a0, a1 = float(picked[5]), float(picked[7])

    # Integer crop window centred on the corner; axes get re-expressed
    # relative to the crop.
    x0 = int(round(cx)) - CROP // 2
    y0 = int(round(cy)) - CROP // 2
    crop = img[y0 : y0 + CROP, x0 : x0 + CROP]

    delta_deg = abs(np.degrees(((a1 - a0) - np.pi / 2.0 + np.pi / 2.0) % np.pi - np.pi / 2.0))

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(crop, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    axis_len = CROP * 0.42
    cxc, cyc = cx - x0, cy - y0
    draw_axis(ax, cxc, cyc, a0, axis_len, color=AXIS0_COLOR, linewidth=1.9)
    draw_axis(ax, cxc, cyc, a1, axis_len, color=AXIS1_COLOR, linewidth=1.9)
    ax.plot([cxc], [cyc], marker="o", color="#ffd24a", markersize=3.6, mew=0)

    ax.set_xlim(-0.5, CROP - 0.5)
    ax.set_ylim(CROP - 0.5, -0.5)

    out = BOOK_IMG / f"readme_warp_{case.name}.png"
    fig.tight_layout(pad=0)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"wrote {out}  (axis split {delta_deg:.1f}° from orthogonal)")


def main() -> None:
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=20260514)
    for case in CASES:
        render_case(case, rng)


if __name__ == "__main__":
    main()
