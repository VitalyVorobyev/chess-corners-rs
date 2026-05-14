#!/usr/bin/env python3
"""Render two chessboard images under projective warp with detector overlays.

Generates:

- ``book/src/img/readme_warp_moderate.png`` — synthetic chessboard under a
  moderate projective tilt (yaw 30°, pitch 20°).
- ``book/src/img/readme_warp_strong.png`` — same board under a much stronger
  tilt (yaw 55°, pitch 40°).

Each PNG shows the warped board with the per-corner two-axis orientation
measured by the ``DiskFit`` orientation method drawn on top. The axes are
**not** assumed orthogonal — under projective warp they tilt independently,
and the overlay shows the detector recovering that tilt.

Run from the workspace root, after building the wheel into ``.venv``::

    (cd crates/chess-corners-py && maturin develop --release)
    .venv/bin/python tools/render_readme_warp_overlays.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import chess_corners as cc

WORKSPACE = Path(__file__).resolve().parents[1]
BOOK_IMG = WORKSPACE / "book" / "src" / "img"

# Texture: 8x6 inner-corner chessboard rendered at 60 pixels per square.
INNER_X, INNER_Y = 9, 6
SQUARE_PX = 60
TEX_W = (INNER_X + 1) * SQUARE_PX
TEX_H = (INNER_Y + 1) * SQUARE_PX

IMG_W, IMG_H = 1024, 720
FOCAL = 1100.0
PRINCIPAL = (IMG_W / 2.0, IMG_H / 2.0)

AXIS0_COLOR = "#ff5a5a"
AXIS1_COLOR = "#37c46a"


def chessboard_texture(invert: bool = False) -> np.ndarray:
    """Black/white chessboard, no anti-aliasing."""
    tex = np.zeros((TEX_H, TEX_W), dtype=np.uint8)
    for iy in range(INNER_Y + 1):
        for ix in range(INNER_X + 1):
            parity = (ix + iy) % 2
            if parity == (0 if invert else 1):
                y0, y1 = iy * SQUARE_PX, (iy + 1) * SQUARE_PX
                x0, x1 = ix * SQUARE_PX, (ix + 1) * SQUARE_PX
                tex[y0:y1, x0:x1] = 235
    # Slight border padding stays at 0 (creates a dark frame visually).
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

    The texture is placed on the plane Z = 0 with origin at its top-left,
    one texture pixel = one world unit. The camera looks down the +Z axis.
    """
    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)
    t = np.array([-TEX_W / 2.0, -TEX_H / 2.0, z], dtype=np.float64)
    K = np.array(
        [[FOCAL, 0, PRINCIPAL[0]], [0, FOCAL, PRINCIPAL[1]], [0, 0, 1]], dtype=np.float64
    )

    # World->camera columns for plane Z=0 keep R[:, 0] and R[:, 1].
    M = np.column_stack([R[:, 0], R[:, 1], t])  # 3x3, mapping (u, v, 1) -> camera.
    # World point on plane: (u + t.x_offset, v + t.y_offset, 0). The t column
    # already encodes the texture-center recentering above.
    H = K @ M
    return H


def bilinear_sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample texture at floating uv coordinates.

    uv: shape (N, 2), columns are (u, v) in texture pixel coordinates.
    Out-of-bounds reads return 0 (a dark surround).
    """
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
    """Render `tex` into an img_w x img_h canvas under homography H (tex -> img)."""
    Hinv = np.linalg.inv(H)
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    pts = np.column_stack(
        [xx.ravel().astype(np.float64), yy.ravel().astype(np.float64), np.ones(xx.size)]
    )
    src = (Hinv @ pts.T).T
    w = src[:, 2:3]
    uv = src[:, :2] / np.where(np.abs(w) < 1e-12, 1.0, w)
    sampled = bilinear_sample(tex, uv).reshape(img_h, img_w)

    # Soft mid-gray background for the masked-out region (was 0).
    mask = (
        (uv[:, 0] >= 0) & (uv[:, 0] < tex.shape[1])
        & (uv[:, 1] >= 0) & (uv[:, 1] < tex.shape[0])
    ).reshape(img_h, img_w)
    img = np.where(mask, sampled, 160.0)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


@dataclass
class WarpCase:
    name: str
    yaw: float
    pitch: float
    roll: float
    z: float
    title: str


CASES = [
    WarpCase(
        name="moderate",
        yaw=30.0, pitch=20.0, roll=8.0, z=900.0,
        title="Moderate projective tilt (yaw 30°, pitch 20°)",
    ),
    WarpCase(
        name="strong",
        yaw=55.0, pitch=40.0, roll=15.0, z=900.0,
        title="Strong projective tilt (yaw 55°, pitch 40°)",
    ),
]


def detect(img: np.ndarray) -> np.ndarray:
    cfg = cc.DetectorConfig.chess_multiscale()
    cfg.strategy.chess.refiner = cc.ChessRefiner.forstner()
    cfg.orientation_method = cc.OrientationMethod.DISK_FIT
    detector = cc.Detector(cfg)
    return detector.detect(np.ascontiguousarray(img))


def draw_axis(ax, cx: float, cy: float, theta: float, length: float, **kwargs) -> None:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    ax.plot(
        [cx - length * c, cx + length * c],
        [cy - length * s, cy + length * s],
        **kwargs,
    )


def render_case(case: WarpCase) -> None:
    tex = chessboard_texture()
    H = texture_to_image_homography(case.yaw, case.pitch, case.roll, case.z)
    img = warp_perspective(tex, H, IMG_W, IMG_H)
    corners = detect(img)

    # Axis length scales with corner spacing in the image plane. Estimate it
    # from the median nearest-neighbour distance, capped at ~½ a square.
    if len(corners) >= 2:
        xy = corners[:, :2]
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        d[d == 0] = np.inf
        nn = np.partition(d, 1, axis=1)[:, 1]
        axis_len = float(np.median(nn)) * 0.42
    else:
        axis_len = 12.0

    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])

    for row in corners:
        cx, cy, a0, a1 = float(row[0]), float(row[1]), float(row[5]), float(row[7])
        draw_axis(ax, cx, cy, a0, axis_len, color=AXIS0_COLOR, linewidth=1.4)
        draw_axis(ax, cx, cy, a1, axis_len, color=AXIS1_COLOR, linewidth=1.4)
        ax.plot([cx], [cy], marker="o", color="#ffd24a", markersize=2.4, mew=0)

    n_corners = len(corners)
    ax.set_title(
        f"{case.title} — {n_corners} corners, DiskFit two-axis fit",
        fontsize=10,
    )
    ax.set_xlim(-0.5, IMG_W - 0.5)
    ax.set_ylim(IMG_H - 0.5, -0.5)

    out = BOOK_IMG / f"readme_warp_{case.name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_corners} corners)")


def main() -> None:
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        render_case(case)


if __name__ == "__main__":
    main()
