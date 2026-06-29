#!/usr/bin/env python3
"""Render synthetic-patch overlays for the orientation methods chapter.

Outputs three PNGs to ``book/src/img/`` showing how `RingFit` and
`DiskFit` behave on three illustrative cases:

- ``orientation_clean.png`` — clean orthogonal corner; both methods agree
  (DiskFit's lazy gate short-circuits to the ring fit).
- ``orientation_skew.png`` — extreme projective skew (30°/150°); the
  ring fit collapses, the disk fit recovers from full-area evidence.
- ``orientation_sharp.png`` — sharp edges (low blur); the ring fit's
  fixed `β = 4` mismatches the transition, the disk fit's per-width
  sweep picks `w = 0.35` and recovers cleanly.

Run from the workspace root after building the wheel into a venv::

    uv venv .venv-overlay
    source .venv-overlay/bin/activate
    uv pip install maturin numpy matplotlib pillow
    (cd crates/chess-corners-py && maturin develop --release)
    python tools/render_orientation_overlays.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import chess_corners as cc

WORKSPACE = Path(__file__).resolve().parents[1]
BOOK_IMG = WORKSPACE / "book" / "src" / "img"

PATCH_SIZE = 41
PATCH_CENTER = PATCH_SIZE / 2

GT_COLOR = "#ffffff"
RING_COLOR = "#ff5a5a"
DISK_COLOR = "#37c46a"


def synthetic_corner(theta0: float, theta1: float, width: float) -> np.ndarray:
    """Render a chessboard corner patch with two crossing transition lines."""
    yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE].astype(np.float32)
    dx = xx - PATCH_CENTER
    dy = yy - PATCH_CENTER
    d0 = -np.sin(theta0) * dx + np.cos(theta0) * dy
    d1 = -np.sin(theta1) * dx + np.cos(theta1) * dy
    q = np.tanh(d0 / width) * np.tanh(d1 / width)
    return np.clip(128.0 + 80.0 * q, 0.0, 255.0).astype(np.uint8)


def make_config(method) -> "cc.DetectorConfig":
    cfg = cc.DetectorConfig.single_scale()
    chess = cc.ChessConfig()
    chess.ring = cc.ChessRing.BROAD
    cfg.strategy = cc.DetectionStrategy.from_chess(chess)
    cfg.detection.nms_radius = 2
    cfg.detection.min_cluster_size = 1
    cfg.threshold = 0.05
    cfg.merge_radius = 1.5
    cfg.orientation_method = method
    return cfg


def detect(patch: np.ndarray, method) -> np.ndarray | None:
    """Return the corner row closest to the patch center, or None."""
    cfg = make_config(method)
    detector = cc.Detector(cfg)
    corners = detector.detect(np.ascontiguousarray(patch))
    if len(corners) == 0:
        return None
    cx0, cy0 = PATCH_CENTER, PATCH_CENTER
    dists = (corners[:, 0] - cx0) ** 2 + (corners[:, 1] - cy0) ** 2
    return corners[int(np.argmin(dists))]


def err_deg(theta_fit: float, theta_gt: float) -> float:
    """Undirected angular error in degrees (lines are mod π)."""
    delta = (theta_fit - theta_gt + np.pi / 2) % np.pi - np.pi / 2
    return float(np.abs(np.degrees(delta)))


def best_pair_err(fit: np.ndarray, gt0: float, gt1: float) -> float:
    a0, a1 = float(fit[5]), float(fit[7])
    direct = max(err_deg(a0, gt0), err_deg(a1, gt1))
    swapped = max(err_deg(a0, gt1), err_deg(a1, gt0))
    return min(direct, swapped)


def draw_line(ax, cx: float, cy: float, theta: float, length: float, **kwargs):
    c, s = float(np.cos(theta)), float(np.sin(theta))
    ax.plot(
        [cx - length * c, cx + length * c],
        [cy - length * s, cy + length * s],
        **kwargs,
    )


def render_case(
    name: str,
    theta0_deg: float,
    theta1_deg: float,
    width: float,
    title: str,
) -> None:
    theta0, theta1 = np.deg2rad(theta0_deg), np.deg2rad(theta1_deg)
    patch = synthetic_corner(theta0, theta1, width)

    ring = detect(patch, cc.OrientationMethod.RING_FIT)
    disk = detect(patch, cc.OrientationMethod.DISK_FIT)

    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    ax.imshow(patch, cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    line_len = PATCH_SIZE * 0.46

    # Ground truth (dashed white)
    draw_line(ax, PATCH_CENTER, PATCH_CENTER, theta0, line_len,
              color=GT_COLOR, linestyle="--", linewidth=1.3, label="ground truth")
    draw_line(ax, PATCH_CENTER, PATCH_CENTER, theta1, line_len,
              color=GT_COLOR, linestyle="--", linewidth=1.3)

    # RingFit (red)
    if ring is not None:
        rx, ry = float(ring[0]), float(ring[1])
        draw_line(ax, rx, ry, float(ring[5]), line_len,
                  color=RING_COLOR, linewidth=1.6, label="RingFit")
        draw_line(ax, rx, ry, float(ring[7]), line_len,
                  color=RING_COLOR, linewidth=1.6)

    # DiskFit (green)
    if disk is not None:
        dx_, dy_ = float(disk[0]), float(disk[1])
        draw_line(ax, dx_, dy_, float(disk[5]), line_len,
                  color=DISK_COLOR, linewidth=1.6, label="DiskFit")
        draw_line(ax, dx_, dy_, float(disk[7]), line_len,
                  color=DISK_COLOR, linewidth=1.6)

    # Per-method axis-pair error
    parts = [title]
    if ring is not None:
        parts.append(f"RingFit error: {best_pair_err(ring, theta0, theta1):.1f}°")
    if disk is not None:
        parts.append(f"DiskFit error: {best_pair_err(disk, theta0, theta1):.1f}°")
    ax.set_title("\n".join(parts), fontsize=9)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_xlim(-0.5, PATCH_SIZE - 0.5)
    ax.set_ylim(PATCH_SIZE - 0.5, -0.5)

    out = BOOK_IMG / f"orientation_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    BOOK_IMG.mkdir(parents=True, exist_ok=True)

    render_case(
        "clean",
        theta0_deg=0.0, theta1_deg=90.0, width=1.0,
        title="Clean orthogonal corner",
    )
    render_case(
        "skew",
        theta0_deg=30.0, theta1_deg=60.0, width=0.7,
        title="Narrow projective skew (axes 30° apart)",
    )
    render_case(
        "sharp",
        theta0_deg=25.0, theta1_deg=115.0, width=0.35,
        title="Sharp orthogonal corner (low blur)",
    )


if __name__ == "__main__":
    main()
