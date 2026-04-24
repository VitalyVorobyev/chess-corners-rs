#!/usr/bin/env python3
"""Render a labelled grid of benchmark fixture samples for Part V.

Produces two figures:

- ``book/src/img/bench/synth_grid.png``: 4×4 grid showing how the
  benchmark fixture looks at representative (cell, blur, noise)
  combinations. Red cross marks the true sub-pixel corner.
- ``book/src/img/bench/synth_modes.png``: side-by-side comparison of
  the two training render modes (hard-cells vs tanh saddle) at equal
  scale.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZE = 45
CENTER = 22.0
SUPER = 8


def render_aa_chessboard(cell: int, offset: tuple[float, float], dark: int = 30, bright: int = 230) -> np.ndarray:
    out = np.empty((SIZE, SIZE), dtype=np.uint8)
    ox, oy = offset
    c = float(cell)
    inv = 1.0 / (SUPER * SUPER)
    sx = (np.arange(SUPER, dtype=np.float32) + 0.5) / SUPER - 0.5
    sy = sx
    for y in range(SIZE):
        yf = y + sy
        cy = np.floor((yf - oy) / c).astype(np.int32)
        for x in range(SIZE):
            xf = x + sx
            cx = np.floor((xf - ox) / c).astype(np.int32)
            parity = (cx[None, :] + cy[:, None]) & 1
            acc = (parity * bright + (1 - parity) * dark).astype(np.float32)
            out[y, x] = np.clip(round(acc.sum() * inv), 0, 255)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img
    import cv2

    radius = max(1, int(math.ceil(3.0 * sigma)))
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)


def add_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    if sigma <= 0.0:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def render_tanh_patch(edge_softness: float = 0.15) -> np.ndarray:
    """A tanh-saddle corner the size of our benchmark patch — the
    distribution the v2 ML refiner was trained on."""
    coords = np.arange(SIZE, dtype=np.float32) - CENTER
    x_grid, y_grid = np.meshgrid(coords, coords)
    val = 0.5 + 0.5 * np.tanh(x_grid / edge_softness) * np.tanh(y_grid / edge_softness)
    # Scale into a 30–230 gray range for visual consistency with the
    # hard-cell fixture.
    return (30 + (230 - 30) * val).clip(0, 255).astype(np.uint8)


def make_fixture_grid(out_path: Path) -> None:
    offset = (CENTER + 0.35, CENTER + 0.8)
    true_corner = offset

    conditions = [
        ("cell=5", 5, 0.3, 0.0),
        ("cell=8", 8, 0.3, 0.0),
        ("blur σ=1.0", 8, 1.0, 0.0),
        ("blur σ=2.0", 8, 2.0, 0.0),
        ("noise σ=5", 8, 0.3, 5.0),
        ("noise σ=12", 8, 0.3, 12.0),
        ("low contrast", 8, 0.3, 0.0),  # dark=90, bright=165
        ("aggressive", 5, 2.0, 10.0),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    axes = axes.ravel()
    for ax, (label, cell, blur, noise) in zip(axes, conditions):
        if label == "low contrast":
            img = render_aa_chessboard(cell, offset, dark=90, bright=165)
        else:
            img = render_aa_chessboard(cell, offset)
        img = gaussian_blur(img, blur)
        img = add_noise(img, noise, seed=hash(label) & 0xFFFF)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.plot(
            [true_corner[0] - 2, true_corner[0] + 2],
            [true_corner[1], true_corner[1]],
            color="#e74c3c",
            linewidth=1.1,
        )
        ax.plot(
            [true_corner[0], true_corner[0]],
            [true_corner[1] - 2, true_corner[1] + 2],
            color="#e74c3c",
            linewidth=1.1,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Benchmark fixture: 45×45 AA chessboard patches at varying conditions",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"wrote {out_path}")


def make_mode_comparison(out_path: Path) -> None:
    hard = render_aa_chessboard(8, (CENTER, CENTER))
    tanh = render_tanh_patch(edge_softness=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, img, title in zip(
        axes,
        [hard, tanh],
        [
            "hard_cells (benchmark + v4 training)",
            "tanh (legacy v2 training distribution)",
        ],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Synthetic training distributions for the ML refiner",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    out_dir = Path("book/src/img/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_fixture_grid(out_dir / "synth_grid.png")
    make_mode_comparison(out_dir / "synth_modes.png")


if __name__ == "__main__":
    main()
