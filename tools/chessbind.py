"""Test script for python binding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import chess_corners


def _to_grayscale_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        rgb = image[..., :3]
        gray = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)

    gray_f = gray.astype(np.float32)
    if gray_f.max() <= 1.0:
        gray_f *= 255.0
    gray_f = np.clip(gray_f, 0.0, 255.0)
    return np.ascontiguousarray(gray_f.astype(np.uint8))


def _plot_overlay(
    image_u8: np.ndarray,
    corners: np.ndarray,
    show_orientation: bool
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image_u8, cmap="gray")

    if corners.size == 0:
        ax.set_title("No corners detected")
        ax.set_axis_off()
        return fig

    pts = corners
    xs = pts[:, 0]
    ys = pts[:, 1]
    responses = pts[:, 2]

    if responses.size:
        r_min = float(responses.min())
        r_max = float(responses.max())
        denom = r_max - r_min
        if denom <= 1e-12:
            t = np.zeros_like(responses)
        else:
            t = (responses - r_min) / denom
        colors = plt.get_cmap("viridis")(t)
    else:
        colors = "#1d3557"

    ax.scatter(
        xs,
        ys,
        s=18,
        facecolors="none",
        edgecolors=colors,
        linewidths=0.7,
        label="chess-corners",
    )

    if show_orientation:
        arrow_len = 12.0
        us = np.cos(pts[:, 3]) * arrow_len
        vs = np.sin(pts[:, 3]) * arrow_len
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors,
            width=0.002,
        )

    ax.set_title(f"ChESS corners: {len(corners)} detected")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.6)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chess-corners PyO3 binding on an image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--out", type=Path, help="Optional path to save the overlay.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure.")
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Disable orientation arrows in the overlay.",
    )
    args = parser.parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image = mpimg.imread(args.image)
    gray_u8 = _to_grayscale_u8(image)

    cfg = chess_corners.ChessConfig()
    cfg.pyramid_num_levels = 1
    print(cfg.to_dict())
    corners = chess_corners.find_chess_corners(gray_u8, cfg)
    fig = _plot_overlay(
        gray_u8,
        corners,
        show_orientation=not args.no_orientation
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0)
        print(f"Saved overlay to {args.out}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
