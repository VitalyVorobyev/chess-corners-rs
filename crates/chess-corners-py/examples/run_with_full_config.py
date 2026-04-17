#!/usr/bin/env python3
"""Run the chess_corners detector with a complete canonical JSON config."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess_corners
import numpy as np


class ConfigError(ValueError):
    """Raised when the example JSON configuration is invalid."""


def load_grayscale_image(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            array = np.asarray(gray, dtype=np.uint8)
    except OSError as exc:
        raise ConfigError(f"failed to read image {path}: {exc}") from exc

    return np.ascontiguousarray(array)


def build_chess_config(data: object) -> chess_corners.ChessConfig:
    try:
        return chess_corners.ChessConfig.from_dict(data)
    except chess_corners.ConfigError as exc:
        raise ConfigError(str(exc)) from exc


def load_chess_config(path: Path) -> chess_corners.ChessConfig:
    try:
        return chess_corners.ChessConfig.from_json_file(path)
    except chess_corners.ConfigError as exc:
        raise ConfigError(str(exc)) from exc


def _strongest_corner(corners: np.ndarray) -> np.ndarray | None:
    if corners.shape[0] == 0:
        return None
    return corners[int(np.argmax(corners[:, 2]))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chess_corners on a user-provided image using a complete canonical "
            "JSON detector configuration."
        )
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("config", type=Path, help="Path to detector config JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_chess_config(args.config)
    image = load_grayscale_image(args.image)
    corners = chess_corners.find_chess_corners(image, cfg)

    print(f"image: {args.image}")
    print(f"image shape: height={image.shape[0]}, width={image.shape[1]}")
    print("effective config:")
    print(cfg.pretty())
    print(f"detected corners: {corners.shape[0]}")

    strongest = _strongest_corner(corners)
    if strongest is not None:
        print(
            "strongest corner: "
            f"x={strongest[0]:.3f}, y={strongest[1]:.3f}, "
            f"response={strongest[2]:.3f}, contrast={strongest[3]:.3f}, "
            f"fit_rms={strongest[4]:.3f}, "
            f"axis0=({strongest[5]:.4f}±{strongest[6]:.4f}), "
            f"axis1=({strongest[7]:.4f}±{strongest[8]:.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
