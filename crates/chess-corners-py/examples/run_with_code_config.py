#!/usr/bin/env python3
"""Run the chess_corners detector with a complete config defined in code."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess_corners
import numpy as np


class ConfigError(ValueError):
    """Raised when the example image cannot be loaded."""


def load_grayscale_image(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            array = np.asarray(gray, dtype=np.uint8)
    except OSError as exc:
        raise ConfigError(f"failed to read image {path}: {exc}") from exc

    return np.ascontiguousarray(array)


def build_chess_config() -> chess_corners.DetectorConfig:
    cfg = chess_corners.DetectorConfig.chess_multiscale()

    # Strategy-specific knobs live inside ChessConfig.
    chess = cfg.strategy.chess
    chess.ring = chess_corners.ChessRing.BROAD
    cfg.detection.nms_radius = 3
    cfg.detection.min_cluster_size = 1

    fcfg = chess_corners.ForstnerConfig()
    fcfg.radius = 3
    fcfg.min_trace = 20.0
    fcfg.min_det = 0.001
    fcfg.max_condition_number = 60.0
    fcfg.max_offset = 2.0
    chess.refiner = chess_corners.ChessRefiner.forstner(fcfg)

    cfg.strategy = chess_corners.DetectionStrategy.from_chess(chess)
    cfg.threshold = chess_corners.Threshold.absolute(0.5)
    cfg.merge_radius = 2.5
    cfg.multiscale = chess_corners.MultiscaleConfig.pyramid(
        levels=3, min_size=96, refinement_radius=4
    )

    return cfg


def _strongest_corner(corners: np.ndarray) -> np.ndarray | None:
    if corners.shape[0] == 0:
        return None
    return corners[int(np.argmax(corners[:, 2]))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chess_corners on a user-provided image using a complete detector "
            "configuration defined directly in Python code."
        )
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_chess_config()
    image = load_grayscale_image(args.image)
    corners = chess_corners.Detector(cfg).detect(image)

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
            f"response={strongest[2]:.3f}, "
            f"axis0=({strongest[3]:.4f}+/-{strongest[4]:.4f}), "
            f"axis1=({strongest[5]:.4f}+/-{strongest[6]:.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
