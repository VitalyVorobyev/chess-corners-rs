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


def build_chess_config() -> chess_corners.ChessConfig:
    cfg = chess_corners.ChessConfig()

    cfg.detector_mode = chess_corners.DetectorMode.BROAD
    cfg.descriptor_mode = chess_corners.DescriptorMode.CANONICAL
    cfg.threshold_mode = chess_corners.ThresholdMode.ABSOLUTE
    cfg.threshold_value = 0.5
    cfg.nms_radius = 3
    cfg.min_cluster_size = 1
    cfg.pyramid_levels = 3
    cfg.pyramid_min_size = 96
    cfg.refinement_radius = 4
    cfg.merge_radius = 2.5

    cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

    cfg.refiner.center_of_mass.radius = 2

    cfg.refiner.forstner.radius = 3
    cfg.refiner.forstner.min_trace = 20.0
    cfg.refiner.forstner.min_det = 0.001
    cfg.refiner.forstner.max_condition_number = 60.0
    cfg.refiner.forstner.max_offset = 2.0

    cfg.refiner.saddle_point.radius = 3
    cfg.refiner.saddle_point.det_margin = 0.002
    cfg.refiner.saddle_point.max_offset = 1.75
    cfg.refiner.saddle_point.min_abs_det = 0.0002

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
            f"response={strongest[2]:.3f}, orientation={strongest[3]:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
