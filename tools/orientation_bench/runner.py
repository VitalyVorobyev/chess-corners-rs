"""Detector wrappers for the benchmark.

The chess_corners Python binding is imported lazily so the rest of the
package (config, gt, metrics) keeps working when the wheel is not
built.
"""

from __future__ import annotations

import numpy as np

from .variants import build_config

__all__ = [
    "DetectionRow",
    "detect_with_method",
    "detect_baseline",
    "ensure_available",
]


class DetectorUnavailable(RuntimeError):
    """Raised when ``chess_corners`` cannot be imported."""


def ensure_available() -> None:
    try:
        import chess_corners  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised via skip in tests
        raise DetectorUnavailable(
            "chess_corners is not importable. Build the wheel with:\n"
            "  maturin develop -m crates/chess-corners-py/pyproject.toml --release"
        ) from exc


class DetectionRow:
    """Light typed accessor for one row of the (n, 9) corners array."""

    __slots__ = (
        "x",
        "y",
        "response",
        "contrast",
        "fit_rms",
        "axis0_angle",
        "axis0_sigma",
        "axis1_angle",
        "axis1_sigma",
    )

    def __init__(self, row: np.ndarray) -> None:
        self.x = float(row[0])
        self.y = float(row[1])
        self.response = float(row[2])
        self.contrast = float(row[3])
        self.fit_rms = float(row[4])
        self.axis0_angle = float(row[5])
        self.axis0_sigma = float(row[6])
        self.axis1_angle = float(row[7])
        self.axis1_sigma = float(row[8])


def detect_with_method(img_u8: np.ndarray, method: str) -> np.ndarray:
    """Run chess_corners.find_chess_corners with the given method's config."""
    ensure_available()
    import chess_corners

    if method == "disk_sector_py":
        from .disk_sector import refine_detection_rows

        cfg = build_config("sigma_correction_lut")
        img = np.ascontiguousarray(img_u8, dtype=np.uint8)
        rows = chess_corners.find_chess_corners(img, cfg)
        return refine_detection_rows(img, rows)

    cfg = build_config(method)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    return chess_corners.find_chess_corners(img, cfg)


def detect_baseline(img_u8: np.ndarray) -> np.ndarray:
    """Convenience wrapper for the baseline method."""
    return detect_with_method(img_u8, "baseline")
