"""Detector wrappers for the benchmark.

The chess_corners Python binding is imported lazily so the rest of the
package (config, gt, metrics) keeps working when the wheel is not
built.
"""

from __future__ import annotations

import numpy as np

from .variants import build_config

__all__ = [
    "detect_with_method",
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


def detect_with_method(img_u8: np.ndarray, method: str):
    """Run detection with the given method; returns ``chess_corners.Detections``
    (``.xy``, ``.response``, ``.angles``, ``.sigmas``; supports ``len()``)."""
    ensure_available()
    import chess_corners

    cfg = build_config(method)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    return chess_corners.Detector(cfg).detect(img)
