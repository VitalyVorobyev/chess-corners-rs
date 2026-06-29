"""Detector wrappers for the benchmark.

The chess_corners Python binding is imported lazily so the rest of the
package (config, gt, metrics) keeps working when the wheel is not
built.
"""

from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass
class _BridgeResult:
    """Detections-compatible result for Python-refiner paths.

    Exposes the same named fields as ``chess_corners.Detections``
    (``.xy``, ``.response``, ``.angles``, ``.sigmas``) so that
    ``__main__.py`` can consume both without branching.
    """

    xy: np.ndarray      # (N, 2) float32
    response: np.ndarray  # (N,) float32
    angles: np.ndarray | None   # (N, 2) float32 or None
    sigmas: np.ndarray | None   # (N, 2) float32 or None

    def __len__(self) -> int:
        return len(self.xy)


def _detections_to_refiner_array(det) -> np.ndarray:
    """Build an (N, 9) float32 array from a Detections object for the
    Python refiners (disk_sector_py, polar_fit_py).

    Column layout:
      0: x  1: y  2: response
      3: contrast (1.0 — not available from detect; keeps base_amp = 1.0)
      4: fit_rms (NaN — not available; gives conservative base_rel = inf)
      5: axis0_angle  6: axis0_sigma  7: axis1_angle  8: axis1_sigma
    """
    n = len(det)
    arr = np.zeros((n, 9), dtype=np.float32)
    if n == 0:
        return arr
    arr[:, 0] = det.xy[:, 0]
    arr[:, 1] = det.xy[:, 1]
    arr[:, 2] = det.response
    arr[:, 3] = 1.0
    arr[:, 4] = np.nan
    if det.angles is not None:
        arr[:, 5] = det.angles[:, 0]
        arr[:, 7] = det.angles[:, 1]
    if det.sigmas is not None:
        arr[:, 6] = det.sigmas[:, 0]
        arr[:, 8] = det.sigmas[:, 1]
    return arr


def _refiner_array_to_bridge(arr: np.ndarray) -> _BridgeResult:
    """Wrap an (N, 9) refiner output as a _BridgeResult."""
    if arr.size == 0 or arr.shape[0] == 0:
        return _BridgeResult(
            xy=np.zeros((0, 2), dtype=np.float32),
            response=np.zeros((0,), dtype=np.float32),
            angles=np.zeros((0, 2), dtype=np.float32),
            sigmas=np.zeros((0, 2), dtype=np.float32),
        )
    xy = arr[:, :2].copy()
    response = arr[:, 2].copy()
    angles = arr[:, [5, 7]].copy()
    sigmas = arr[:, [6, 8]].copy()
    return _BridgeResult(xy=xy, response=response, angles=angles, sigmas=sigmas)


def detect_with_method(img_u8: np.ndarray, method: str):
    """Run detection with the given method.

    Returns a ``chess_corners.Detections`` object (or a ``_BridgeResult``
    duck-typing it for Python-refiner paths). Both expose ``.xy``,
    ``.response``, ``.angles``, and ``.sigmas`` named array attributes
    and support ``len()``.
    """
    ensure_available()
    import chess_corners

    if method == "disk_sector_py":
        from .disk_sector import refine_detection_rows

        cfg = build_config("ring_fit")
        img = np.ascontiguousarray(img_u8, dtype=np.uint8)
        det = chess_corners.Detector(cfg).detect(img)
        rows = _detections_to_refiner_array(det)
        refined = refine_detection_rows(img, rows)
        return _refiner_array_to_bridge(refined)

    if method == "polar_fit_py":
        from .polar_fit import refine_detection_rows

        cfg = build_config("ring_fit")
        img = np.ascontiguousarray(img_u8, dtype=np.uint8)
        det = chess_corners.Detector(cfg).detect(img)
        rows = _detections_to_refiner_array(det)
        refined = refine_detection_rows(img, rows)
        return _refiner_array_to_bridge(refined)

    cfg = build_config(method)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    return chess_corners.Detector(cfg).detect(img)
