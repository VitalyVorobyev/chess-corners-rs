"""Method-name → DetectorConfig mapping for the orientation benchmark."""

from __future__ import annotations

from typing import Callable

__all__ = ["VARIANTS", "build_config", "list_methods"]


def _build_permissive_cfg():
    """Return a permissive but typed DetectorConfig.

    Single-scale, lower threshold so single-corner 41x41 patches are
    still detected. ChESS strategy on the broad (r=10) ring; defaults
    are otherwise preserved.
    """
    import chess_corners  # local import: only needed when running detection

    cfg = chess_corners.DetectorConfig.single_scale()
    chess = chess_corners.ChessConfig()
    chess.ring = chess_corners.ChessRing.BROAD
    chess.nms_radius = 2
    chess.min_cluster_size = 1
    cfg.strategy = chess_corners.DetectionStrategy.from_chess(chess)
    cfg.threshold = chess_corners.Threshold.absolute(0.05)
    cfg.merge_radius = 1.5
    return cfg


def _build_ring_fit():
    """Ring-fit: Gauss-Newton on 16 ring samples with calibrated σ-LUT."""
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.RING_FIT
    return cfg


def _build_disk_fit():
    """Disk-fit: full-disk crossing-line estimator with ring-fit fallback."""
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.DISK_FIT
    return cfg


VARIANTS: dict[str, Callable[[], object]] = {
    "ring_fit": _build_ring_fit,
    "disk_fit": _build_disk_fit,
}


def list_methods() -> list[str]:
    return sorted(VARIANTS.keys())


def build_config(method: str):
    if method not in VARIANTS:
        raise KeyError(
            f"unknown method {method!r}; available: {sorted(VARIANTS.keys())}"
        )
    return VARIANTS[method]()
