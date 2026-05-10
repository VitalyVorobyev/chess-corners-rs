"""Method-name → ChessConfig mapping for the orientation benchmark."""

from __future__ import annotations

from typing import Callable

__all__ = ["VARIANTS", "build_config", "list_methods"]


def _build_permissive_cfg():
    """Return a permissive but typed ChessConfig.

    Single-scale, lower threshold so single-corner 41x41 patches are
    still detected. Defaults are otherwise preserved.
    """
    import chess_corners  # local import: only needed when running detection

    cfg = chess_corners.ChessConfig()
    cfg.detector_mode = chess_corners.DetectorMode.BROAD
    cfg.threshold_mode = chess_corners.ThresholdMode.ABSOLUTE
    cfg.threshold_value = 0.05
    cfg.nms_radius = 2
    cfg.min_cluster_size = 1
    cfg.pyramid_levels = 1
    cfg.pyramid_min_size = 32
    cfg.refinement_radius = 3
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
