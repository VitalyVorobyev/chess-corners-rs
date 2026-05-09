"""Method-name → ChessConfig mapping.

Phase 1 only shipped ``"baseline"``. Phase 3 variants register here
once the corresponding ``OrientationMethod`` value is plumbed through
to ``ChessConfig``. Benchmark-only Python post-processors can also
register a base config here; ``runner.py`` owns their extra work.
"""

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


def _build_baseline():
    """Baseline 2nd-harmonic-seeded Gauss-Newton fit (legacy default)."""
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.BASELINE
    return cfg


def _build_sigma_correction_lut():
    """V6b — baseline fit with a fit_rms-keyed σ-multiplier LUT applied."""
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.SIGMA_CORRECTION_LUT
    return cfg


def _build_adaptive_beta():
    """V1 — promote the tanh slope β to a 5th GN parameter so the fit
    adapts to the local edge sharpness across the blur sweep.
    """
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.ADAPTIVE_BETA
    return cfg


def _build_disk_sector_py():
    """Benchmark-only full-disk post-processor.

    The runner first executes the sigma-LUT baseline to obtain centers
    and calibrated fallback sigmas, then recomputes axes in Python.
    Returning the sigma-LUT config here keeps config construction
    available to generic callers while preserving the no-public-API
    prototype boundary.
    """

    return _build_sigma_correction_lut()


def _build_disk_sector_rust():
    """Rust full-disk estimator with sigma-LUT fallback."""
    import chess_corners

    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.FULL_DISK_SECTOR
    return cfg


VARIANTS: dict[str, Callable[[], object]] = {
    "baseline": _build_baseline,
    "sigma_correction_lut": _build_sigma_correction_lut,
    "adaptive_beta": _build_adaptive_beta,
    "disk_sector_py": _build_disk_sector_py,
    "disk_sector_rust": _build_disk_sector_rust,
}


def list_methods() -> list[str]:
    return sorted(VARIANTS.keys())


def build_config(method: str):
    if method not in VARIANTS:
        raise KeyError(
            f"unknown method {method!r}; available: {sorted(VARIANTS.keys())}"
        )
    return VARIANTS[method]()
