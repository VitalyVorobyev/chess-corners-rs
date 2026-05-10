"""Per-cell aggregation metrics for the orientation benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CellArrays",
    "aggregate_cell",
    "round_floats",
]


@dataclass
class CellArrays:
    """Per-cell raw arrays. All angle quantities in radians."""

    delta0: np.ndarray
    delta1: np.ndarray
    sigma0: np.ndarray
    sigma1: np.ndarray
    fit_rms: np.ndarray
    axis_swap_flag: np.ndarray
    degenerate_flag: np.ndarray
    matched: int
    gt_total: int


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _to_deg(x: float | np.floating) -> float:
    return float(x) * 180.0 / math.pi


def _finite(delta: np.ndarray) -> np.ndarray:
    """Return the subarray of finite (non-NaN, non-inf) entries."""
    if delta.size == 0:
        return delta
    return delta[np.isfinite(delta)]


def _bias(delta: np.ndarray) -> float:
    finite = _finite(delta)
    return _to_deg(np.mean(finite)) if finite.size else float("nan")


def _rmse(delta: np.ndarray) -> float:
    finite = _finite(delta)
    return (
        _to_deg(math.sqrt(float(np.mean(finite * finite)))) if finite.size else float("nan")
    )


def _mae(delta: np.ndarray) -> float:
    finite = _finite(delta)
    return _to_deg(float(np.median(np.abs(finite)))) if finite.size else float("nan")


def _z_stats(delta: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float, int]:
    if delta.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    finite = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0.0) & (sigma < math.pi)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan"), 0
    z = delta[finite] / sigma[finite]
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    z_outlier_frac = float(np.mean(np.abs(z) > 2.0))
    return z_mean, z_std, z_outlier_frac, int(finite.sum())


def aggregate_cell(arrays: CellArrays) -> dict:
    """Aggregate per-corner arrays into a JSON-serialisable cell dict.

    ``failure_pct`` reports the fraction of corners whose two-axis fit was
    flagged as degenerate (non-finite or non-positive sigma). It does **not**
    include axis-swap events: the detector's `canonicalize` step picks the
    axis ordering by sector polarity, while the GT axes are emitted in a
    fixed ``(theta, theta+pi/2)`` ordering. Pairing through the swap branch
    is therefore a labeling-convention difference, not a fit failure. The
    swap rate is still reported separately as ``axis_swap_frac`` for
    informational use.
    """
    d0 = np.asarray(arrays.delta0, dtype=np.float64)
    d1 = np.asarray(arrays.delta1, dtype=np.float64)
    s0 = np.asarray(arrays.sigma0, dtype=np.float64)
    s1 = np.asarray(arrays.sigma1, dtype=np.float64)
    fit = np.asarray(arrays.fit_rms, dtype=np.float64)
    swap = np.asarray(arrays.axis_swap_flag, dtype=bool)
    degen = np.asarray(arrays.degenerate_flag, dtype=bool)
    matched = int(arrays.matched)
    gt_total = int(arrays.gt_total)

    z0_mean, z0_std, z0_out, z0_n = _z_stats(d0, s0)
    z1_mean, z1_std, z1_out, z1_n = _z_stats(d1, s1)

    failure_pct = float(np.mean(degen)) if degen.size else 0.0

    detection_pct = float(matched) / float(gt_total) if gt_total > 0 else 0.0

    abs_d0 = np.abs(d0)
    abs_d1 = np.abs(d1)
    # Drop NaN pairs from correlation inputs (parallel masking).
    mask0 = np.isfinite(fit) & np.isfinite(abs_d0)
    mask1 = np.isfinite(fit) & np.isfinite(abs_d1)
    corr0 = _safe_pearson(fit[mask0], abs_d0[mask0])
    corr1 = _safe_pearson(fit[mask1], abs_d1[mask1])

    return {
        "n_samples": int(d0.size),
        "matched": matched,
        "gt_total": gt_total,
        "detection_pct": detection_pct,
        "failure_pct": failure_pct,
        "bias_axis0_deg": _bias(d0),
        "bias_axis1_deg": _bias(d1),
        "rmse_axis0_deg": _rmse(d0),
        "rmse_axis1_deg": _rmse(d1),
        "mae_axis0_deg": _mae(d0),
        "mae_axis1_deg": _mae(d1),
        "z_mean_axis0": z0_mean,
        "z_std_axis0": z0_std,
        "z_outlier_frac_axis0": z0_out,
        "z_n_axis0": z0_n,
        "z_mean_axis1": z1_mean,
        "z_std_axis1": z1_std,
        "z_outlier_frac_axis1": z1_out,
        "z_n_axis1": z1_n,
        "corr_fitrms_abserr_axis0": corr0,
        "corr_fitrms_abserr_axis1": corr1,
        "axis_swap_frac": float(swap.mean()) if swap.size else 0.0,
        "degenerate_frac": float(degen.mean()) if degen.size else 0.0,
    }


def _round_value(value, digits: int = 6):
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        return round(value, digits)
    if isinstance(value, dict):
        return round_floats(value, digits)
    if isinstance(value, list):
        return [_round_value(v, digits) for v in value]
    return value


def round_floats(obj: dict, digits: int = 6) -> dict:
    return {k: _round_value(v, digits) for k, v in obj.items()}
