"""Benchmark-only multi-radius polar sector orientation prototype.

This module deliberately lives in ``tools/orientation_bench`` rather
than the Rust crates. It tests a representation between the 16-sample
ring fit and the full Cartesian disk estimator: several polar traces
around the detected center vote for two projective line directions.

Coordinate convention: pixel centers are integer image coordinates.
The supplied detection row center ``(x, y)`` is in image pixels, and
polar angles use image coordinates with y increasing downward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "PolarFitDebug",
    "PolarFitParams",
    "polar_fit_refine_row",
    "refine_detection_rows",
]


PI = math.pi
TAU = 2.0 * math.pi


@dataclass(frozen=True)
class PolarFitParams:
    """Tuning values for the benchmark-only polar prototype."""

    radii: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    angle_bins: int = 64
    beta: float = 4.0
    seed_offsets_deg: tuple[float, ...] = (
        -24.0,
        -16.0,
        -8.0,
        -4.0,
        0.0,
        4.0,
        8.0,
        16.0,
        24.0,
    )
    global_grid_step_deg: float = 4.0
    min_valid_radii: int = 4
    min_corr: float = 0.55
    min_radial_consistency: float = 0.50
    weak_disagree_deg: float = 12.0
    inflated_sigma_deg: float = 10.0
    max_full_image_rows: int = 80
    radial_consistency_scale_deg: float = 8.0
    retained_pairs: int = 12


@dataclass(frozen=True)
class PolarFitDebug:
    """Diagnostic details for one row refinement."""

    accepted: bool
    reason: str
    valid_radii: int
    theta0: float
    theta1: float
    corr: float
    radial_consistency: float
    rms: float
    amp: float
    disagreement_deg: float


@dataclass(frozen=True)
class _PolarTrace:
    radius: float
    values_centered: np.ndarray
    energy: float


@dataclass(frozen=True)
class _PolarFit:
    theta0: float
    theta1: float
    corr: float
    amp: float
    rms: float


def _wrap_pi(a: float) -> float:
    out = math.fmod(a, PI)
    if out < 0.0:
        out += PI
    return out


def _line_delta(a: float, b: float) -> float:
    return abs(((a - b + PI / 2.0) % PI) - PI / 2.0)


def _pair_disagreement(a0: float, a1: float, b0: float, b1: float) -> float:
    direct = max(_line_delta(a0, b0), _line_delta(a1, b1))
    swapped = max(_line_delta(a0, b1), _line_delta(a1, b0))
    return min(direct, swapped)


def _line_separation(a0: float, a1: float) -> float:
    return _line_delta(a0, a1)


def _valid_pair(a0: float, a1: float) -> bool:
    sep = _line_separation(a0, a1)
    return math.radians(12.0) <= sep <= math.radians(89.5)


def _canonical_pair(a0: float, a1: float) -> tuple[float, float]:
    a0 = _wrap_pi(a0)
    a1 = _wrap_pi(a1)
    if a1 < a0:
        a0, a1 = a1, a0
    return a0, a1


def _phi_bins(params: PolarFitParams) -> np.ndarray:
    # Mid-bin samples avoid repeatedly landing exactly on model zero crossings.
    idx = np.arange(int(params.angle_bins), dtype=np.float64)
    return (idx + 0.5) * TAU / float(params.angle_bins)


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = xs - x0.astype(np.float64)
    ty = ys - y0.astype(np.float64)

    src = img.astype(np.float64, copy=False)
    i00 = src[y0, x0]
    i10 = src[y0, x1]
    i01 = src[y1, x0]
    i11 = src[y1, x1]
    a = i00 + (i10 - i00) * tx
    b = i01 + (i11 - i01) * tx
    return a + (b - a) * ty


def _extract_polar_traces(
    img_u8: np.ndarray,
    cx: float,
    cy: float,
    params: PolarFitParams,
) -> tuple[list[_PolarTrace], np.ndarray]:
    h, w = img_u8.shape
    phis = _phi_bins(params)
    cos_phi = np.cos(phis)
    sin_phi = np.sin(phis)
    traces: list[_PolarTrace] = []

    for radius in params.radii:
        r = float(radius)
        if (
            cx - r < 0.0
            or cy - r < 0.0
            or cx + r > (w - 1)
            or cy + r > (h - 1)
        ):
            continue

        vals = _bilinear_sample(img_u8, cx + r * cos_phi, cy + r * sin_phi)
        vals_centered = vals - float(vals.mean())
        energy = float(np.dot(vals_centered, vals_centered))
        if energy <= 1e-9 or not math.isfinite(energy):
            continue
        traces.append(
            _PolarTrace(
                radius=r,
                values_centered=vals_centered.astype(np.float64),
                energy=energy,
            )
        )

    return traces, phis


def _model_centered(
    phis: np.ndarray,
    theta0: float,
    theta1: float,
    beta: float,
) -> np.ndarray:
    q = np.tanh(beta * np.sin(phis - theta0)) * np.tanh(beta * np.sin(phis - theta1))
    return q - float(q.mean())


def _score_pair(
    traces: list[_PolarTrace],
    phis: np.ndarray,
    theta0: float,
    theta1: float,
    params: PolarFitParams,
) -> _PolarFit | None:
    if not _valid_pair(theta0, theta1):
        return None

    q = _model_centered(phis, theta0, theta1, params.beta)
    q_energy_one = float(np.dot(q, q))
    if q_energy_one <= 1e-9 or not math.isfinite(q_energy_one):
        return None

    num = 0.0
    val_energy = 0.0
    for trace in traces:
        num += float(np.dot(trace.values_centered, q))
        val_energy += trace.energy

    q_energy = q_energy_one * float(len(traces))
    if val_energy <= 1e-9:
        return None

    corr = abs(num) / math.sqrt(q_energy * val_energy)
    amp = num / q_energy
    ssr = 0.0
    for trace in traces:
        residual = trace.values_centered - amp * q
        ssr += float(np.dot(residual, residual))
    rms = math.sqrt(ssr / float(len(traces) * q.size))
    a0, a1 = _canonical_pair(theta0, theta1)
    return _PolarFit(
        theta0=a0,
        theta1=a1,
        corr=float(corr),
        amp=abs(float(amp)),
        rms=rms,
    )


def _candidate_angles(base0: float, base1: float, params: PolarFitParams) -> list[float]:
    candidates: list[float] = []
    for seed in (base0, base1):
        for off_deg in params.seed_offsets_deg:
            candidates.append(_wrap_pi(seed + math.radians(float(off_deg))))

    n_steps = int(round(180.0 / float(params.global_grid_step_deg)))
    for i in range(n_steps):
        candidates.append(math.radians(float(i) * float(params.global_grid_step_deg)))

    unique: list[float] = []
    for angle in candidates:
        angle = _wrap_pi(float(angle))
        if all(_line_delta(angle, existing) > math.radians(0.5) for existing in unique):
            unique.append(angle)
    return unique


def _rank_candidate_fits(
    traces: list[_PolarTrace],
    phis: np.ndarray,
    candidates: Iterable[float],
    params: PolarFitParams,
) -> list[_PolarFit]:
    angles = list(candidates)
    fits: list[_PolarFit] = []
    for i, a0 in enumerate(angles):
        for a1 in angles[i + 1 :]:
            fit = _score_pair(traces, phis, a0, a1, params)
            if fit is not None:
                fits.append(fit)
    fits.sort(key=lambda f: (-f.corr, f.rms, f.theta0, f.theta1))
    return fits


def _fit_is_better(a: _PolarFit, b: _PolarFit) -> bool:
    return a.corr > b.corr + 1e-12 or (
        abs(a.corr - b.corr) <= 1e-12
        and (a.rms, a.theta0, a.theta1) < (b.rms, b.theta0, b.theta1)
    )


def _refine_fit(
    traces: list[_PolarTrace],
    phis: np.ndarray,
    seed: _PolarFit,
    params: PolarFitParams,
) -> _PolarFit:
    best = seed
    for step_deg in (2.0, 1.0, 0.5, 0.25):
        step = math.radians(step_deg)
        for d0 in (-step, 0.0, step):
            for d1 in (-step, 0.0, step):
                if d0 == 0.0 and d1 == 0.0:
                    continue
                fit = _score_pair(
                    traces,
                    phis,
                    best.theta0 + d0,
                    best.theta1 + d1,
                    params,
                )
                if fit is not None and _fit_is_better(fit, best):
                    best = fit
    return best


def _radial_consistency(
    traces: list[_PolarTrace],
    phis: np.ndarray,
    global_fit: _PolarFit,
    retained: list[_PolarFit],
    params: PolarFitParams,
) -> float:
    pairs = [(global_fit.theta0, global_fit.theta1)]
    for fit in retained:
        pair = (fit.theta0, fit.theta1)
        if all(
            _pair_disagreement(pair[0], pair[1], p[0], p[1]) > 1e-12
            for p in pairs
        ):
            pairs.append(pair)

    disagreements: list[float] = []
    for trace in traces:
        best: _PolarFit | None = None
        for theta0, theta1 in pairs:
            fit = _score_pair([trace], phis, theta0, theta1, params)
            if fit is not None and (best is None or _fit_is_better(fit, best)):
                best = fit
        if best is not None:
            disagreements.append(
                _pair_disagreement(
                    best.theta0,
                    best.theta1,
                    global_fit.theta0,
                    global_fit.theta1,
                )
            )

    if not disagreements:
        return 0.0

    rms_rad = math.sqrt(float(np.mean(np.square(disagreements))))
    rms_deg = math.degrees(rms_rad)
    scale = max(float(params.radial_consistency_scale_deg), 1e-6)
    return math.exp(-((rms_deg / scale) ** 2))


def _best_polar_fit(
    traces: list[_PolarTrace],
    phis: np.ndarray,
    base0: float,
    base1: float,
    params: PolarFitParams,
) -> tuple[_PolarFit, float] | None:
    candidates = _candidate_angles(base0, base1, params)
    ranked = _rank_candidate_fits(traces, phis, candidates, params)
    if not ranked:
        return None

    best = _refine_fit(traces, phis, ranked[0], params)
    retained = ranked[: max(1, int(params.retained_pairs))]
    consistency = _radial_consistency(traces, phis, best, retained, params)
    return best, consistency


def polar_fit_refine_row(
    img_u8: np.ndarray,
    row: np.ndarray,
    params: PolarFitParams | None = None,
) -> tuple[np.ndarray, PolarFitDebug]:
    """Refine one detection row and return ``(row, debug)``.

    ``row`` follows the benchmark/Python binding schema:
    ``x, y, response, contrast, fit_rms, axis0_angle, axis0_sigma,
    axis1_angle, axis1_sigma``.
    """

    params = params or PolarFitParams()
    out = np.array(row, dtype=np.float32, copy=True)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    if img.ndim != 2:
        raise ValueError("polar_fit expects a 2D uint8 grayscale image")

    cx = float(row[0])
    cy = float(row[1])
    base0 = _wrap_pi(float(row[5]))
    base1 = _wrap_pi(float(row[7]))
    base_sigma0 = float(row[6])
    base_sigma1 = float(row[8])

    fallback = PolarFitDebug(
        accepted=False,
        reason="fallback",
        valid_radii=0,
        theta0=base0,
        theta1=base1,
        corr=0.0,
        radial_consistency=0.0,
        rms=float(row[4]),
        amp=float(row[3]),
        disagreement_deg=0.0,
    )

    traces, phis = _extract_polar_traces(img, cx, cy, params)
    if len(traces) < int(params.min_valid_radii):
        return out, PolarFitDebug(
            **{
                **fallback.__dict__,
                "reason": "insufficient_support",
                "valid_radii": len(traces),
            }
        )

    result = _best_polar_fit(traces, phis, base0, base1, params)
    if result is None:
        return out, PolarFitDebug(
            **{**fallback.__dict__, "reason": "no_valid_pair", "valid_radii": len(traces)}
        )

    fit, consistency = result
    disagreement = _pair_disagreement(fit.theta0, fit.theta1, base0, base1)
    accepted = fit.corr >= params.min_corr and consistency >= params.min_radial_consistency
    reason = "accepted" if accepted else "confidence_gate"

    if accepted:
        out[5] = np.float32(fit.theta0)
        out[7] = np.float32(fit.theta1)
    elif disagreement > math.radians(params.weak_disagree_deg):
        inflated = max(base_sigma0, base_sigma1, math.radians(params.inflated_sigma_deg))
        out[6] = np.float32(inflated)
        out[8] = np.float32(inflated)
        reason = "weak_disagreement"

    return out, PolarFitDebug(
        accepted=accepted,
        reason=reason,
        valid_radii=len(traces),
        theta0=fit.theta0,
        theta1=fit.theta1,
        corr=fit.corr,
        radial_consistency=consistency,
        rms=fit.rms,
        amp=fit.amp,
        disagreement_deg=math.degrees(disagreement),
    )


def refine_detection_rows(
    img_u8: np.ndarray,
    rows: np.ndarray,
    params: PolarFitParams | None = None,
) -> np.ndarray:
    """Apply the polar prototype to an ``(N, 9)`` detection array."""

    params = params or PolarFitParams()
    if rows.size == 0:
        return rows.astype(np.float32, copy=True)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    refined = np.asarray(rows, dtype=np.float32).copy()
    if img.shape[0] > 128 or img.shape[1] > 128:
        n = min(int(params.max_full_image_rows), refined.shape[0])
        row_indices = np.argsort(refined[:, 2], kind="stable")[-n:]
    else:
        row_indices = np.arange(refined.shape[0])
    for i in row_indices.tolist():
        refined[i], _ = polar_fit_refine_row(img, refined[i], params=params)
    return refined
