"""Benchmark-only full-disk two-axis orientation prototype.

This module deliberately lives in ``tools/orientation_bench`` rather
than the Rust crates. It is an experimental local-orientation estimator
used to test whether image evidence inside the descriptor disk can beat
the legacy 16-ring fit before any public API is added.

Coordinate convention: pixel centers are integer image coordinates.
The supplied detection row center ``(x, y)`` is in image pixels, and
all signed distances are measured in that same frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "DiskSectorDebug",
    "DiskSectorParams",
    "disk_sector_refine_row",
    "refine_detection_rows",
]


PI = math.pi
WIDTHS_PX = (0.35, 0.70, 1.40, 2.80)


@dataclass(frozen=True)
class DiskSectorParams:
    """Tuning values for the benchmark-only prototype."""

    support_radius: float = 8.0
    inner_radius: float = 1.0
    min_support: int = 64
    min_contrast: float = 10.0
    min_corr: float = 0.74
    accept_rel_margin: float = 0.03
    accept_rel_ratio: float = 0.92
    weak_disagree_deg: float = 12.0
    inflated_sigma_deg: float = 10.0
    max_full_image_rows: int = 80


@dataclass(frozen=True)
class DiskSectorDebug:
    """Diagnostic details for one row refinement."""

    accepted: bool
    reason: str
    support: int
    theta0: float
    theta1: float
    sigma: float
    rms: float
    amp: float
    rel_rms: float
    base_rel_rms: float
    corr: float
    edge_score: float
    width: float
    separation_deg: float
    disagreement_deg: float


@dataclass(frozen=True)
class _DiskData:
    xs: np.ndarray
    ys: np.ndarray
    vals: np.ndarray
    vals_centered: np.ndarray
    val_energy: float
    grad_angles: np.ndarray
    grad_weights: np.ndarray


@dataclass(frozen=True)
class _Fit:
    theta0: float
    theta1: float
    width: float
    mu: float
    amp: float
    rms: float
    rel_rms: float
    corr: float
    edge_score: float

    @property
    def objective(self) -> float:
        # The residual alone can prefer the wrong acute crossing on
        # sheared finite patches. The gradient term anchors the score
        # to visible edge-line evidence while the residual still rejects
        # noisy histogram coincidences.
        return self.rel_rms - 1.25 * self.edge_score


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
    # Preserve an ordered line pair in [0, pi). The benchmark scorer
    # pairs axes unordered/mod-pi, so polarity is not encoded here.
    if a1 < a0:
        a0, a1 = a1, a0
    return a0, a1


def _sobel_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(img, dtype=np.float32)
    pad = np.pad(src, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -pad[:-2, :-2]
        + pad[:-2, 2:]
        - 2.0 * pad[1:-1, :-2]
        + 2.0 * pad[1:-1, 2:]
        - pad[2:, :-2]
        + pad[2:, 2:]
    )
    gy = (
        -pad[:-2, :-2]
        - 2.0 * pad[:-2, 1:-1]
        - pad[:-2, 2:]
        + pad[2:, :-2]
        + 2.0 * pad[2:, 1:-1]
        + pad[2:, 2:]
    )
    return gx.astype(np.float32), gy.astype(np.float32)


def _extract_disk(
    img: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    cx: float,
    cy: float,
    params: DiskSectorParams,
) -> _DiskData | None:
    h, w = img.shape
    r = float(params.support_radius)
    x0 = int(math.floor(cx - r))
    x1 = int(math.ceil(cx + r))
    y0 = int(math.floor(cy - r))
    y1 = int(math.ceil(cy + r))
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    xs = xx.astype(np.float64) - float(cx)
    ys = yy.astype(np.float64) - float(cy)
    rr = np.hypot(xs, ys)
    mask = (rr <= r) & (rr >= float(params.inner_radius))
    if int(mask.sum()) < params.min_support:
        return None

    vals = img[y0 : y1 + 1, x0 : x1 + 1][mask].astype(np.float64)
    vals_centered = vals - float(vals.mean())
    val_energy = float(np.dot(vals_centered, vals_centered))
    gxx = gx[y0 : y1 + 1, x0 : x1 + 1][mask].astype(np.float64)
    gyy = gy[y0 : y1 + 1, x0 : x1 + 1][mask].astype(np.float64)
    weights = np.hypot(gxx, gyy)
    # Gradient normal + pi/2 gives the local edge/line direction.
    grad_angles = np.mod(np.arctan2(gyy, gxx) + PI / 2.0, PI)

    return _DiskData(
        xs=xs[mask].astype(np.float64),
        ys=ys[mask].astype(np.float64),
        vals=vals,
        vals_centered=vals_centered,
        val_energy=val_energy,
        grad_angles=grad_angles,
        grad_weights=weights,
    )


def _histogram_candidates(data: _DiskData, base0: float, base1: float) -> list[float]:
    candidates: list[float] = []
    weights = data.grad_weights
    if weights.size and float(np.max(weights)) > 0.0:
        bins = 72
        hist = np.zeros((bins,), dtype=np.float64)
        idx = np.floor(data.grad_angles / PI * bins).astype(np.int32) % bins
        np.add.at(hist, idx, weights)
        hist = 0.25 * np.roll(hist, 1) + 0.5 * hist + 0.25 * np.roll(hist, -1)
        order = np.argsort(hist)[::-1]
        picked = 0
        max_v = float(hist[order[0]]) if order.size else 0.0
        for bi in order:
            if picked >= 8 or max_v <= 0.0 or float(hist[bi]) < max_v * 0.12:
                break
            angle = (float(bi) + 0.5) * PI / bins
            if all(_line_delta(angle, existing) > math.radians(4.0) for existing in candidates):
                candidates.append(angle)
                picked += 1

    # Baseline axes and small offsets keep the method conservative near
    # already-good fits and provide deterministic fallback seeds.
    for seed in (base0, base1):
        for off_deg in (-8.0, -4.0, 0.0, 4.0, 8.0):
            candidates.append(_wrap_pi(float(seed) + math.radians(off_deg)))

    # A sparse global safety net catches strongly skewed synthetic cells
    # whose gradient histogram has one dominant edge only.
    for deg in range(0, 180, 30):
        candidates.append(math.radians(float(deg)))

    unique: list[float] = []
    for a in candidates:
        a = _wrap_pi(a)
        if all(_line_delta(a, b) > math.radians(1.0) for b in unique):
            unique.append(a)
    return unique


def _score_pair(
    data: _DiskData,
    theta0: float,
    theta1: float,
    width: float,
    edge_score: float | None = None,
) -> _Fit | None:
    if not _valid_pair(theta0, theta1):
        return None
    s0, c0 = math.sin(theta0), math.cos(theta0)
    s1, c1 = math.sin(theta1), math.cos(theta1)
    # Signed distance to a line through the origin with direction theta.
    d0 = -s0 * data.xs + c0 * data.ys
    d1 = -s1 * data.xs + c1 * data.ys
    q = np.tanh(d0 / width) * np.tanh(d1 / width)
    q_mean = float(q.mean())
    q_centered = q - q_mean
    denom = float(np.dot(q_centered, q_centered))
    val_energy = data.val_energy
    if denom <= 1e-9 or val_energy <= 1e-9:
        return None
    amp = float(np.dot(q_centered, data.vals_centered) / denom)
    mu = float(data.vals.mean()) - amp * q_mean
    residual = data.vals - (mu + amp * q)
    rms = math.sqrt(float(np.mean(residual * residual)))
    corr = abs(float(np.dot(q_centered, data.vals_centered))) / math.sqrt(denom * val_energy)
    rel_rms = rms / max(abs(amp), 1.0)
    if edge_score is None:
        edge_score = _edge_pair_score(data, theta0, theta1)
    a0, a1 = _canonical_pair(theta0, theta1)
    return _Fit(
        theta0=a0,
        theta1=a1,
        width=float(width),
        mu=mu,
        amp=abs(amp),
        rms=rms,
        rel_rms=rel_rms,
        corr=corr,
        edge_score=edge_score,
    )


def _edge_pair_score(data: _DiskData, theta0: float, theta1: float) -> float:
    weights = data.grad_weights
    total = float(np.sum(weights))
    if total <= 1e-9:
        return 0.0
    sigma = math.radians(4.0)
    d0 = np.mod(data.grad_angles - theta0 + PI / 2.0, PI) - PI / 2.0
    d1 = np.mod(data.grad_angles - theta1 + PI / 2.0, PI) - PI / 2.0
    k0 = np.exp(-0.5 * (d0 / sigma) * (d0 / sigma))
    k1 = np.exp(-0.5 * (d1 / sigma) * (d1 / sigma))
    s0 = float(np.dot(weights, k0)) / total
    s1 = float(np.dot(weights, k1)) / total
    balance = 2.0 * min(s0, s1) / max(s0 + s1, 1e-9)
    return (s0 + s1) * balance


def _candidate_fits(data: _DiskData, candidates: Iterable[float]) -> list[_Fit]:
    angles = list(candidates)
    fits: list[_Fit] = []
    for i, a0 in enumerate(angles):
        for a1 in angles[i + 1 :]:
            if not _valid_pair(a0, a1):
                continue
            edge_score = _edge_pair_score(data, a0, a1)
            for width in WIDTHS_PX:
                fit = _score_pair(data, a0, a1, width, edge_score=edge_score)
                if fit is not None:
                    fits.append(fit)
    fits.sort(key=lambda f: (f.objective, f.rel_rms, -f.edge_score, f.width, f.theta0, f.theta1))
    return fits


def _refine_fit(data: _DiskData, seed: _Fit) -> _Fit:
    best = seed
    for step_deg in (2.0, 1.0, 0.5, 0.25):
        step = math.radians(step_deg)
        for d0 in (-step, 0.0, step):
            for d1 in (-step, 0.0, step):
                if d0 == 0.0 and d1 == 0.0:
                    continue
                fit = _score_pair(data, best.theta0 + d0, best.theta1 + d1, best.width)
                if fit is not None and (
                    fit.objective < best.objective - 1e-9
                    or (
                        abs(fit.objective - best.objective) <= 1e-9
                        and (fit.rel_rms, -fit.edge_score) < (best.rel_rms, -best.edge_score)
                    )
                ):
                    best = fit
    return best


def _best_disk_fit(data: _DiskData, base0: float, base1: float) -> _Fit | None:
    candidates = _histogram_candidates(data, base0, base1)
    fits = _candidate_fits(data, candidates)
    if not fits:
        return None
    best = fits[0]
    # Refine several top seeds to reduce dependence on histogram/bin ties.
    for seed in fits[:4]:
        refined = _refine_fit(data, seed)
        if refined.objective < best.objective or (
            abs(refined.objective - best.objective) <= 1e-9
            and (refined.rel_rms, -refined.edge_score) < (best.rel_rms, -best.edge_score)
        ):
            best = refined
    return best


def _sigma_for_fit(base_sigma0: float, base_sigma1: float, fit: _Fit) -> float:
    _ = (base_sigma0, base_sigma1)
    sep_deg = math.degrees(_line_separation(fit.theta0, fit.theta1))
    # Very acute line pairs are inherently less stable: small center
    # errors rotate the short sector boundaries substantially.
    floor_deg = 1.5 if sep_deg >= 55.0 else 3.0
    model_deg = floor_deg + min(6.0, 8.0 * fit.rel_rms)
    return math.radians(model_deg * 0.55)


def disk_sector_refine_row(
    img_u8: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    row: np.ndarray,
    params: DiskSectorParams | None = None,
) -> tuple[np.ndarray, DiskSectorDebug]:
    """Refine one detection row and return ``(row, debug)``.

    ``row`` follows the benchmark/Python binding schema:
    ``x, y, response, contrast, fit_rms, axis0_angle, axis0_sigma,
    axis1_angle, axis1_sigma``.
    """

    params = params or DiskSectorParams()
    out = np.array(row, dtype=np.float32, copy=True)
    cx = float(row[0])
    cy = float(row[1])
    base0 = _wrap_pi(float(row[5]))
    base1 = _wrap_pi(float(row[7]))
    base_sigma0 = float(row[6])
    base_sigma1 = float(row[8])
    base_amp = max(abs(float(row[3])), 1.0)
    base_rel = max(0.0, float(row[4])) / base_amp if math.isfinite(float(row[4])) else float("inf")

    fallback = DiskSectorDebug(
        accepted=False,
        reason="fallback",
        support=0,
        theta0=base0,
        theta1=base1,
        sigma=max(base_sigma0, base_sigma1),
        rms=float(row[4]),
        amp=float(row[3]),
        rel_rms=base_rel,
        base_rel_rms=base_rel,
        corr=0.0,
        edge_score=0.0,
        width=0.0,
        separation_deg=math.degrees(_line_separation(base0, base1)),
        disagreement_deg=0.0,
    )

    data = _extract_disk(img_u8, gx, gy, cx, cy, params)
    if data is None:
        return out, DiskSectorDebug(**{**fallback.__dict__, "reason": "insufficient_support"})

    fit = _best_disk_fit(data, base0, base1)
    if fit is None:
        return out, DiskSectorDebug(
            **{**fallback.__dict__, "reason": "no_valid_pair", "support": int(data.vals.size)}
        )

    disagreement = _pair_disagreement(fit.theta0, fit.theta1, base0, base1)
    sep = _line_separation(fit.theta0, fit.theta1)
    accept_by_margin = fit.rel_rms <= base_rel - params.accept_rel_margin
    accept_by_ratio = fit.rel_rms <= base_rel * params.accept_rel_ratio
    strong_nonorthogonal = sep < math.radians(55.0)
    sharp_orthogonal = sep >= math.radians(75.0) and fit.width <= 0.70
    edge_disagreement = (
        disagreement > math.radians(params.weak_disagree_deg)
        and fit.edge_score >= 0.18
        and fit.rel_rms <= 0.45
    )
    near_orthogonal_blur = (
        sep >= math.radians(65.0)
        and fit.width >= 2.80
        and disagreement <= math.radians(params.weak_disagree_deg)
        and fit.rel_rms > base_rel - 0.08
    )
    residual_accept = (accept_by_margin or accept_by_ratio) and not near_orthogonal_blur
    accepted = (
        fit.amp >= params.min_contrast
        and fit.corr >= params.min_corr
        and fit.edge_score >= 0.035
        and math.isfinite(fit.rel_rms)
        and (residual_accept or edge_disagreement or strong_nonorthogonal or sharp_orthogonal)
    )

    sigma = _sigma_for_fit(base_sigma0, base_sigma1, fit)
    reason = "accepted" if accepted else "residual_gate"

    if accepted:
        out[3] = np.float32(fit.amp)
        out[4] = np.float32(fit.rms)
        out[5] = np.float32(fit.theta0)
        out[6] = np.float32(sigma)
        out[7] = np.float32(fit.theta1)
        out[8] = np.float32(sigma)
    elif disagreement > math.radians(params.weak_disagree_deg):
        inflated = max(base_sigma0, base_sigma1, math.radians(params.inflated_sigma_deg))
        out[6] = np.float32(inflated)
        out[8] = np.float32(inflated)
        reason = "weak_disagreement"

    return out, DiskSectorDebug(
        accepted=accepted,
        reason=reason,
        support=int(data.vals.size),
        theta0=fit.theta0,
        theta1=fit.theta1,
        sigma=sigma,
        rms=fit.rms,
        amp=fit.amp,
        rel_rms=fit.rel_rms,
        base_rel_rms=base_rel,
        corr=fit.corr,
        edge_score=fit.edge_score,
        width=fit.width,
        separation_deg=math.degrees(sep),
        disagreement_deg=math.degrees(disagreement),
    )


def refine_detection_rows(
    img_u8: np.ndarray,
    rows: np.ndarray,
    params: DiskSectorParams | None = None,
) -> np.ndarray:
    """Apply the disk-sector prototype to an ``(N, 9)`` detection array."""

    params = params or DiskSectorParams()
    if rows.size == 0:
        return rows.astype(np.float32, copy=True)
    img = np.ascontiguousarray(img_u8, dtype=np.uint8)
    gx, gy = _sobel_gradients(img)
    refined = np.asarray(rows, dtype=np.float32).copy()
    if img.shape[0] > 128 or img.shape[1] > 128:
        # Full chessboard frames contain thousands of low-response
        # ChESS candidates. The true board corners sit at the top of
        # the response distribution in the synthetic benchmark, so
        # refine only the strongest rows and leave the rest on the
        # sigma-LUT fallback. Patch benchmarks still refine every row.
        n = min(int(params.max_full_image_rows), refined.shape[0])
        row_indices = np.argsort(refined[:, 2], kind="stable")[-n:]
    else:
        row_indices = np.arange(refined.shape[0])
    for i in row_indices.tolist():
        refined[i], _ = disk_sector_refine_row(img, gx, gy, refined[i], params=params)
    return refined
