"""Matching predicted corners to GT and pairing axes under mod-pi swap.

The two-axis fit emits orthogonal lines, so an absolute angle and an
angle ``+ pi`` are the same line. To compare a predicted axis pair
``(p1, p2)`` to a GT pair ``(g1, g2)`` we wrap differences into
(-pi/2, pi/2] and pick the pairing (direct vs. swap) that minimises
the worst per-axis error.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "wrap_to_pi_over_2",
    "pair_axes",
    "pair_axes_array",
    "match_to_gt",
]


def wrap_to_pi_over_2(d):
    """Wrap angle differences into ``(-pi/2, pi/2]``. Vectorised.

    The half-open interval is ``(-pi/2, pi/2]``: any value within
    ``1e-15`` of ``-pi/2`` (including an exact ``-pi/2`` input) is
    promoted to ``+pi/2``. For line-angle differences the two endpoints
    are equivalent, so this normalisation is harmless and keeps the
    output strictly in the documented interval.
    """
    arr = np.asarray(d, dtype=np.float64)
    out = np.mod(arr + math.pi / 2.0, math.pi) - math.pi / 2.0
    # numpy's mod is non-negative for positive divisor, so result is in
    # [-pi/2, pi/2). Promote -pi/2 to pi/2 to land in (-pi/2, pi/2].
    out = np.where(out <= -math.pi / 2.0 + 1e-15, math.pi / 2.0, out)
    return out


def pair_axes(
    p1: float, p2: float, g1: float, g2: float
) -> Tuple[float, float, bool]:
    """Pick the axis pairing with the smaller maximum error.

    Returns (delta0, delta1, swap_flag) where:
      - direct: delta0 = wrap(p1-g1), delta1 = wrap(p2-g2), swap=False
      - swap:   delta0 = wrap(p1-g2), delta1 = wrap(p2-g1), swap=True
    """
    d_direct_0 = float(wrap_to_pi_over_2(p1 - g1))
    d_direct_1 = float(wrap_to_pi_over_2(p2 - g2))
    d_swap_0 = float(wrap_to_pi_over_2(p1 - g2))
    d_swap_1 = float(wrap_to_pi_over_2(p2 - g1))
    direct_score = max(abs(d_direct_0), abs(d_direct_1))
    swap_score = max(abs(d_swap_0), abs(d_swap_1))
    if direct_score <= swap_score:
        return d_direct_0, d_direct_1, False
    return d_swap_0, d_swap_1, True


def pair_axes_array(
    p1: np.ndarray, p2: np.ndarray, g1: np.ndarray, g2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised version of `pair_axes`.

    Returns three arrays of identical shape: (delta0, delta1, swap_flag).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    d_direct_0 = wrap_to_pi_over_2(p1 - g1)
    d_direct_1 = wrap_to_pi_over_2(p2 - g2)
    d_swap_0 = wrap_to_pi_over_2(p1 - g2)
    d_swap_1 = wrap_to_pi_over_2(p2 - g1)
    direct_score = np.maximum(np.abs(d_direct_0), np.abs(d_direct_1))
    swap_score = np.maximum(np.abs(d_swap_0), np.abs(d_swap_1))
    swap_flag = swap_score < direct_score
    delta0 = np.where(swap_flag, d_swap_0, d_direct_0)
    delta1 = np.where(swap_flag, d_swap_1, d_direct_1)
    return delta0, delta1, swap_flag


def match_to_gt(
    pred_xy: np.ndarray, gt_xy: np.ndarray, threshold_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour matching within a pixel threshold.

    Returns (matched_pred_idx, matched_gt_idx) where the entries are
    aligned: matched_pred_idx[i] is paired with matched_gt_idx[i].
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if pred.size == 0 or gt.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    diff = pred[:, None, :] - gt[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    candidates = np.argwhere(dists <= float(threshold_px))
    if candidates.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    cand_dists = dists[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(cand_dists, kind="stable")
    pred_used = np.zeros(pred.shape[0], dtype=bool)
    gt_used = np.zeros(gt.shape[0], dtype=bool)
    matched_pred: list[int] = []
    matched_gt: list[int] = []
    for k in order:
        pi = int(candidates[k, 0])
        gi = int(candidates[k, 1])
        if pred_used[pi] or gt_used[gi]:
            continue
        pred_used[pi] = True
        gt_used[gi] = True
        matched_pred.append(pi)
        matched_gt.append(gi)
    return (
        np.asarray(matched_pred, dtype=np.int64),
        np.asarray(matched_gt, dtype=np.int64),
    )
