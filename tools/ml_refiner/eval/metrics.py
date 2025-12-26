"""Metrics helpers for ML/OpenCV comparison."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def summarize_errors(errors: np.ndarray, valid_mask: np.ndarray | None = None) -> Dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(errors)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    total = int(errors.size)
    valid = int(np.sum(valid_mask))
    invalid = total - valid

    summary: Dict[str, Any] = {
        "count": total,
        "count_valid": valid,
        "failure_rate": float(invalid / max(1, total)),
    }
    if valid == 0:
        summary.update(
            {
                "mean": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
        )
        return summary

    valid_errors = errors[valid_mask]
    summary.update(
        {
            "mean": float(np.mean(valid_errors)),
            "p50": float(np.percentile(valid_errors, 50)),
            "p90": float(np.percentile(valid_errors, 90)),
            "p95": float(np.percentile(valid_errors, 95)),
            "p99": float(np.percentile(valid_errors, 99)),
            "max": float(np.max(valid_errors)),
        }
    )
    return summary


def summarize_binned(
    values: np.ndarray,
    errors: np.ndarray,
    valid_mask: np.ndarray,
    bins: Iterable[float],
) -> List[Dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    errors = np.asarray(errors, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    edges = np.asarray(list(bins), dtype=np.float32)
    if edges.size < 2:
        raise ValueError("bins must have at least two edges")

    summaries: List[Dict[str, Any]] = []
    for i in range(edges.size - 1):
        low = float(edges[i])
        high = float(edges[i + 1])
        if i == edges.size - 2:
            mask = (values >= low) & (values <= high)
        else:
            mask = (values >= low) & (values < high)
        mask_valid = mask & valid_mask
        summary = summarize_errors(errors[mask], mask_valid[mask])
        summary["range"] = [low, high]
        summary["count_in_bin"] = int(np.sum(mask))
        summaries.append(summary)

    return summaries


def summarize_conf(conf_pred: np.ndarray, is_pos: np.ndarray) -> Dict[str, Any]:
    conf_pred = np.asarray(conf_pred, dtype=np.float32)
    is_pos = np.asarray(is_pos, dtype=np.float32)
    pos_mask = is_pos > 0.5
    neg_mask = ~pos_mask

    out: Dict[str, Any] = {
        "pos_count": int(np.sum(pos_mask)),
        "neg_count": int(np.sum(neg_mask)),
    }
    if np.any(pos_mask):
        out["conf_pos_mean"] = float(np.mean(conf_pred[pos_mask]))
        out["conf_pos_p95"] = float(np.percentile(conf_pred[pos_mask], 95))
    else:
        out["conf_pos_mean"] = None
        out["conf_pos_p95"] = None
    if np.any(neg_mask):
        out["conf_neg_mean"] = float(np.mean(conf_pred[neg_mask]))
        out["conf_neg_p95"] = float(np.percentile(conf_pred[neg_mask], 95))
    else:
        out["conf_neg_mean"] = None
        out["conf_neg_p95"] = None
    return out
