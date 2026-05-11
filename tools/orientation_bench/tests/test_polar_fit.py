"""Deterministic tests for the benchmark-only polar-sector estimator."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from orientation_bench.config import load_config
from orientation_bench.match import pair_axes
from orientation_bench.polar_fit import (
    PolarFitParams,
    _PolarTrace,
    _extract_polar_traces,
    _model_centered,
    _radial_consistency,
    _score_pair,
    polar_fit_refine_row,
)
from orientation_bench.sweep import iter_synth_patch_cells


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "tools" / "orientation_bench" / "configs" / "bench_default.yaml"


def _row(
    xy: tuple[float, float],
    axis0: float,
    axis1: float,
    *,
    contrast: float = 120.0,
    fit_rms: float = 32.0,
    sigma_deg: float = 3.0,
) -> np.ndarray:
    return np.array(
        [
            xy[0],
            xy[1],
            1.0,
            contrast,
            fit_rms,
            axis0,
            math.radians(sigma_deg),
            axis1,
            math.radians(sigma_deg),
        ],
        dtype=np.float32,
    )


def _axis_error_deg(row: np.ndarray, gt: tuple[float, float]) -> float:
    d0, d1, _ = pair_axes(float(row[5]), float(row[7]), float(gt[0]), float(gt[1]))
    return max(abs(d0), abs(d1)) * 180.0 / math.pi


def _patch_sample(sweep_name: str, value: float, sample_idx: int = 0):
    cfg = load_config(CONFIG)
    sweep = next(s for s in cfg.sweeps if s.name == sweep_name)
    for _, params, samples in iter_synth_patch_cells(cfg, sweep):
        if abs(float(params.get(sweep.param)) - float(value)) < 1e-9:
            return samples[sample_idx]
    raise AssertionError(f"missing sample for {sweep_name}={value}")


def test_polar_fit_recovers_extreme_projective_axes_from_bad_seed() -> None:
    sample = _patch_sample("axis_skew", 30.0, sample_idx=0)
    bad = _row(
        sample.gt_xy,
        sample.gt_angles[0] + math.radians(100.0),
        sample.gt_angles[1] + math.radians(130.0),
        fit_rms=25.0,
    )

    refined, debug = polar_fit_refine_row(sample.img, bad)

    assert debug.accepted, debug
    assert debug.valid_radii >= 4
    assert _axis_error_deg(refined, sample.gt_angles) < 5.0


def test_polar_fit_matches_clean_orthogonal_seed() -> None:
    sample = _patch_sample("axis_skew", 90.0, sample_idx=2)
    good = _row(
        sample.gt_xy,
        sample.gt_angles[0],
        sample.gt_angles[1],
        fit_rms=8.0,
        sigma_deg=2.0,
    )

    refined, debug = polar_fit_refine_row(sample.img, good)

    assert debug.reason in {"accepted", "confidence_gate"}
    assert _axis_error_deg(refined, sample.gt_angles) < 2.0


def test_polar_fit_falls_back_near_border_support() -> None:
    img = np.zeros((41, 41), dtype=np.uint8)
    row = _row((1.0, 1.0), 0.0, math.pi / 2.0)

    refined, debug = polar_fit_refine_row(img, row)

    assert not debug.accepted
    assert debug.reason == "insufficient_support"
    assert debug.valid_radii == 0
    assert np.allclose(refined, row)


def test_radial_consistency_drops_on_corrupted_radii() -> None:
    params = replace(PolarFitParams(), min_valid_radii=4)
    sample = _patch_sample("axis_skew", 30.0, sample_idx=0)
    traces, phis = _extract_polar_traces(
        sample.img,
        sample.gt_xy[0],
        sample.gt_xy[1],
        params,
    )
    assert len(traces) >= 4

    good_model = _model_centered(
        phis,
        sample.gt_angles[0],
        sample.gt_angles[1],
        params.beta,
    ) * 80.0
    clean_traces = [
        _PolarTrace(
            radius=float(radius),
            values_centered=good_model,
            energy=float(np.dot(good_model, good_model)),
        )
        for radius in params.radii
    ]

    good = _score_pair(
        clean_traces,
        phis,
        sample.gt_angles[0],
        sample.gt_angles[1],
        params,
    )
    assert good is not None
    wrong0 = sample.gt_angles[0] + math.radians(40.0)
    wrong1 = sample.gt_angles[1] + math.radians(40.0)
    wrong = _score_pair(clean_traces, phis, wrong0, wrong1, params)
    assert wrong is not None

    clean_consistency = _radial_consistency(
        clean_traces,
        phis,
        good,
        [good, wrong],
        params,
    )
    wrong_model = _model_centered(phis, wrong0, wrong1, params.beta) * 80.0
    corrupted: list[_PolarTrace] = []
    for idx, trace in enumerate(clean_traces):
        if idx % 2 == 0:
            corrupted.append(trace)
        else:
            values = wrong_model.astype(np.float64, copy=True)
            corrupted.append(
                _PolarTrace(
                    radius=trace.radius,
                    values_centered=values,
                    energy=float(np.dot(values, values)),
                )
            )

    corrupted_consistency = _radial_consistency(
        corrupted,
        phis,
        good,
        [good, wrong],
        params,
    )

    assert clean_consistency > 0.90
    assert corrupted_consistency < 0.50
    assert corrupted_consistency < clean_consistency
