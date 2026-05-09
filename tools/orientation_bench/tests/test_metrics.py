"""Aggregate metric tests."""

from __future__ import annotations

import math

import numpy as np

from orientation_bench.metrics import CellArrays, aggregate_cell, round_floats


def _empty_arrays(matched: int = 0, gt_total: int = 0) -> CellArrays:
    z = np.empty((0,), dtype=np.float64)
    return CellArrays(
        delta0=z,
        delta1=z,
        sigma0=z,
        sigma1=z,
        fit_rms=z,
        axis_swap_flag=np.empty((0,), dtype=bool),
        degenerate_flag=np.empty((0,), dtype=bool),
        matched=matched,
        gt_total=gt_total,
    )


def test_bias_unbiased_gaussian():
    rng = np.random.default_rng(0)
    n = 10000
    sigma_rad = 0.01
    delta = rng.normal(0.0, sigma_rad, size=n)
    arrays = CellArrays(
        delta0=delta,
        delta1=delta,
        sigma0=np.full(n, sigma_rad),
        sigma1=np.full(n, sigma_rad),
        fit_rms=np.full(n, 0.05),
        axis_swap_flag=np.zeros(n, dtype=bool),
        degenerate_flag=np.zeros(n, dtype=bool),
        matched=n,
        gt_total=n,
    )
    out = aggregate_cell(arrays)
    # 5/sqrt(N) sigma in radians, converted to degrees
    threshold_rad = 5.0 / math.sqrt(n) * sigma_rad
    threshold_deg = threshold_rad * 180.0 / math.pi
    assert abs(out["bias_axis0_deg"]) < threshold_deg
    assert abs(out["bias_axis1_deg"]) < threshold_deg


def test_z_std_calibrated():
    rng = np.random.default_rng(1)
    n = 50000
    sigmas = rng.uniform(0.005, 0.05, size=n)
    deltas = rng.normal(0.0, 1.0, size=n) * sigmas
    arrays = CellArrays(
        delta0=deltas,
        delta1=deltas,
        sigma0=sigmas,
        sigma1=sigmas,
        fit_rms=np.full(n, 0.05),
        axis_swap_flag=np.zeros(n, dtype=bool),
        degenerate_flag=np.zeros(n, dtype=bool),
        matched=n,
        gt_total=n,
    )
    out = aggregate_cell(arrays)
    assert abs(out["z_std_axis0"] - 1.0) < 0.05
    assert abs(out["z_std_axis1"] - 1.0) < 0.05
    # outlier fraction for a true normal should be ~4.5%
    assert 0.02 < out["z_outlier_frac_axis0"] < 0.08


def test_failure_pct_empty():
    out = aggregate_cell(_empty_arrays())
    assert out["failure_pct"] == 0.0
    assert out["detection_pct"] == 0.0
    assert out["n_samples"] == 0


def test_failure_pct_clamped():
    n = 100
    arrays = CellArrays(
        delta0=np.zeros(n),
        delta1=np.zeros(n),
        sigma0=np.full(n, 0.01),
        sigma1=np.full(n, 0.01),
        fit_rms=np.full(n, 0.0),
        axis_swap_flag=np.ones(n, dtype=bool),
        degenerate_flag=np.ones(n, dtype=bool),
        matched=n,
        gt_total=n,
    )
    out = aggregate_cell(arrays)
    # both flags fire on every sample => raw sum is 2.0; we clamp to 1.0
    assert out["failure_pct"] == 1.0


def test_round_floats_recursive():
    payload = {
        "a": 1.234567890123,
        "b": [0.5555555555, {"c": 0.9876543210}],
        "d": float("nan"),
        "e": "string",
    }
    out = round_floats(payload, digits=6)
    assert out["a"] == round(1.234567890123, 6)
    assert out["b"][0] == round(0.5555555555, 6)
    assert out["b"][1]["c"] == round(0.9876543210, 6)
    assert math.isnan(out["d"])
    assert out["e"] == "string"
