"""Tests for matching and axis pairing."""

from __future__ import annotations

import math

import numpy as np

from orientation_bench.match import match_to_gt, pair_axes, wrap_to_pi_over_2


def test_wrap_to_pi_over_2_scalar_and_array():
    assert math.isclose(wrap_to_pi_over_2(0.0), 0.0, abs_tol=1e-12)
    arr = np.array([math.pi, -math.pi, math.pi / 4.0, math.pi - 1e-9])
    out = wrap_to_pi_over_2(arr)
    # +pi -> 0
    assert abs(out[0]) < 1e-9
    # -pi -> 0
    assert abs(out[1]) < 1e-9
    assert math.isclose(out[2], math.pi / 4.0, abs_tol=1e-12)


def test_pair_axes_direct_zero_error():
    p1 = 0.3
    p2 = 0.3 + math.pi / 2.0
    g1 = 0.3
    g2 = 0.3 + math.pi / 2.0
    d0, d1, swap = pair_axes(p1, p2, g1, g2)
    assert abs(d0) < 1e-12
    assert abs(d1) < 1e-12
    assert swap is False


def test_pair_axes_swap_required():
    rng = np.random.default_rng(123)
    correct_swap = 0
    for _ in range(100):
        theta = rng.uniform(0.0, math.pi)
        # GT pair, but provide them swapped
        true1 = theta
        true2 = theta + math.pi / 2.0
        # noise
        eps0 = rng.normal(0.0, 0.005)
        eps1 = rng.normal(0.0, 0.005)
        # Predictions are at (true1, true2) but we deliberately label GT as
        # (true2, true1) — the swap pairing should win.
        p1 = true1 + eps0
        p2 = true2 + eps1
        g1 = true2
        g2 = true1
        d0, d1, swap = pair_axes(p1, p2, g1, g2)
        assert swap is True
        if abs(d0) < 0.05 and abs(d1) < 0.05:
            correct_swap += 1
    assert correct_swap > 90


def test_pair_axes_direct_wins():
    """Mirror of `test_pair_axes_swap_required` with un-swapped GT.

    Predictions and GT share the same ordering ``(theta, theta+pi/2)``,
    so the direct pairing must always be chosen.
    """
    rng = np.random.default_rng(456)
    direct_count = 0
    for _ in range(100):
        theta = rng.uniform(0.0, math.pi)
        true1 = theta
        true2 = theta + math.pi / 2.0
        eps0 = rng.normal(0.0, 0.005)
        eps1 = rng.normal(0.0, 0.005)
        p1 = true1 + eps0
        p2 = true2 + eps1
        g1 = true1
        g2 = true2
        d0, d1, swap = pair_axes(p1, p2, g1, g2)
        assert swap is False
        if abs(d0) < 0.05 and abs(d1) < 0.05:
            direct_count += 1
    assert direct_count == 100


def test_match_to_gt_simple():
    pred = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    gt = np.array([[0.1, 0.0], [10.05, 0.0]], dtype=np.float64)
    p, g = match_to_gt(pred, gt, threshold_px=1.0)
    assert p.size == 2 and g.size == 2
    pairs = sorted(zip(p.tolist(), g.tolist()))
    assert pairs == [(0, 0), (1, 1)]


def test_match_to_gt_rejects_far_outlier():
    pred = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float64)
    gt = np.array([[0.05, 0.0]], dtype=np.float64)
    p, g = match_to_gt(pred, gt, threshold_px=1.0)
    assert p.tolist() == [0]
    assert g.tolist() == [0]


def test_match_to_gt_empty_inputs():
    p, g = match_to_gt(
        np.empty((0, 2), dtype=np.float64),
        np.array([[1.0, 2.0]]),
        threshold_px=1.0,
    )
    assert p.size == 0 and g.size == 0
