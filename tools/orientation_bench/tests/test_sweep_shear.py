"""Verify the axis-skew shear realises the requested skew angle exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orientation_bench import gt


def angle_between(a: float, b: float) -> float:
    """Acute angle between two line directions (mod pi) in degrees."""
    d = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(d))


@pytest.mark.parametrize("skew_deg", [50, 60, 70, 80, 90, 100, 110, 120])
def test_axis_skew_shear_realised(skew_deg):
    if abs(skew_deg - 90.0) < 1e-6:
        H = np.eye(3, dtype=np.float32)
    else:
        k = math.tan(math.radians(90.0 - skew_deg))
        H = np.array(
            [[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    theta1, theta2 = gt.gt_axes_synthetic(theta=0.0, render_mode="tanh", H=H)
    realised = angle_between(theta1, theta2)
    expected = min(skew_deg, 180 - skew_deg)  # mod-pi folding
    assert abs(realised - expected) < 1e-3, (
        f"got {realised}, expected {expected} (skew_deg={skew_deg})"
    )
