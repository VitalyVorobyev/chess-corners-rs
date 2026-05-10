"""Ground-truth angle math tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orientation_bench.gt import (
    axes_mod_pi,
    gt_axes_from_homography,
    gt_axes_synthetic,
    jacobian_2x2,
)


def _wrap(a: float) -> float:
    out = math.fmod(a, math.pi)
    if out < 0:
        out += math.pi
    return out


def test_synthetic_no_homography_tanh():
    theta = 0.3
    a, b = gt_axes_synthetic(theta, "tanh", H=None)
    assert math.isclose(a, _wrap(theta), abs_tol=1e-12)
    assert math.isclose(b, _wrap(theta + math.pi / 2.0), abs_tol=1e-12)


def test_synthetic_no_homography_hard_cells():
    theta = 1.0
    a, b = gt_axes_synthetic(theta, "hard_cells", H=None)
    assert math.isclose(a, _wrap(theta), abs_tol=1e-12)
    assert math.isclose(b, _wrap(theta + math.pi / 2.0), abs_tol=1e-12)


def test_synthetic_with_rotation_matrix():
    theta = 0.0
    rot_deg = 30.0
    c = math.cos(math.radians(rot_deg))
    s = math.sin(math.radians(rot_deg))
    H = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    a, b = gt_axes_synthetic(theta, "hard_cells", H=H)
    expected_a = _wrap(math.radians(rot_deg))
    expected_b = _wrap(math.radians(rot_deg) + math.pi / 2.0)
    assert math.isclose(a, expected_a, abs_tol=1e-9)
    assert math.isclose(b, expected_b, abs_tol=1e-9)


def test_unknown_render_mode():
    with pytest.raises(ValueError):
        gt_axes_synthetic(0.0, "garbage", H=None)


def test_homography_identity_basis():
    H = np.eye(3, dtype=np.float64)
    grid = np.array([[10.0, 20.0], [0.0, 0.0], [-5.0, 7.0]], dtype=np.float64)
    angles = gt_axes_from_homography(H, grid)
    assert angles.shape == (3, 2)
    for row in angles:
        assert math.isclose(row[0], 0.0, abs_tol=1e-12)
        assert math.isclose(row[1], math.pi / 2.0, abs_tol=1e-12)


def _local_jacobian_fd(H: np.ndarray, u: float, v: float, eps: float = 1e-3) -> np.ndarray:
    """Local copy of the FD helper from tools/ml_refiner/synth/homography.py.

    We import a fresh implementation rather than reach into ml_refiner so this
    test does not depend on the bench's path bootstrap.
    """

    def project(uu: float, vv: float) -> tuple[float, float]:
        w = float(H[2, 0] * uu + H[2, 1] * vv + H[2, 2])
        x = float(H[0, 0] * uu + H[0, 1] * vv + H[0, 2]) / w
        y = float(H[1, 0] * uu + H[1, 1] * vv + H[1, 2]) / w
        return x, y

    bx, by = project(u, v)
    dxu, dyu = project(u + eps, v)
    dxv, dyv = project(u, v + eps)
    j00 = (dxu - bx) / eps
    j10 = (dyu - by) / eps
    j01 = (dxv - bx) / eps
    j11 = (dyv - by) / eps
    return np.array([[j00, j01], [j10, j11]], dtype=np.float64)


def _random_homography(rng: np.random.Generator) -> np.ndarray:
    while True:
        H = rng.normal(0.0, 0.5, size=(3, 3))
        # bias the affine part toward identity so we don't get pathological cases
        H[0, 0] += 1.0
        H[1, 1] += 1.0
        H[2, 2] = 1.0
        H[2, 0] *= 0.05
        H[2, 1] *= 0.05
        try:
            np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        # ensure non-degenerate at the test point
        w = H[2, 0] * 0.5 + H[2, 1] * 0.5 + H[2, 2]
        if abs(w) > 0.1:
            return H


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(42)
    for _ in range(50):
        H = _random_homography(rng)
        u = float(rng.uniform(-5.0, 5.0))
        v = float(rng.uniform(-5.0, 5.0))
        J_closed = jacobian_2x2(H, u, v)
        J_fd = _local_jacobian_fd(H, u, v, eps=1e-4)
        diff = np.max(np.abs(J_closed - J_fd))
        assert diff < 1e-3, f"jacobian mismatch {diff}: closed={J_closed} fd={J_fd}"


def test_axes_mod_pi_array():
    arr = np.array([0.0, math.pi, -math.pi / 4.0, 2.0 * math.pi])
    out = axes_mod_pi(arr)
    assert math.isclose(out[0], 0.0, abs_tol=1e-12)
    assert math.isclose(out[1], 0.0, abs_tol=1e-12)
    assert math.isclose(out[2], math.pi - math.pi / 4.0, abs_tol=1e-12)
    assert math.isclose(out[3], 0.0, abs_tol=1e-12)


def test_gt_axes_from_homography_flags_degenerate():
    # Near-singular H: the upper-left 2x2 block has a near-zero second
    # column (rank deficiency), so the local Jacobian's e_2 projection
    # collapses. The guard should emit (NaN, NaN).
    H = np.array(
        [
            [1.0, 1e-9, 0.0],
            [0.0, 1e-9, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = gt_axes_from_homography(H, np.array([[0.0, 0.0]]))
    assert np.isnan(out).all()


def test_gt_axes_from_homography_well_conditioned_finite():
    # Identity H: J = I, both columns unit-norm, well-conditioned.
    H = np.eye(3, dtype=np.float64)
    out = gt_axes_from_homography(H, np.array([[0.0, 0.0], [3.0, -2.0]]))
    assert np.isfinite(out).all()
