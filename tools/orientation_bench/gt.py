"""Ground-truth axis angles for the orientation benchmark.

Two paths:
  * `gt_axes_synthetic` — a single patch rendered with rotation `theta`
    and (optionally) warped by a 3x3 homography `H`. The unwarped grid
    axes sit at `theta` and `theta + pi/2` (mod pi). Under H they are
    transformed by the literal 2x2 Jacobian of H at the patch center
    (0, 0). No re-orthonormalisation: the axes are allowed to skew.
  * `gt_axes_from_homography` — for full chessboards, project the
    texture-space basis vectors (1, 0) and (0, 1) through the local
    Jacobian of the texture-to-image homography at each corner.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "jacobian_2x2",
    "gt_axes_synthetic",
    "gt_axes_from_homography",
    "axes_mod_pi",
]


def _wrap_mod_pi(angle: float) -> float:
    a = math.fmod(angle, math.pi)
    if a < 0.0:
        a += math.pi
    return a


def axes_mod_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap an array of angles into [0, pi)."""
    return np.mod(angles, math.pi)


def jacobian_2x2(H: np.ndarray, u: float, v: float) -> np.ndarray:
    """Closed-form 2x2 Jacobian of the homography mapping (u, v) -> (x, y).

    For ``H = [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]``:

        w = h20*u + h21*v + h22
        x = (h00*u + h01*v + h02) / w
        y = (h10*u + h11*v + h12) / w
        J = (1/w) * [[h00 - x*h20,  h01 - x*h21],
                     [h10 - y*h20,  h11 - y*h21]]
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got {H.shape}")
    h00, h01, h02 = float(H[0, 0]), float(H[0, 1]), float(H[0, 2])
    h10, h11, h12 = float(H[1, 0]), float(H[1, 1]), float(H[1, 2])
    h20, h21, h22 = float(H[2, 0]), float(H[2, 1]), float(H[2, 2])
    w = h20 * u + h21 * v + h22
    if not np.isfinite(w) or abs(w) < 1e-12:
        raise ValueError("homography degenerate at point (zero w)")
    x = (h00 * u + h01 * v + h02) / w
    y = (h10 * u + h11 * v + h12) / w
    j00 = (h00 - x * h20) / w
    j01 = (h01 - x * h21) / w
    j10 = (h10 - y * h20) / w
    j11 = (h11 - y * h21) / w
    return np.array([[j00, j01], [j10, j11]], dtype=np.float64)


def _project_basis_through_jacobian(J: np.ndarray, theta: float) -> tuple[float, float]:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    e1 = np.array([cos_t, sin_t], dtype=np.float64)
    e2 = np.array([-sin_t, cos_t], dtype=np.float64)
    a1 = J @ e1
    a2 = J @ e2
    theta1 = _wrap_mod_pi(math.atan2(float(a1[1]), float(a1[0])))
    theta2 = _wrap_mod_pi(math.atan2(float(a2[1]), float(a2[0])))
    return theta1, theta2


def gt_axes_synthetic(
    theta: float,
    render_mode: str,
    H: np.ndarray | None = None,
) -> tuple[float, float]:
    """GT axis angles (mod pi) for a synthetic patch.

    The unwarped axes sit at ``theta`` and ``theta + pi/2``. If ``H`` is
    provided, the axes are transformed by the 2x2 Jacobian of H at the
    patch center (0, 0). No re-orthonormalisation.

    `render_mode` is accepted for symmetry with the renderers
    (``"tanh"`` / ``"hard_cells"``); it does not affect the GT angles.
    """
    if render_mode not in ("tanh", "hard_cells"):
        raise ValueError(f"unknown render_mode {render_mode!r}")
    if H is None:
        return _wrap_mod_pi(theta), _wrap_mod_pi(theta + math.pi / 2.0)
    J = jacobian_2x2(H, 0.0, 0.0)
    return _project_basis_through_jacobian(J, theta)


def gt_axes_from_homography(
    H_tex2img: np.ndarray,
    tex_uv: np.ndarray,
) -> np.ndarray:
    """For each (u_tex, v_tex), return (N, 2) of axis angles (mod pi).

    Reference texture-space basis vectors are e1=(1,0) and e2=(0,1).
    They are projected through the local 2x2 Jacobian of H at each
    point and reduced to (-pi, pi] then wrapped to [0, pi).

    Cells where the local Jacobian is rank-deficient or extremely
    anisotropic (one projected basis vector ~zero, or column-norm ratio
    below 5%) are returned as ``(NaN, NaN)``. Callers should treat NaN
    GT angles as degenerate and exclude them from delta aggregations.
    """
    H = np.asarray(H_tex2img, dtype=np.float64)
    pts = np.asarray(tex_uv, dtype=np.float64).reshape(-1, 2)
    out = np.empty((pts.shape[0], 2), dtype=np.float64)
    for i, (u, v) in enumerate(pts):
        J = jacobian_2x2(H, float(u), float(v))
        n1 = math.hypot(float(J[0, 0]), float(J[1, 0]))
        n2 = math.hypot(float(J[0, 1]), float(J[1, 1]))
        if min(n1, n2) < 1e-6 or min(n1, n2) / max(n1, n2, 1e-30) < 0.05:
            out[i] = (math.nan, math.nan)
            continue
        a1 = J @ np.array([1.0, 0.0])
        a2 = J @ np.array([0.0, 1.0])
        out[i, 0] = _wrap_mod_pi(math.atan2(float(a1[1]), float(a1[0])))
        out[i, 1] = _wrap_mod_pi(math.atan2(float(a2[1]), float(a2[0])))
    return out
