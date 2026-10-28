"""Shared pieces of the real-data validation harness (UQ-30).

Run (from this directory, with the chess_corners bindings installed):

    python detect.py <dataset_root>
    python calibrate.py <dataset_root> --set all     # and/or --set chess
    python analyze.py --set all

Report: docs/reference/real-data-validation.md.

Dataset layout (read-only, never copied): ``<root>/cam1/Cam1_<token>.png`` and
``<root>/cam2/Cam2_<token>.png``; ``<token>`` = ``<a>Rx<b>Ry<x>X<y>Y<z>Z``, the
stage pose (two rotations in degrees, translations in mm). Views pair across
cameras by the token. Target: ChArUco 22 x 22 squares, 1.35 mm squares,
1.0125 mm markers, DICT_4X4_1000, non-legacy layout.

Everything derived from the images is written to ``out/`` next to this file
(gitignored), or to ``$RV_OUT`` when set.

Conventions: image pixels with the origin at the top-left pixel *centre*
(OpenCV, and the chess-corners detector); board frame from
``CharucoBoard.getChessboardCorners`` (mm, z = 0); a pose ``(R, t)`` maps board
(or stage) points into the camera frame, ``X_c = R X + t``.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("RV_OUT", HERE / "out"))
sys.path.insert(0, str(HERE.parent))

SQUARE = 1.35
MARKER = 1.0125
NSQ = 22
CAMS = ("cam1", "cam2")
LEVELS = np.array([-14.0, -9.0, 0.0, 9.0, 14.0])  # commanded stage angles, deg
TOKEN = re.compile(r"^Cam\d+_(?P<tok>([+-]\d+\.\d)Rx([+-]\d+\.\d)Ry([+-]\d+\.\d)X([+-]\d+\.\d)Y([+-]\d+\.\d)Z)\.png$")

# Localizers. Each yields positions and, when the bindings report one, a
# position covariance. `cSP-11` is cv2.cornerSubPix with an 11x11 window,
# seeded from the rounded reference corners.
LOCALIZERS = ["CoM", "Forstner", "SaddlePoint", "Radon", "Radon+SP", "cSP-11"]


def parse_token(name: str):
    m = TOKEN.match(name)
    if m is None:
        return None
    a, b, x, y, z = (float(m.group(i)) for i in range(2, 7))
    return m.group("tok"), np.array([a, b, x, y, z])


def board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    b = cv2.aruco.CharucoBoard((NSQ, NSQ), SQUARE, MARKER, d)
    b.setLegacyPattern(False)
    return b


def board_points() -> np.ndarray:
    """(441, 3) inner-corner coordinates in mm, indexed by ChArUco corner id."""
    return board().getChessboardCorners().astype(np.float64)


def folds(stage: np.ndarray) -> np.ndarray:
    """2-fold split of views: a checkerboard on the (Rx, Ry) grid, so each fold
    spans both axes at every angle magnitude; the two Z views fall in
    different folds."""
    ia = np.searchsorted(LEVELS, stage[:, 0])
    ib = np.searchsorted(LEVELS, stage[:, 1])
    f = (ia + ib) % 2
    zv = stage[:, 4] != 0
    f[zv] = (stage[zv, 4] > 0).astype(int)
    return f


# ---------------------------------------------------------------------------
# Camera model: OpenCV pinhole + (k1, k2, p1, p2, k3)
# ---------------------------------------------------------------------------


def project(Xc: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Camera-frame points (N, 3) -> pixels (N, 2). ``intr`` = fx, fy, cx, cy,
    k1, k2, p1, p2, k3."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def intr_from_cv(K, D) -> np.ndarray:
    D = np.asarray(D, float).ravel()
    D = np.concatenate([D, np.zeros(5 - len(D))])[:5]
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D])


def intr_to_cv(intr):
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1.0]])
    return K, np.asarray(intr[4:9], float)


def rt_to_T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.ravel(rvec)).as_matrix()
    T[:3, 3] = np.ravel(tvec)
    return T


def T_to_rt(T):
    return Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3].copy()


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def axis_rotation(omega: np.ndarray, q: np.ndarray, theta: float) -> np.ndarray:
    """Rotation by ``theta`` (rad) about the line through ``q`` along unit ``omega``."""
    T = np.eye(4)
    R = Rotation.from_rotvec(omega * theta).as_matrix()
    T[:3, :3] = R
    T[:3, 3] = q - R @ q
    return T


def board_jacobian(R, t, X, intr, h: float = 1e-3) -> np.ndarray:
    """d(pixel)/d(board x, y, z) in px/mm per observation: (N, 2, 3).
    ``R`` (N, 3, 3), ``t`` (N, 3), ``X`` (N, 3) board points."""
    cols = []
    for k in (0, 1, 2):
        dX = np.zeros(3)
        dX[k] = h
        up = project(np.einsum("nij,nj->ni", R, X + dX) + t, intr)
        dn = project(np.einsum("nij,nj->ni", R, X - dX) + t, intr)
        cols.append((up - dn) / (2 * h))
    return np.stack(cols, 2)


# ---------------------------------------------------------------------------
# Localizers
# ---------------------------------------------------------------------------

_DET: dict = {}


def _detector(name: str):
    import chess_corners as cc

    if name not in _DET:
        if name in ("Radon", "Radon+SP"):
            cfg = cc.DetectorConfig.radon().without_orientation()
            if name == "Radon+SP":
                cfg.strategy.radon.refiner = cc.RadonRefiner.saddle_point()
        else:
            cfg = cc.DetectorConfig.chess().without_orientation()
            cfg.strategy.chess.refiner = {
                "CoM": cc.ChessRefiner.center_of_mass(),
                "Forstner": cc.ChessRefiner.forstner(),
                "SaddlePoint": cc.ChessRefiner.saddle_point(),
            }[name]
        _DET[name] = cc.Detector(cfg)
    return _DET[name]


def localize(img: np.ndarray, name: str, ref_xy: np.ndarray):
    """``(xy (N, 2), cov (N, 2, 2) or None)`` for one localizer. The covariance
    is whatever the installed bindings report (``position_covariance``);
    all-NaN becomes ``None``, so newer bindings that report a covariance for
    more localizers are picked up without code changes."""
    if name == "cSP-11":
        s0 = np.round(ref_xy).astype(np.float32).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        out = cv2.cornerSubPix(img, s0.copy(), (5, 5), (-1, -1), crit).reshape(-1, 2).astype(np.float64)
        ok = np.all(np.abs(out - s0.reshape(-1, 2)) <= 1.5, axis=1)
        return out[ok], None
    d = _detector(name).detect(img)
    xy = np.asarray(d.xy, dtype=np.float64)
    cov = np.asarray(d.position_covariance, dtype=np.float64)
    if cov.size == 0 or np.isnan(cov).all():
        cov = None
    return xy, cov


def match(xy: np.ndarray, ref: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Index into ``xy`` of the detection nearest each reference corner, or -1."""
    if len(xy) == 0:
        return np.full(len(ref), -1)
    from scipy.spatial import cKDTree

    d, j = cKDTree(xy).query(ref)
    return np.where(d <= radius, j, -1)


# ---------------------------------------------------------------------------
# Covariance scoring
# ---------------------------------------------------------------------------


def sigma0_sq(e: np.ndarray, S: np.ndarray, dof: float = 0.0) -> float:
    """``mean(e' S^-1 e) / 2 / (1 - dof)`` over the rows where ``e`` (N, 2) and
    ``S`` (N, 2, 2) are finite; ``dof`` is the fraction of the residual's
    degrees of freedom a fit already absorbed."""
    ok = np.isfinite(S).all(axis=(1, 2)) & np.isfinite(e).all(1)
    q = np.einsum("ni,nij,nj->n", e[ok], np.linalg.inv(S[ok]), e[ok])
    return float(q.mean() / 2 / (1 - dof))


def fit_floor(e: np.ndarray, S: np.ndarray, dof: float = 0.0) -> float:
    """Isotropic floor ``f`` (px per axis) for which ``S + f^2 I`` whitens
    ``e`` to sigma0^2 = 1; 0 when ``S`` alone is already conservative."""
    from scipy.optimize import brentq

    def s0(f):
        return sigma0_sq(e, S + f * f * np.eye(2)[None], dof)

    return float(brentq(lambda f: s0(f) - 1.0, 0.0, 1.0)) if s0(0.0) > 1 else 0.0
