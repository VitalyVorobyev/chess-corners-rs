"""Corner estimators compared by the robustness audit, all run on identical
uint8 images.

- ``CoM`` / ``Forstner`` / ``SaddlePoint``: ChESS detector + built-in refiner
  (Python bindings, default configs, single scale, orientation off).
  SaddlePoint also returns its ``position_covariance`` (noise σ estimated per
  image, the default).
- ``Radon``: the Radon detector with its default peak fit.
- ``cSP-5``: ``cv2.cornerSubPix`` with ``winSize=(2, 2)`` (5x5, the same
  support as the built-in refiners). ``cSP-w``: ``winSize=(h, h)`` with ``h``
  chosen per dataset for the cell size (4 -> 9x9 on the ~10 px synthetic
  cells). ``cSP-11``: OpenCV's common ``winSize=(5, 5)`` (11x11), the Part IX
  setting.
- ``RC-Forstner``: a numpy prototype of the built-in Förstner refiner made
  re-centring: the same 5x5 window, weights ``1/(1 + 0.5 d^2)`` and central
  differences, but evaluated on a patch resampled bilinearly around the
  current estimate, iterated by Newton steps (finite-difference Jacobian,
  halving) until the window centre is its own estimate (|u| < 1e-3 px).
- ``Forstner-py``: the same numpy code, single shot at the integer seed (a
  check that the prototype reproduces the built-in Förstner).

Seeds for the OpenCV / numpy estimators: the rounded ChESS detections (union
over the three ChESS refiners).
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndi

import chess_corners as cc

# PRIMARY define the common evaluation set; DIAGNOSTIC estimators are scored
# on (common set) ∩ (their own matches), their failures counted in the match rate.
PRIMARY = ["CoM", "Forstner", "SaddlePoint", "Radon", "cSP-w"]
DIAGNOSTIC = ["cSP-5", "cSP-11", "RC-Forstner"]
ALL = PRIMARY + DIAGNOSTIC

_DETS: dict = {}


def _det(name: str):
    if name not in _DETS:
        if name == "Radon":
            cfg = cc.DetectorConfig.radon().without_orientation()
        else:
            cfg = cc.DetectorConfig.chess().without_orientation()
            cfg.strategy.chess.refiner = {
                "CoM": cc.ChessRefiner.center_of_mass(),
                "Forstner": cc.ChessRefiner.forstner(),
                "SaddlePoint": cc.ChessRefiner.saddle_point(),
            }[name]
        _DETS[name] = cc.Detector(cfg)
    return _DETS[name]


def chess(img: np.ndarray, name: str):
    d = _det(name).detect(img)
    xy = np.asarray(d.xy, dtype=np.float64)
    cov = np.asarray(d.position_covariance, dtype=np.float64) if name == "SaddlePoint" else None
    return xy, cov


def saddle_with_sigma(img: np.ndarray, sigma: float):
    """SaddlePoint detections with a supplied noise sigma (book section 8.5.2)."""
    cfg = cc.DetectorConfig.chess().without_orientation().with_noise_sigma(float(sigma))
    cfg.strategy.chess.refiner = cc.ChessRefiner.saddle_point()
    d = cc.Detector(cfg).detect(img)
    return np.asarray(d.xy, dtype=np.float64), np.asarray(d.position_covariance, dtype=np.float64)


def dedupe(pts: np.ndarray, r: float = 1.5) -> np.ndarray:
    if len(pts) == 0:
        return pts
    keep = []
    for p in pts:
        if all(np.hypot(*(p - q)) > r for q in keep):
            keep.append(p)
    return np.array(keep)


# ---------------------------------------------------------------------------
# Förstner on a (possibly resampled) window
# ---------------------------------------------------------------------------


def forstner_u(img: np.ndarray, c: np.ndarray, r: int = 2):
    """Förstner offset ``u`` (estimate - window centre) for windows centred at
    ``c`` (N, 2) [x, y], resampled bilinearly. Returns (u, ok)."""
    k = np.arange(-r - 1, r + 2)
    oy, ox = np.meshgrid(k, k, indexing="ij")
    ys = c[:, 1, None, None] + oy[None]
    xs = c[:, 0, None, None] + ox[None]
    p = ndi.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(ys.shape)
    gx = 0.5 * (p[:, 1:-1, 2:] - p[:, 1:-1, :-2])
    gy = 0.5 * (p[:, 2:, 1:-1] - p[:, :-2, 1:-1])
    ix, iy = ox[1:-1, 1:-1].astype(float), oy[1:-1, 1:-1].astype(float)
    w = 1.0 / (1.0 + 0.5 * (ix * ix + iy * iy))
    a00 = (w * gx * gx).sum((1, 2))
    a01 = (w * gx * gy).sum((1, 2))
    a11 = (w * gy * gy).sum((1, 2))
    bx = (w * (gx * gx * ix + gx * gy * iy)).sum((1, 2))
    by = (w * (gx * gy * ix + gy * gy * iy)).sum((1, 2))
    det = a00 * a11 - a01 * a01
    ok = det > 1e-6
    det = np.where(ok, det, 1.0)
    ux = (a11 * bx - a01 * by) / det
    uy = (-a01 * bx + a00 * by) / det
    return np.stack([ux, uy], 1), ok


def forstner_single(img: np.ndarray, seeds: np.ndarray):
    c = np.round(seeds)
    u, ok = forstner_u(img.astype(np.float64), c)
    ok &= np.all(np.abs(u) <= 1.5, axis=1)
    return (c + u)[ok]


def forstner_recentred(img: np.ndarray, seeds: np.ndarray, tol: float = 1e-3, max_it: int = 30, h: float = 0.02):
    """Returns (positions of converged corners, convergence mask over seeds)."""
    im = img.astype(np.float64)
    c0 = np.round(seeds).astype(np.float64)
    c = c0.copy()
    n = len(c)
    conv = np.zeros(n, bool)
    alive = np.ones(n, bool)
    u, ok = forstner_u(im, c)
    alive &= ok
    for _ in range(max_it):
        a = np.where(alive & ~conv)[0]
        if len(a) == 0:
            break
        done = np.linalg.norm(u[a], axis=1) < tol
        conv[a[done]] = True
        a = a[~done]
        if len(a) == 0:
            break
        ca, ua = c[a], u[a]
        jx = (forstner_u(im, ca + [h, 0])[0] - ua) / h
        jy = (forstner_u(im, ca + [0, h])[0] - ua) / h
        J = np.stack([jx, jy], 2)  # dF/dc, (n, 2, 2)
        detj = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        good = np.abs(detj) > 1e-9
        step = np.zeros_like(ua)
        Jg = J[good]
        step[good] = -np.linalg.solve(Jg, ua[good][..., None])[..., 0]
        step[~good] = ua[~good]  # plain fixed-point step as fallback
        lam = np.ones(len(a))
        accepted = np.zeros(len(a), bool)
        nu = ua.copy()
        nc = ca.copy()
        for _ in range(8):
            t = ~accepted
            if not t.any():
                break
            trial = ca[t] + lam[t, None] * step[t]
            ut, okt = forstner_u(im, trial)
            better = okt & (np.linalg.norm(ut, axis=1) < np.linalg.norm(ua[t], axis=1))
            idx = np.where(t)[0]
            nc[idx[better]] = trial[better]
            nu[idx[better]] = ut[better]
            accepted[idx[better]] = True
            lam[t] *= 0.5
        alive[a[~accepted]] = False
        c[a], u[a] = nc, nu
        far = np.any(np.abs(c[a] - c0[a]) > 1.5, axis=1)
        alive[a[far]] = False
    conv &= alive
    return c[conv], conv


def corner_subpix(img: np.ndarray, seeds: np.ndarray, half: int):
    if len(seeds) == 0:
        return seeds
    s0 = np.round(seeds).astype(np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    out = cv2.cornerSubPix(img, s0.copy(), (half, half), (-1, -1), crit).reshape(-1, 2).astype(np.float64)
    ok = np.all(np.abs(out - s0.reshape(-1, 2)) <= 1.5, axis=1)
    return out[ok]


# ---------------------------------------------------------------------------


def run_all(img: np.ndarray, names=ALL, wide: int = 4):
    """``{name: (xy (N, 2), cov (N, 2, 2) or None)}``."""
    res: dict = {}
    for n in ("CoM", "Forstner", "SaddlePoint"):
        if n in names or any(m in names for m in ("cSP-5", "cSP-w", "cSP-11", "RC-Forstner", "Forstner-py")):
            res[n] = chess(img, n)
    seeds = None
    if any(m in names for m in ("cSP-5", "cSP-w", "cSP-11", "RC-Forstner", "Forstner-py")):
        allp = np.concatenate([np.round(res[n][0]) for n in ("CoM", "Forstner", "SaddlePoint")])
        seeds = dedupe(np.unique(allp, axis=0))
        h, w = img.shape
        m = 8
        seeds = seeds[(seeds[:, 0] >= m) & (seeds[:, 0] < w - m) & (seeds[:, 1] >= m) & (seeds[:, 1] < h - m)]
    if "Radon" in names:
        res["Radon"] = chess(img, "Radon")
    if "cSP-5" in names:
        res["cSP-5"] = (corner_subpix(img, seeds, 2), None)
    if "cSP-w" in names:
        res["cSP-w"] = (corner_subpix(img, seeds, wide), None)
    if "cSP-11" in names:
        res["cSP-11"] = (corner_subpix(img, seeds, 5), None)
    if "RC-Forstner" in names:
        res["RC-Forstner"] = (forstner_recentred(img, seeds)[0], None)
    if "Forstner-py" in names:
        res["Forstner-py"] = (forstner_single(img, seeds), None)
    return {k: v for k, v in res.items() if k in names}


def match(xy: np.ndarray, truth: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Index into ``xy`` of the detection nearest each truth corner, or -1."""
    if len(xy) == 0:
        return np.full(len(truth), -1)
    from scipy.spatial import cKDTree

    d, j = cKDTree(xy).query(truth)
    return np.where(d <= radius, j, -1)
