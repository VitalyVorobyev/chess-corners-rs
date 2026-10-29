"""Corner estimators compared by the robustness audit, all run on identical
uint8 images.

- ``CoM`` / ``SaddlePoint``: ChESS detector + built-in refiner
  (Python bindings, default configs, single scale, orientation off).
  SaddlePoint also returns its ``position_covariance`` (noise σ estimated per
  image, the default).
- ``Radon``: the Radon detector with its default peak fit.
- ``cSP-5``: ``cv2.cornerSubPix`` with ``winSize=(2, 2)`` (5x5, the same
  support as the built-in refiners). ``cSP-w``: ``winSize=(h, h)`` with ``h``
  chosen per dataset for the cell size (4 -> 9x9 on the ~10 px synthetic
  cells). ``cSP-11``: OpenCV's common ``winSize=(5, 5)`` (11x11), the Part IX
  setting.

Seeds for the OpenCV estimators: the rounded ChESS detections (union over
the ChESS refiners).
"""
from __future__ import annotations

import cv2
import numpy as np

import chess_corners as cc

# PRIMARY define the common evaluation set; DIAGNOSTIC estimators are scored
# on (common set) ∩ (their own matches), their failures counted in the match rate.
PRIMARY = ["CoM", "SaddlePoint", "Radon", "cSP-w"]
DIAGNOSTIC = ["cSP-5", "cSP-11"]
ALL = PRIMARY + DIAGNOSTIC

_DETS: dict = {}


def _det(name: str):
    if name not in _DETS:
        if name == "Radon":
            cfg = cc.DetectorConfig.radon().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
        else:
            cfg = cc.DetectorConfig.chess().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
            cfg.strategy.chess.refiner = {
                "CoM": cc.ChessRefiner.center_of_mass(),
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
    cfg = cc.DetectorConfig.chess().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED).with_noise_sigma(float(sigma))
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
    subpix = ("cSP-5", "cSP-w", "cSP-11")
    for n in ("CoM", "SaddlePoint"):
        if n in names or any(m in names for m in subpix):
            res[n] = chess(img, n)
    seeds = None
    if any(m in names for m in subpix):
        allp = np.concatenate([np.round(res[n][0]) for n in ("CoM", "SaddlePoint")])
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
    return {k: v for k, v in res.items() if k in names}


def match(xy: np.ndarray, truth: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Index into ``xy`` of the detection nearest each truth corner, or -1."""
    if len(xy) == 0:
        return np.full(len(truth), -1)
    from scipy.spatial import cKDTree

    d, j = cKDTree(xy).query(truth)
    return np.where(d <= radius, j, -1)
