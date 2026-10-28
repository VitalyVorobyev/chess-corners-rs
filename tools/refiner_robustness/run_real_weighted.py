"""Real-data check of the position covariance: does weighting a camera
calibration by each corner's reported covariance help?

Datasets (read-only): ``stereo_simple`` (20 views per camera, 11 x 7 inner
corners, 30 mm squares, board IDs from ``cv2.findChessboardCornersSB``) and
``table`` (ChArUco 22 x 22 squares, 1.35 mm, DICT_4X4_1000, non-legacy layout,
board IDs from ``cv2.aruco.CharucoDetector``). Per localizer (ChESS + CenterOfMass / Förstner / SaddlePoint,
the Radon peak fit, Radon + SaddlePoint; ``Uncertainty.PROPAGATED``,
noise sigma estimated per image), on the corners every localizer matched and
gave a covariance:

- Calibration (pinhole + k1 k2 p1 p2 k3): ``cv2.calibrateCamera`` for the
  initial estimate, then a least-squares refinement of intrinsics,
  distortion and every view's pose, either **uniform** (residual = observed -
  projected) or **weighted** (residual whitened per corner by ``L^-1`` with
  ``Sigma = L L'``, so the cost is ``sum r' Sigma^-1 r``). Both go through the
  same solver (``scipy.optimize.least_squares``, analytic Jacobian from
  ``cv2.projectPoints``), so only the weights differ.
- 2-fold cross-validation (even / odd views): calibrate on one fold, then fit
  each held-out view's pose (uniform ``solvePnP`` + LM refinement) with those
  intrinsics and take its reprojection residuals. Reported: pooled held-out
  RMS (px), |Δ fx| between the two folds' calibrations, and the RMS
  disagreement of the two folds' camera models (px) at the observed corners:
  where one fold's model reprojects the rays the other assigns to them.
- Spearman rank correlation between each corner's predicted sigma
  (``sqrt(tr Sigma / 2)``) and its held-out residual norm.
- sigma0^2 of the full weighted calibration (residuals whitened by Sigma,
  divided by ``2N - p``; includes board and model error).

Writes tools/refiner_robustness/out/real_weighted.json and prints a table.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chess_corners as cc  # noqa: E402
import estimators as est  # noqa: E402

NAMES = ["CoM", "Forstner", "SaddlePoint", "RadonPeak", "RadonSaddle"]
SS = "/Users/vitalyvorobyev/vision/calib-examples/data/stereo_simple/imgs"
TB = "/Users/vitalyvorobyev/vision/data/25_09_17_Table_Calibration"


def ref_stereo_simple(img):
    ok, c = cv2.findChessboardCornersSB(img, (11, 7))
    if not ok:
        return None
    c = c.reshape(-1, 2).astype(np.float64)
    gx, gy = np.meshgrid(np.arange(11), np.arange(7))
    obj = np.stack([gx.ravel() * 30.0, gy.ravel() * 30.0, np.zeros(77)], 1)
    return c, obj


_CH = None


def ref_table(img):
    global _CH
    if _CH is None:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        b = cv2.aruco.CharucoBoard((22, 22), 1.35, 1.35 * 0.75, d)
        b.setLegacyPattern(False)
        _CH = (b, cv2.aruco.CharucoDetector(b))
    b, det = _CH
    cc_, ids, _mc, _mi = det.detectBoard(img)
    if ids is None or len(ids) < 30:
        return None
    obj = b.getChessboardCorners()[ids.ravel()].astype(np.float64)
    return cc_.reshape(-1, 2).astype(np.float64), obj

_DETS: dict = {}


def detector(name: str):
    if name not in _DETS:
        if name.startswith("Radon"):
            cfg = cc.DetectorConfig.radon().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
            if name == "RadonSaddle":
                cfg.strategy.radon.refiner = cc.RadonRefiner.saddle_point()
        else:
            cfg = cc.DetectorConfig.chess().without_orientation().with_uncertainty(cc.Uncertainty.PROPAGATED)
            cfg.strategy.chess.refiner = {
                "CoM": cc.ChessRefiner.center_of_mass(),
                "Forstner": cc.ChessRefiner.forstner(),
                "SaddlePoint": cc.ChessRefiner.saddle_point(),
            }[name]
        _DETS[name] = cc.Detector(cfg)
    return _DETS[name]


def detect(img, name):
    d = detector(name).detect(img)
    return np.asarray(d.xy, np.float64), np.asarray(d.position_covariance, np.float64)


def collect(files, ref_fn):
    views = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        r = ref_fn(img)
        if r is None:
            continue
        refxy, obj = r
        res = {n: detect(img, n) for n in NAMES}
        idx = {n: est.match(res[n][0], refxy, 2.0) for n in NAMES}
        ok = np.all([idx[n] >= 0 for n in NAMES], axis=0)
        for n in NAMES:
            cov = res[n][1][np.where(ok, idx[n], 0)]
            ok &= ~np.isnan(cov).any(axis=(1, 2))
        if ok.sum() < 12:
            continue
        v = dict(obj=obj[ok])
        for n in NAMES:
            v[n] = res[n][0][idx[n][ok]]
            v[n + "_cov"] = res[n][1][idx[n][ok]]
        views.append(v)
    return views


# ---------------------------------------------------------------------------
# Calibration with per-corner whitening
# ---------------------------------------------------------------------------


def unpack(x, nv):
    fx, fy, cx, cy = x[:4]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    d = x[4:9].copy()
    poses = x[9:].reshape(nv, 6)
    return k, d, poses


def whiteners(covs):
    """L^-1 per corner for Sigma = L L' (identity when covs is None)."""
    if covs is None:
        return None
    lower = np.linalg.cholesky(covs)
    return np.linalg.inv(lower)


def refine(objs, imgs, covs, k0, d0, rv0, tv0):
    nv = len(objs)
    x0 = np.concatenate([[k0[0, 0], k0[1, 1], k0[0, 2], k0[1, 2]], d0.ravel()[:5],
                         np.concatenate([np.concatenate([r.ravel(), t.ravel()]) for r, t in zip(rv0, tv0)])])
    winv = [whiteners(c) if covs is not None else None for c in (covs or [None] * nv)]
    sizes = [len(o) for o in objs]
    offs = np.concatenate([[0], np.cumsum(sizes)])

    def project(x):
        k, d, poses = unpack(x, nv)
        out = []
        for i in range(nv):
            p, jac = cv2.projectPoints(objs[i], poses[i, :3], poses[i, 3:], k, d)
            out.append((p.reshape(-1, 2), jac))
        return out

    def fun(x):
        res = []
        for i, (p, _) in enumerate(project(x)):
            r = imgs[i] - p
            if winv[i] is not None:
                r = np.einsum("nij,nj->ni", winv[i], r)
            res.append(r.ravel())
        return np.concatenate(res)

    def jac(x):
        rows, cols, vals = [], [], []
        for i, (_, j) in enumerate(project(x)):
            n = sizes[i]
            # cv2 jacobian columns: rvec(3) tvec(3) f(2) c(2) dist(5)
            jp = np.concatenate([j[:, 6:8], j[:, 8:10], j[:, 10:15], j[:, 0:6]], axis=1)  # (2n, 15)
            jp = -jp.reshape(n, 2, 15)
            if winv[i] is not None:
                jp = np.einsum("nij,njk->nik", winv[i], jp)
            jp = jp.reshape(2 * n, 15)
            r0 = 2 * offs[i]
            colidx = np.concatenate([np.arange(9), 9 + 6 * i + np.arange(6)])
            rr, cc_ = np.meshgrid(np.arange(2 * n) + r0, colidx, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc_.ravel())
            vals.append(jp.ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(2 * offs[-1], len(x0)))

    sol = least_squares(fun, x0, jac=jac, method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    k, d, poses = unpack(sol.x, nv)
    return k, d, poses, sol.fun


def calibrate(views, name, size, weighted, f0=None):
    objs = [v["obj"].astype(np.float64) for v in views]
    imgs = [v[name] for v in views]
    k_init, flags = None, 0
    if f0 is not None:
        k_init = np.array([[f0, 0, (size[0] - 1) / 2], [0, f0, (size[1] - 1) / 2], [0, 0, 1.0]])
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
    _, k0, d0, rv, tv = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objs], [p.astype(np.float32) for p in imgs], size, k_init, None,
        flags=flags, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12))
    covs = [v[name + "_cov"] for v in views] if weighted else None
    return refine(objs, imgs, covs, k0, d0, rv, tv)


def held_out(views, name, k, d):
    out = []
    for v in views:
        obj = v["obj"].astype(np.float64)
        ok, r, t = cv2.solvePnP(obj, v[name], k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        r, t = cv2.solvePnPRefineLM(obj, v[name], k, d, r, t)
        p, _ = cv2.projectPoints(obj, r, t, k, d)
        out.append(v[name] - p.reshape(-1, 2))
    return out


def distortion_rms(k1, d1, k2, d2, pixels):
    """RMS (px) of where model 2 puts the rays model 1 assigns to `pixels`."""
    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    # Undistort with model 1, re-distort with model 2 through model-1 rays.
    n1 = cv2.undistortPoints(pts, k1, d1).reshape(-1, 2)
    obj = np.concatenate([n1, np.ones((len(n1), 1))], 1)
    p2, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), k2, d2)
    return float(np.sqrt(((p2.reshape(-1, 2) - pts.reshape(-1, 2)) ** 2).sum(1).mean()))


def evaluate(views, name, size, f0):
    out = {}
    folds = [[v for i, v in enumerate(views) if i % 2 == f] for f in (0, 1)]
    for mode in ("uniform", "weighted"):
        w = mode == "weighted"
        cal = [calibrate(folds[f], name, size, w, f0) for f in (0, 1)]
        resid, sig = [], []
        for f in (0, 1):
            k, d, _, _ = cal[f]
            test = folds[1 - f]
            for v, e in zip(test, held_out(test, name, k, d)):
                resid.append(e)
                c = v[name + "_cov"]
                sig.append(np.sqrt(0.5 * (c[:, 0, 0] + c[:, 1, 1])))
        r = np.concatenate(resid)
        s = np.concatenate(sig)
        rho = spearmanr(s, np.linalg.norm(r, axis=1)).statistic
        (ka, da, _, _), (kb, db, _, _) = cal
        out[mode] = dict(
            cv=float(np.sqrt((r**2).sum(1).mean() / 2)),
            dfx=float(abs(ka[0, 0] - kb[0, 0])),
            ddist=distortion_rms(ka, da, kb, db, np.concatenate([v[name] for v in views])),
            spearman=float(rho),
        )
    # sigma0^2 of the full weighted calibration
    k, d, poses, fun = calibrate(views, name, size, True, f0)
    n = sum(len(v["obj"]) for v in views)
    p = 9 + 6 * len(views)
    out["s0"] = float((fun**2).sum() / (2 * n - p))
    out["pred_sigma_med"] = float(np.median(np.concatenate(
        [np.sqrt(0.5 * (v[name + "_cov"][:, 0, 0] + v[name + "_cov"][:, 1, 1])) for v in views])))
    return out


def main() -> None:
    sets = {
        "stereo L": (sorted(glob.glob(f"{SS}/leftcamera/*.png")), ref_stereo_simple, None),
        "stereo R": (sorted(glob.glob(f"{SS}/rightcamera/*.png")), ref_stereo_simple, None),
        "table cam1": (sorted(glob.glob(f"{TB}/cam1/*.png")), ref_table, 21440.0),
        "table cam2": (sorted(glob.glob(f"{TB}/cam2/*.png")), ref_table, 21343.0),
    }
    only = sys.argv[1:]
    results = {}
    for label, (files, fn, f0) in sets.items():
        if only and label not in only:
            continue
        views = collect(files, fn)
        h, w = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
        r = dict(views=len(views), corners=int(sum(len(v["obj"]) for v in views)))
        for n in NAMES:
            r[n] = evaluate(views, n, (w, h), f0)
            print(f"{label} {n}: {r[n]}", flush=True)
        results[label] = r
    od = HERE / "out"
    od.mkdir(exist_ok=True)
    (od / "real_weighted.json").write_text(json.dumps(results, indent=1))
    print()
    print("| set | localizer | held-out RMS uniform / weighted | |Δfx| u / w | Δdist RMS u / w | Spearman(σ, |e|) | σ̂₀² (weighted) | median σ |")
    print("|---|---|---|---|---|---|---|---|")
    for label, r in results.items():
        for n in NAMES:
            m = r[n]
            u, wt = m["uniform"], m["weighted"]
            print(f"| {label} ({r['views']} views, {r['corners']}) | {n} | {u['cv']:.4f} / {wt['cv']:.4f} | "
                  f"{u['dfx']:.2f} / {wt['dfx']:.2f} | {u['ddist']:.3f} / {wt['ddist']:.3f} | {u['spearman']:.2f} | "
                  f"{m['s0']:.1f} | {m['pred_sigma_med']:.4f} |")


if __name__ == "__main__":
    main()
