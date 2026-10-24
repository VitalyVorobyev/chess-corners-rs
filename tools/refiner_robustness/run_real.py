"""Real-data comparison: camera calibration residuals per corner estimator.

Datasets (read-only):
- ``stereo_simple``: calib-examples/data/stereo_simple/imgs/{left,right}camera,
  20 views each, 11 x 7 inner corners, 30 mm squares, 1024 x 576 colour images
  (converted to gray). Board IDs from ``cv2.findChessboardCornersSB``.
- ``table``: data/25_09_17_Table_Calibration/cam{1,2}, 28 / 27 views,
  2048 x 1536 mono, ChArUco 22 x 22 squares, 1.35 mm, marker 0.75,
  DICT_4X4_1000 (non-legacy layout). Board IDs from ``cv2.aruco.CharucoDetector``.

Every estimator's detections are matched (within 2 px) to the board-ID'd
reference corners, and only corners matched by *every* estimator in that view
are used, so all estimators calibrate on the same corners and views. Per
estimator and distortion model:
  rms   = cv2.calibrateCamera RMS reprojection error over all views
  cv    = 2-fold cross-validated residual: calibrate on the even (odd) views,
          then solvePnP + reproject each held-out view with those intrinsics;
          pooled RMS over both folds
SaddlePoint also gets sigma0^2 = sum r' Sigma^-1 r / (2N - p) of the full
calibration residuals whitened by its reported covariance (model error
included).

Writes tools/refiner_robustness/out/real.json and prints markdown tables.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import estimators as est  # noqa: E402

NAMES = ["CoM", "Forstner", "SaddlePoint", "Radon", "cSP-11", "cSP-5", "RC-Forstner"]
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


def collect(files, ref_fn, wide):
    views = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        r = ref_fn(img)
        if r is None:
            continue
        refxy, obj = r
        res = est.run_all(img, [n if n != "cSP-11" else "cSP-w" for n in NAMES], wide=wide)
        res["cSP-11"] = res.pop("cSP-w")
        idx = {n: est.match(res[n][0], refxy, 2.0) for n in NAMES}
        common = np.all([idx[n] >= 0 for n in NAMES], axis=0)
        v = dict(file=Path(f).name, obj=obj[common], nref=len(refxy), ncommon=int(common.sum()))
        v["match"] = {n: float((idx[n] >= 0).mean()) for n in NAMES}
        for n in NAMES:
            v[n] = res[n][0][idx[n][common]]
        cov = res["SaddlePoint"][1][idx["SaddlePoint"][common]]
        v["cov"] = cov
        views.append(v)
    return views


FLAGS = {
    "stereo_simple": {
        "k1k2k3p1p2": 0,
        "rational": cv2.CALIB_RATIONAL_MODEL,
    },
    "table": {
        "k1k2 (pp fixed)": cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
        "k1k2k3p1p2": 0,
    },
}


def calib(objs, imgs, size, flags, k0):
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    f = flags
    K = None
    if k0 is not None:
        K = k0.copy()
        f |= cv2.CALIB_USE_INTRINSIC_GUESS
    rms, K, D, rv, tv = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objs], [p.astype(np.float32) for p in imgs], size, K, None, flags=f, criteria=crit
    )
    return rms, K, D, rv, tv


def residuals(obj, img, K, D, rv, tv):
    p, _ = cv2.projectPoints(obj.astype(np.float64), rv, tv, K, D)
    return img - p.reshape(-1, 2)


def evaluate(views, name, size, flags, k0):
    objs = [v["obj"] for v in views]
    imgs = [v[name] for v in views]
    rms, K, D, rv, tv = calib(objs, imgs, size, flags, k0)
    res = [residuals(o, i, K, D, r, t) for o, i, r, t in zip(objs, imgs, rv, tv)]
    R = np.concatenate(res)
    out = dict(rms=float(rms), fx=float(K[0, 0]), dist=D.ravel().tolist())
    # 2-fold CV
    sq, n = 0.0, 0
    for fold in (0, 1):
        tr = [i for i in range(len(views)) if i % 2 == fold]
        te = [i for i in range(len(views)) if i % 2 != fold]
        _, Kt, Dt, _, _ = calib([objs[i] for i in tr], [imgs[i] for i in tr], size, flags, k0)
        for i in te:
            ok, r, t = cv2.solvePnP(objs[i], imgs[i], Kt, Dt, flags=cv2.SOLVEPNP_ITERATIVE)
            r, t = cv2.solvePnPRefineLM(objs[i], imgs[i], Kt, Dt, r, t)
            e = residuals(objs[i], imgs[i], Kt, Dt, r, t)
            sq += (e**2).sum()
            n += len(e)
    out["cv"] = float(np.sqrt(sq / n))
    if name == "SaddlePoint":
        covs = np.concatenate([v["cov"] for v in views])
        has = ~np.isnan(covs).any(axis=(1, 2))
        q = np.einsum("ni,nij,nj->n", R[has], np.linalg.inv(covs[has]), R[has])
        nparam = 4 + len(D.ravel()) + 6 * len(views)
        out["s0"] = float(q.sum() / (2 * has.sum() - nparam * has.mean()))
        out["cov_frac"] = float(has.mean())
        ps = np.sqrt(0.5 * (covs[has, 0, 0] + covs[has, 1, 1]))
        out["pred_sigma_med"] = float(np.median(ps))
        # Does the predicted sigma track the residual? Residual RMS per axis by
        # predicted-sigma quartile.
        qs = np.quantile(ps, [0, 0.25, 0.5, 0.75, 1.0])
        rr = R[has]
        bins = []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (ps >= a) & (ps <= b)
            bins.append([float(np.median(ps[m])), float(np.sqrt((rr[m] ** 2).mean()))])
        out["sigma_bins"] = bins
    return out, R


def hat(P):
    """Per-axis independent-error variances from pairwise differences."""
    rows, rhs = [], []
    for a in range(len(NAMES)):
        for b in range(a + 1, len(NAMES)):
            row = np.zeros(len(NAMES))
            row[a] = row[b] = 1
            rows.append(row)
            rhs.append(((P[NAMES[a]] - P[NAMES[b]]) ** 2).mean())
    return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]


def main() -> None:
    sets = {
        "stereo_simple L": (sorted(glob.glob(f"{SS}/leftcamera/*.png")), ref_stereo_simple, 5, "stereo_simple", None),
        "stereo_simple R": (sorted(glob.glob(f"{SS}/rightcamera/*.png")), ref_stereo_simple, 5, "stereo_simple", None),
        "table cam1": (sorted(glob.glob(f"{TB}/cam1/*.png")), ref_table, 5, "table", 21440.0),
        "table cam2": (sorted(glob.glob(f"{TB}/cam2/*.png")), ref_table, 5, "table", 21343.0),
    }
    results = {}
    for label, (files, fn, wide, kind, f0) in sets.items():
        views = collect(files, fn, wide)
        h, w = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
        k0 = None
        if f0 is not None:
            k0 = np.array([[f0, 0, (w - 1) / 2], [0, f0, (h - 1) / 2], [0, 0, 1]], float)
        r = dict(
            views=len(views),
            corners=int(sum(v["ncommon"] for v in views)),
            ref=int(sum(v["nref"] for v in views)),
            match={n: float(np.mean([v["match"][n] for v in views])) for n in NAMES},
            models={},
        )
        # Pairwise disagreement with SaddlePoint (px RMS) on the common set.
        sp = np.concatenate([v["SaddlePoint"] for v in views])
        r["vs_sp"] = {n: float(np.sqrt(((np.concatenate([v[n] for v in views]) - sp) ** 2).sum(1).mean())) for n in NAMES}
        # N-cornered hat: if estimator errors are independent, the mean squared
        # pairwise difference D_ij = v_i + v_j; least squares for v (per-axis
        # variance). Positive error correlation biases v low.
        P = {n: np.concatenate([v[n] for v in views]) for n in NAMES}
        rows, rhs = [], []
        for a in range(len(NAMES)):
            for b in range(a + 1, len(NAMES)):
                row = np.zeros(len(NAMES))
                row[a] = row[b] = 1
                rows.append(row)
                rhs.append(((P[NAMES[a]] - P[NAMES[b]]) ** 2).mean())
        vv = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        r["hat_sigma"] = {n: float(np.sqrt(max(x, 0.0))) for n, x in zip(NAMES, vv)}
        for model, flags in FLAGS[kind].items():
            r["models"][model] = {}
            resid = {}
            for n in NAMES:
                out, resid[n] = evaluate(views, n, (w, h), flags, k0)
                r["models"][model][n] = out
            # N-cornered hat on the calibration residuals: each residual is
            # r_i = m + e_i (m = model / board error common to all
            # estimators, e_i = the estimator's non-absorbed localisation
            # error). If the e_i are independent, mean |r_i - r_j|^2 / 2 =
            # v_i + v_j (per axis) and mean |r_i|^2 / 2 = M + v_i.
            hv = hat(resid)
            r["models"][model]["_hat"] = {n: float(np.sqrt(max(x, 0.0))) for n, x in zip(NAMES, hv)}
            msq = [((resid[n] ** 2).mean()) - x for n, x in zip(NAMES, hv)]
            r["models"][model]["_common"] = float(np.sqrt(max(np.median(msq), 0.0)))
        results[label] = r
        print(f"done {label}: {len(views)} views, {r['corners']} common corners of {r['ref']}", flush=True)
    od = HERE / "out"
    od.mkdir(exist_ok=True)
    (od / "real.json").write_text(json.dumps(results, indent=1))
    for label, r in results.items():
        print(f"\n### {label}: {r['views']} views, {r['corners']} corners (common to all estimators, of {r['ref']} ID'd)\n")
        print("| model | " + " | ".join(NAMES) + " |")
        print("|---|" + "---|" * len(NAMES))
        print("| match rate | " + " | ".join(f"{r['match'][n]:.3f}" for n in NAMES) + " |")
        print("| RMS diff vs SaddlePoint | " + " | ".join(f"{r['vs_sp'][n]:.3f}" for n in NAMES) + " |")
        print("| N-cornered-hat sigma/axis | " + " | ".join(f"{r['hat_sigma'][n]:.3f}" for n in NAMES) + " |")
        for model, m in r["models"].items():
            print(f"| {model}: rms / cv | " + " | ".join(f"{m[n]['rms']:.4f} / {m[n]['cv']:.4f}" for n in NAMES) + " |")
        for model, m in r["models"].items():
            print(f"| {model}: residual-hat sigma/axis (common {m['_common']:.3f}) | " + " | ".join(f"{m['_hat'][n]:.3f}" for n in NAMES) + " |")
        for model, m in r["models"].items():
            sp = m["SaddlePoint"]
            print(f"\nSaddlePoint {model}: sigma0^2 {sp['s0']:.2f}, cov frac {sp['cov_frac']:.2f}, median predicted sigma {sp['pred_sigma_med']:.4f} px, fx {sp['fx']:.1f}")
            print("  residual RMS/axis by predicted-sigma quartile (median pred sigma -> resid): " + ", ".join(f"{a:.4f}->{b:.3f}" for a, b in sp["sigma_bins"]))


if __name__ == "__main__":
    main()
