"""UQ-26 gate on real calibration data: does weighting / correcting help?

Datasets and board IDs as ``../run_real.py``. Per view: SaddlePoint
(bindings, default config) matched within 2 px to the board-ID'd corners;
``core.methods`` at the image's estimated noise sigma (``estimate_noise_sigma``;
1/12 gray level^2 floor when it declines). Only corners valid for every
method are used.

Calibration: a weighted bundle adjustment (scipy least_squares, analytic
Jacobian from cv2.projectPoints) with residuals whitened per corner by
Sigma^-1/2, started from the uniform cv2.calibrateCamera solution. 2-fold
cross-validation by view parity: intrinsics from one fold, held-out views'
poses by uniform solvePnP on the same positions, held-out residual RMS
(per axis, pooled over both folds). Cases:

  uniform           positions c,         unweighted
  w:S_150           positions c,         weighted by the 1.5.0 covariance
  w:S_a_<cfg>       positions c,         weighted by Sigma + asymmetry term (a)
  w:S_c             positions c,         weighted by Sigma + multi-radius spread (c)
  w:<S>+m           as w:<S>, with Sigma + s^2 I (s: per-axis RMS residual of a
                    uniform calibration of the training fold; the board / model
                    error a consumer should add)
  corr:<cfg>        positions c - beta,  unweighted (b)
  corrg:<cfg>       positions corrected where the ramp is significant (b, gated)

Also: fold-to-fold change of fx and the principal point (stability), and the
Spearman correlation between each predicted sigma (sqrt(tr Sigma / 2)) and
the per-corner held-out residual of the uniform case.

Writes ``../out/asym_real.json`` and prints markdown tables.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import chess_corners as cc  # noqa: E402
import core  # noqa: E402
import estimators as est  # noqa: E402
import run_real as rr  # noqa: E402

ASYM = (("mul4", 4, "mul"), ("r2", 2, False))
WEIGHTS = ["S_150", "S_a_mul4", "S_a_r2", "S_c"]
CORR = ["mul4"]

# free parameters of [fx, fy, cx, cy, k1, k2, p1, p2, k3]
FREE = {"stereo_simple": [0, 1, 2, 3, 4, 5, 6, 7, 8], "table": [0, 1, 4, 5]}
CVFLAGS = {"stereo_simple": 0,
           "table": cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3}


def collect(files, ref_fn):
    views = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        r = ref_fn(img)
        if r is None:
            continue
        refxy, obj = r
        xy, _ = est.chess(img, "SaddlePoint")
        jr = est.match(xy, refxy, 2.0)  # for each ref, nearest detection
        m = jr >= 0
        sig = cc.estimate_noise_sigma(np.ascontiguousarray(img))
        sig = float(sig) if sig is not None else 0.0
        h, w = img.shape
        pos = xy[jr[m]]
        inb = (pos[:, 0] > core.KB + 3) & (pos[:, 0] < w - core.KB - 4) & (pos[:, 1] > core.KB + 3) & (pos[:, 1] < h - core.KB - 4)
        idx = np.where(m)[0][inb]
        out, valid = core.methods(img.astype(float), xy[jr[idx]], sig, ASYM)
        keep = valid
        v = dict(file=Path(f).name, sigma=sig, obj=obj[idx][keep], n=int(keep.sum()), nref=len(refxy))
        v["c"] = out["c"][keep]
        for k in WEIGHTS:
            v[k] = out[k][keep]
        for k in CORR:
            v["corr:" + k] = out[f"c_b_{k}"][keep]
            v["corrg:" + k] = out[f"c_bg_{k}"][keep]
        views.append(v)
    return views


def _whiten(S):
    """Upper factor U with U^T U = S^-1 per corner."""
    Si = np.linalg.inv(S)
    Lc = np.linalg.cholesky(Si)  # Si = L L^T -> U = L^T
    return np.swapaxes(Lc, 1, 2)


def bundle(objs, imgs, Ws, size, kind, k0):
    flags = CVFLAGS[kind]
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    K = None
    if k0 is not None:
        K = k0.copy()
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    _, K, D, rv, tv = cv2.calibrateCamera([o.astype(np.float32) for o in objs], [p.astype(np.float32) for p in imgs],
                                         size, K, None, flags=flags, criteria=crit)
    D = np.concatenate([D.ravel(), np.zeros(5)])[:5]
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D])
    free = FREE[kind]
    if Ws is None:
        return intr, [r.ravel() for r in rv], [t.ravel() for t in tv]
    nv = len(objs)
    x0 = np.concatenate([intr[free]] + [np.concatenate([r.ravel(), t.ravel()]) for r, t in zip(rv, tv)])
    nf = len(free)

    def unpack(x):
        it = intr.copy()
        it[free] = x[:nf]
        return it, x[nf:].reshape(nv, 6)

    def fun_jac(x, want_jac):
        it, poses = unpack(x)
        Kk = np.array([[it[0], 0, it[2]], [0, it[1], it[3]], [0, 0, 1]])
        Dd = it[4:9]
        res, rows = [], []
        for i in range(nv):
            p, jac = cv2.projectPoints(objs[i].astype(np.float64), poses[i, :3], poses[i, 3:], Kk, Dd)
            r = (imgs[i] - p.reshape(-1, 2))
            U = Ws[i]
            res.append(np.einsum("nij,nj->ni", U, r).ravel())
            if want_jac:
                jac = -jac.reshape(-1, 2, jac.shape[1])  # dr/dparam
                jw = np.einsum("nij,njp->nip", U, jac).reshape(-1, jac.shape[2])
                full = np.zeros((jw.shape[0], len(x)))
                # cv2 column order: rvec 3, tvec 3, fx fy, cx cy, dist...
                intr_cols = np.concatenate([jw[:, 6:10], jw[:, 10:15]], 1)
                full[:, :nf] = intr_cols[:, free]
                full[:, nf + 6 * i: nf + 6 * i + 6] = jw[:, :6]
                rows.append(full)
        if want_jac:
            return np.concatenate(rows)
        return np.concatenate(res)

    sol = least_squares(lambda x: fun_jac(x, False), x0, jac=lambda x: fun_jac(x, True), method="lm",
                        xtol=1e-12, ftol=1e-12, max_nfev=200)
    it, poses = unpack(sol.x)
    return it, [p[:3] for p in poses], [p[3:] for p in poses]


def kmat(it):
    return np.array([[it[0], 0, it[2]], [0, it[1], it[3]], [0, 0, 1]]), it[4:9]


def crossval(views, pos_key, w_key, size, kind, k0):
    objs = [v["obj"] for v in views]
    imgs = [v[pos_key] for v in views]
    # Uniform runs through the same bundle adjustment with identity weights.
    # A "+m" suffix adds the isotropic model/board error s^2 I to Sigma, with s
    # the per-axis RMS residual of a uniform calibration of the training fold.
    model = w_key is not None and w_key.endswith("+m")
    key = w_key[:-2] if model else w_key
    eye = [np.broadcast_to(np.eye(2), (len(v["obj"]), 2, 2)) for v in views]
    Ws = eye if key is None else [_whiten(v[key]) for v in views]
    sq, n = 0.0, 0
    per = [None] * len(views)
    intrs = []
    for fold in (0, 1):
        tr = [i for i in range(len(views)) if i % 2 == fold]
        te = [i for i in range(len(views)) if i % 2 != fold]
        Wtr = [Ws[i] for i in tr]
        if model:
            iu, rv, tv = bundle([objs[i] for i in tr], [imgs[i] for i in tr], [eye[i] for i in tr], size, kind, k0)
            Ku, Du = kmat(iu)
            res = np.concatenate([imgs[i] - cv2.projectPoints(objs[i], r_, t_, Ku, Du)[0].reshape(-1, 2)
                                  for i, r_, t_ in zip(tr, rv, tv)])
            s2 = float((res**2).mean())
            Wtr = [_whiten(views[i][key] + s2 * np.eye(2)) for i in tr]
        it, _, _ = bundle([objs[i] for i in tr], [imgs[i] for i in tr], Wtr, size, kind, k0)
        intrs.append(it)
        K, D = kmat(it)
        for i in te:
            ok, r, t = cv2.solvePnP(objs[i], imgs[i], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            r, t = cv2.solvePnPRefineLM(objs[i], imgs[i], K, D, r, t)
            p, _ = cv2.projectPoints(objs[i], r, t, K, D)
            e = imgs[i] - p.reshape(-1, 2)
            per[i] = e
            sq += (e**2).sum()
            n += e.size
    d = intrs[0] - intrs[1]
    return dict(cv=float(np.sqrt(sq / n)), dfx=float(abs(d[0])), dpp=float(np.hypot(d[2], d[3])),
                fx=float(0.5 * (intrs[0][0] + intrs[1][0]))), per


def main():
    sets = {
        "stereo L": (sorted(glob.glob(f"{rr.SS}/leftcamera/*.png")), rr.ref_stereo_simple, "stereo_simple", None),
        "stereo R": (sorted(glob.glob(f"{rr.SS}/rightcamera/*.png")), rr.ref_stereo_simple, "stereo_simple", None),
        "table cam1": (sorted(glob.glob(f"{rr.TB}/cam1/*.png")), rr.ref_table, "table", 21440.0),
        "table cam2": (sorted(glob.glob(f"{rr.TB}/cam2/*.png")), rr.ref_table, "table", 21343.0),
    }
    results = {}
    for label, (files, fn, kind, f0) in sets.items():
        views = collect(files, fn)
        h, w = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
        k0 = None if f0 is None else np.array([[f0, 0, (w - 1) / 2], [0, f0, (h - 1) / 2], [0, 0, 1]], float)
        r = dict(views=len(views), corners=int(sum(v["n"] for v in views)),
                 sigma_med=float(np.median([v["sigma"] for v in views])), cases={})
        cases = [("uniform", "c", None)] + [(f"w:{k}", "c", k) for k in WEIGHTS] + \
                [(f"w:{k}+m", "c", f"{k}+m") for k in ("S_150", "S_a_mul4", "S_c")] + \
                [(f"corr:{k}", f"corr:{k}", None) for k in CORR] + [(f"corrg:{k}", f"corrg:{k}", None) for k in CORR]
        per_u = None
        ss = {}
        for name, pk, wk in cases:
            out, per = crossval(views, pk, wk, (w, h), kind, k0)
            r["cases"][name] = out
            ss[name] = np.array([(e**2).sum() for e in per])
            if name == "uniform":
                per_u = per
        # Paired bootstrap over views of cv(case) - cv(uniform).
        cnt = np.array([e.size for e in per_u])
        rng = np.random.default_rng(0)
        B = rng.integers(0, len(views), size=(2000, len(views)))
        cu = np.sqrt(ss["uniform"][B].sum(1) / cnt[B].sum(1))
        for name in ss:
            cb = np.sqrt(ss[name][B].sum(1) / cnt[B].sum(1))
            lo, hi = np.percentile(cb - cu, [2.5, 97.5])
            r["cases"][name]["ci"] = [float(lo), float(hi)]
        # Spearman: predicted sigma vs held-out residual (uniform case).
        res_mag = np.concatenate([np.linalg.norm(e, axis=1) for e in per_u])
        r["spearman"] = {}
        for k in WEIGHTS:
            ps = np.concatenate([np.sqrt(0.5 * np.trace(v[k], axis1=1, axis2=2)) for v in views])
            rho, pv = spearmanr(ps, res_mag)
            r["spearman"][k] = [float(rho), float(pv), float(np.median(ps))]
        # correction magnitude
        for k in CORR:
            dd = np.concatenate([v[f"corr:{k}"] - v["c"] for v in views])
            r[f"corr_rms_{k}"] = float(np.sqrt((dd**2).sum(1).mean()))
            dg = np.concatenate([v[f"corrg:{k}"] - v["c"] for v in views])
            r[f"corrg_frac_{k}"] = float((np.abs(dg).sum(1) > 0).mean())
        r["resid_rms_uniform"] = float(np.sqrt((res_mag**2).mean() / 2))
        results[label] = r
        print(f"done {label}: {r['views']} views, {r['corners']} corners, sigma_med {r['sigma_med']:.2f}", flush=True)
    od = HERE.parent / "out"
    od.mkdir(exist_ok=True)
    (od / "asym_real.json").write_text(json.dumps(results, indent=1))
    names = list(next(iter(results.values()))["cases"].keys())
    print("\n### Held-out residual RMS per axis px (2-fold CV) [95% CI of the difference to uniform, view bootstrap]\n")
    print("| case | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for n in names:
        print(f"| {n} | " + " | ".join(f"{r['cases'][n]['cv']:.4f} [{r['cases'][n]['ci'][0]:+.4f}, {r['cases'][n]['ci'][1]:+.4f}]" for r in results.values()) + " |")
    print("\n### Fold-to-fold |dfx| px / |dpp| px\n")
    print("| case | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for n in names:
        print(f"| {n} | " + " | ".join(f"{r['cases'][n]['dfx']:.1f} / {r['cases'][n]['dpp']:.1f}" for r in results.values()) + " |")
    print("\n### Spearman(predicted sigma, held-out |residual|) [median predicted sigma px]\n")
    print("| Sigma | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for k in WEIGHTS:
        print(f"| {k} | " + " | ".join(f"{r['spearman'][k][0]:+.3f} [{r['spearman'][k][2]:.4f}]" for r in results.values()) + " |")
    for k in CORR:
        print(f"\ncorrection {k}: RMS shift " + ", ".join(f"{l} {r[f'corr_rms_{k}']:.4f}" for l, r in results.items())
              + "; gated fraction " + ", ".join(f"{r[f'corrg_frac_{k}']:.2f}" for r in results.values()))


if __name__ == "__main__":
    main()
