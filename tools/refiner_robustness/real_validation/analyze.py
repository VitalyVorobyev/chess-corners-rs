"""Stage 3: residual decomposition, stereo consistency, covariance validation.

Usage: python analyze.py [--set all|chess]

Reads ``out/calib_<set>.pkl`` (``calibrate.py``), prints markdown tables and
writes ``out/summary_<set>.json``.

Decomposition of the rig-model residual ``r`` (px) of every observation:

    r = f_cam(u) + J_b beta_id + e

- ``f_cam`` (smooth): per camera, a tensor cubic B-spline over the image
  (``--knots`` basis functions per axis; 5 x 4, ~1000 px knot spacing, gives
  the lowest cross-fold remainder of 5x4 .. 16x12) for each
  residual component. It captures lens-model error *and* any smooth target
  deformation: a corner id sits within ~60 px of the same image position in
  every view, so smooth board and smooth field terms are not separable here.
- ``beta_id`` (board): per ChArUco corner id, the mean board-frame residual
  (``J_b`` = d pixel / d board-plane mm of that observation), pooled over
  views and cameras, after the smooth term; the per-corner part of the
  target / appearance error. Ids seen fewer than ``MIN_OBS`` times get 0.
- ``e`` (remainder): what is left; it carries the localizer's view-dependent
  error (noise, illumination, blur) and pose / stage effects.

Fitted alternately (smooth first, two rounds). ``board-first`` reverses the
order to show how much of the smooth part a per-id term would take.
``xfold``: components estimated on one fold of views, subtracted from the
other fold (out-of-sample remainder).

ChArUco neighbourhood: for each corner, the antisymmetric first moment of the
ideal board pattern (render) within radius R squares; a localizer that sees
marker bits would have a board component correlated with it.

Covariance: any localizer whose observations carry a covariance (bindings or
``calibrate.py --cov``) gets sigma0^2 = mean(e' S^-1 e) / 2 and the Spearman
rank correlation between predicted sigma = sqrt(tr S / 2) and |e|, for the
raw residual, the in-sample and the cross-fold remainder.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

import calibrate as K
import common as C

MIN_OBS = 6
# Localizers left out of the common/specific split: Radon+SP runs the
# SaddlePoint refiner, so its error is not independent of SaddlePoint's.
HAT_EXCLUDE = {"Radon+SP"}


def rms(r):
    r = r[np.isfinite(r[:, 0])]
    return float(np.sqrt((r**2).sum(1).mean())) if len(r) else float("nan")


def bspline_basis(x, lo, hi, nb, k=3):
    t = np.r_[[lo] * k, np.linspace(lo, hi, nb - k + 1), [hi] * k]
    x = np.clip(x, lo, hi - 1e-9)
    return BSpline.design_matrix(x, t, k).toarray()


class Smooth:
    def __init__(self, shape, nb):
        self.h, self.w = shape
        self.nb = nb

    def basis(self, u):
        bx = bspline_basis(u[:, 0], -0.5, self.w - 0.5, self.nb[0])
        by = bspline_basis(u[:, 1], -0.5, self.h - 0.5, self.nb[1])
        return np.einsum("ni,nj->nij", bx, by).reshape(len(u), -1)

    def fit(self, u, cam, r, fit_mask):
        """Per-camera smooth field; returns a predictor."""
        coef = {}
        for c in (0, 1):
            m = fit_mask & (cam == c)
            if not m.any():
                coef[c] = None
                continue
            B = self.basis(u[m])
            lam = 1e-6 * np.trace(B.T @ B) / B.shape[1]
            coef[c] = np.linalg.solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ r[m])

        def predict(uu, cc):
            out = np.zeros((len(uu), 2))
            for c in (0, 1):
                if coef[c] is not None and (cc == c).any():
                    out[cc == c] = self.basis(uu[cc == c]) @ coef[c]
            return out

        return predict


def board_term(ids, Jb, r, fit_mask, nid):
    """Per-id least-squares board displacement (mm, ``Jb.shape[2]`` axes:
    in-plane x, y and optionally out-of-plane z) -> beta (nid, d)."""
    d = Jb.shape[2]
    A = np.zeros((nid, d, d))
    g = np.zeros((nid, d))
    m = fit_mask
    np.add.at(A, ids[m], np.einsum("nki,nkj->nij", Jb[m], Jb[m]))
    np.add.at(g, ids[m], np.einsum("nki,nk->ni", Jb[m], r[m]))
    ok = np.bincount(ids[m], minlength=nid) >= MIN_OBS
    beta = np.zeros((nid, d))
    beta[ok] = np.linalg.solve(A[ok], g[ok][..., None])[..., 0]
    return beta, ok


def xfold_remainder(o, r, Jb, fold, nb, dims=3):
    """Remainder of each fold with the smooth and board terms estimated on
    the other fold."""
    out = np.full_like(r, np.nan)
    for f in np.unique(fold):
        out[fold == f] = decompose(o, r, Jb, fold != f, nb, dims=dims)["rem"][fold == f]
    return out


def decompose(o, r, Jb, fit_mask, nb, order="smooth", dims=3):
    Jb = Jb[:, :, :dims]
    u, cam, ids = o["P"], o["cam"], o["id"]
    nid = (C.NSQ - 1) ** 2
    sm = Smooth(o["shape"], nb)
    f = np.zeros_like(r)
    bpx = np.zeros_like(r)
    beta = None
    for it in range(2):
        if order == "smooth" or it > 0:
            pred = sm.fit(u, cam, r - bpx, fit_mask)
            f = pred(u, cam)
        beta, ok = board_term(ids, Jb, r - f, fit_mask, nid)
        bpx = np.einsum("nij,nj->ni", Jb, beta[ids])
        if order != "smooth" and it == 0:
            pred = sm.fit(u, cam, r - bpx, fit_mask)
            f = pred(u, cam)
    return dict(smooth=f, board=bpx, rem=r - f - bpx, beta=beta, beta_ok=ok)


# ---------------------------------------------------------------------------
# ChArUco neighbourhood
# ---------------------------------------------------------------------------


def neighbourhood_moments(radii=(0.5, 0.75, 1.0), S=80):
    """(nid, len(radii), 2) antisymmetric first moment of the ideal pattern
    (white = 1) about each corner, in squares, board axes."""
    img = C.board().generateImage((C.NSQ * S, C.NSQ * S), marginSize=0, borderBits=1).astype(np.float64) / 255.0
    X = C.board_points()[:, :2] / C.SQUARE * S  # px (pixel-edge coordinates)
    out = np.zeros((len(X), len(radii), 2))
    R = int(max(radii) * S)
    g = np.arange(-R, R) + 0.5
    gx, gy = np.meshgrid(g, g)
    for i, (cx, cy) in enumerate(X):
        xs = np.clip((cx + gx).astype(int), 0, img.shape[1] - 1)
        ys = np.clip((cy + gy).astype(int), 0, img.shape[0] - 1)
        p = img[ys, xs]
        a = p - p[::-1, ::-1]
        rr = np.hypot(gx, gy)
        for k, rad in enumerate(radii):
            m = rr <= rad * S
            out[i, k] = [(a * gx)[m].sum() / m.sum() / S, (a * gy)[m].sum() / m.sum() / S]
    # sanity: the 4 squares around each corner alternate (checked on corner 0)
    return out


def corr2(a, b):
    """Fraction of variance of a (n, 2) explained by a linear map of b (n, 2)."""
    A = np.c_[b, np.ones(len(b))]
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    res = a - A @ coef
    return float(1 - (res**2).sum() / ((a - a.mean(0)) ** 2).sum())


# ---------------------------------------------------------------------------


def per_view_smooth(o, P, e, deg=3):
    """How much of the remainder is a smooth pattern within each view and
    camera (a 2-D polynomial of degree ``deg`` in image coordinates, fitted per
    view): returns (RMS of the fitted part, RMS expected from white noise of
    the same size)."""
    fit = np.zeros_like(e)
    dof_row = np.zeros(len(e))
    frac = []
    for c in (0, 1):
        for v in np.unique(o["view"]):
            s = np.where((o["cam"] == c) & (o["view"] == v))[0]
            if len(s) < 30:
                continue
            x = (P[s, 0] - 1024) / 1024
            y = (P[s, 1] - 768) / 1024
            A = np.stack([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)
            coef, *_ = np.linalg.lstsq(A, e[s], rcond=None)
            fit[s] = A @ coef
            dof_row[s] = A.shape[1] / len(s)
            frac.append((A.shape[1], len(s)))
    k = np.array(frac, float)
    # white noise: a k-term fit to n points captures k / n of its energy
    white = np.sqrt(k[:, 0].sum() / k[:, 1].sum())
    # dof_row: per observation, its view's share k / n; its mean over any set
    # of whole views is that set's dof fraction.
    return dict(fitted=rms(fit), white_expected=float(white * rms(e)), fit=fit, dof_frac=float(k[:, 0].sum() / k[:, 1].sum()), dof_row=dof_row)


def local_remainder(oo, o, r, Jb, fit_mask, nb):
    """Local remainder (smooth and board terms fitted on ``fit_mask``, then
    the per-view smooth pattern removed) and each observation's per-view
    dof share (``per_view_smooth``)."""
    dec = decompose(oo, r, Jb, fit_mask, nb)
    pvs = per_view_smooth(o, oo["P"], dec["rem"])
    return dec["rem"] - pvs["fit"], pvs["dof_row"]


def rem_by_tilt(o, e):
    """Remainder RMS by view class: the two Z-step views, and the larger
    commanded stage angle (0, 9, 14 deg) of the others."""
    st = o["stage"][o["view"]]
    t = np.max(np.abs(st[:, :2]), 1)
    z = st[:, 4] != 0
    out = {"z-step": rms(e[z])}
    out.update({f"tilt {int(lv)}": rms(e[~z & (t == lv)]) for lv in (0.0, 9.0, 14.0)})
    return out


def stereo_scatter(T21: dict):
    Ts = list(T21.values())
    Rm = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean()
    ang = np.array([(Rm.inv() * Rotation.from_matrix(T[:3, :3])).magnitude() for T in Ts]) * 1e3
    t = np.stack([T[:3, 3] for T in Ts])
    dt = (t - t.mean(0)) * 1e3
    return dict(rot_mrad=float(np.sqrt((ang**2).mean())), t_um=float(np.sqrt((dt**2).sum(1).mean())), t_um_xyz=np.sqrt((dt**2).mean(0)).tolist(), n=len(Ts))


def cov_stats(e, S):
    ok = np.isfinite(S).all(axis=(1, 2)) & np.isfinite(e[:, 0])
    ps = np.sqrt(0.5 * (S[ok, 0, 0] + S[ok, 1, 1]))
    return dict(s0=C.sigma0_sq(e[ok], S[ok]), spearman=float(spearmanr(ps, np.linalg.norm(e[ok], axis=1))[0]), pred_sigma_med=float(np.median(ps)), frac=float(ok.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--knots", default="5,4", help="cubic B-spline basis functions per axis (x,y), >= 4")
    a = ap.parse_args()
    nb = tuple(int(x) for x in a.knots.split(","))
    d = pickle.load(open(C.OUT / f"calib_{a.set}.pkl", "rb"))
    o = d["obs"]
    locs = list(d["locs"])
    fold = o["fold"][o["view"]]
    allm = np.ones(len(o["cam"]), bool)
    mom = neighbourhood_moments()
    summ = dict(set=a.set, n_obs=int(len(o["cam"])), n_views=len(o["tokens"]), locs={})

    # Stage-model check (SaddlePoint poses)
    if "stage_check" in d:
        sc = d["stage_check"]
        print("\n### Stage model fit to rig poses (pose space, px through cam1)\n")
        print("| model | pose params | RMS px |\n|---|---|---|")
        print(f"| free per view (rig) | {6 * len(o['tokens'])} | 0 (reference) |")
        for k in ("rigid", "scale", "rx_level_free", "ry_level_free"):
            print(f"| {k} | {sc[k][0]} | {sc[k][1]:.2f} |")
        D = sc["discrepancy"]
        print("\nRigid-model pose discrepancy by Rx level (mean over views; rot mrad x/y/z, trans um x/y/z at board centre):\n")
        print("| Rx level | views | rot mrad | trans um | within-level trans scatter um |\n|---|---|---|---|---|")
        for lv in range(5):
            m = (sc["rx_level"] == lv) & ~sc["is_z"]
            if m.any():
                mu = D[m].mean(0)
                sd = np.sqrt(((D[m, 3:5] - mu[3:5]) ** 2).sum(1).mean())
                print(f"| {C.LEVELS[lv]:+.0f} | {m.sum()} | {np.round(mu[:3], 2).tolist()} | {np.round(mu[3:], 0).tolist()} | {sd:.0f} |")
        summ["stage_check"] = {k: sc[k] for k in ("rigid", "scale", "rx_level_free", "ry_level_free")}

    rows = []
    betas = {}
    rems = {}
    for loc in locs:
        L = d["locs"][loc]
        P = o["xy"][loc]
        oo = dict(P=P, cam=o["cam"], id=o["id"], shape=o["shape"])
        rig = L["rig"]
        r = rig["res"]
        Jb = rig["Jb"]
        dec = decompose(oo, r, Jb, allm, nb)
        bf = decompose(oo, r, Jb, allm, nb, order="board")
        dec2 = decompose(oo, r, Jb, allm, nb, dims=2)
        # cross-fold remainder: components from the other fold's views
        xrem = np.full_like(r, np.nan)
        xrem2 = np.full_like(r, np.nan)
        fb = []
        for f in (0, 1):
            dd = decompose(oo, r, Jb, fold != f, nb)
            xrem[fold == f] = dd["rem"][fold == f]
            xrem2[fold == f] = decompose(oo, r, Jb, fold != f, nb, dims=2)["rem"][fold == f]
            fb.append(dd)
        okb = fb[0]["beta_ok"] & fb[1]["beta_ok"]
        beta_xfold = corr2(fb[0]["beta"][okb, :2], fb[1]["beta"][okb, :2])
        z0, z1 = fb[0]["beta"][okb, 2], fb[1]["beta"][okb, 2]
        betaz_xfold = float(np.corrcoef(z0, z1)[0, 1])
        # target vs field: board terms from each camera alone (same ids, image
        # positions ~100 px apart between the cameras)
        bc = [decompose(oo, r, Jb, o["cam"] == c, nb) for c in (0, 1)]
        okc = bc[0]["beta_ok"] & bc[1]["beta_ok"]
        beta_cam_ve = corr2(bc[0]["beta"][okc, :2], bc[1]["beta"][okc, :2])
        # stereo consistency on observations with the smooth + board terms removed
        st_clean = stereo_scatter(K.stereo_poses(o, P - dec["smooth"] - dec["board"], rig["p"]))
        cvdec = decompose(oo, rig["cv"], Jb, allm, nb)
        st = stereo_scatter(L["stereo"])
        row = dict(
            free_fit=rms(L["free"]["res"]),
            free_cv=rms(L["free"]["cv"]),
            rig_fit=rms(r),
            rig_cv=rms(rig["cv"]),
            stage_fit=rms(L["stage"]["res"]) if "stage" in L else None,
            stage_pred=rms(L["stage"]["cv_pred"]) if "stage" in L else None,
            smooth=rms(dec["smooth"]),
            board=rms(dec["board"]),
            rem=rms(dec["rem"]),
            rem_xfold=rms(xrem),
            rem_cv=rms(cvdec["rem"]),
            rem_2d=rms(dec2["rem"]),
            rem_2d_xfold=rms(xrem2),
            board_2d=rms(dec2["board"]),
            bf_board=rms(bf["board"]),
            bf_smooth=rms(bf["smooth"]),
            bf_rem=rms(bf["rem"]),
            stereo=st,
            stereo_clean=st_clean,
            beta_xfold_ve=beta_xfold,
            betaz_xfold_r=betaz_xfold,
            beta_cam_ve=beta_cam_ve,
            beta_um=float(np.sqrt((dec["beta"][dec["beta_ok"], :2] ** 2).sum(1).mean()) * 1e3),
            beta_z_um=float(np.sqrt((dec["beta"][dec["beta_ok"], 2] ** 2).mean()) * 1e3),
        )
        betas[loc] = (dec["beta"][:, :2], dec["beta_ok"])
        pvs = per_view_smooth(o, P, dec["rem"])
        local = dec["rem"] - pvs.pop("fit")
        pvs.pop("dof_row")
        rems[loc] = (dec["rem"], xrem, local)
        row["local"] = rms(local)
        # neighbourhood correlation of the in-plane board component
        ok = dec["beta_ok"]
        row["nbhd_r2"] = [corr2(dec["beta"][ok, :2], mom[ok, k]) for k in range(mom.shape[1])]
        row["per_view_smooth"] = pvs
        row["rem_by_tilt"] = rem_by_tilt(o, dec["rem"])
        S = o["cov"][loc] if loc in o["cov"] else None
        if S is not None and np.isfinite(S).all(axis=(1, 2)).mean() > 0.5:
            row["cov"] = dict(raw=cov_stats(r, S), rem=cov_stats(dec["rem"], S), rem_xfold=cov_stats(xrem, S), local=cov_stats(local, S))
            if "rig_w" in L:
                rw = L["rig_w"]
                xw = xfold_remainder(oo, rw["res"], rw["Jb"], fold, nb)
                row["weighted"] = dict(fit=rms(rw["res"]), cv=rms(rw["cv"]), rem_xfold=rms(xw))
            row["pred_var_mean"] = float(np.nanmean(0.5 * (S[:, 0, 0] + S[:, 1, 1])))
        summ["locs"][loc] = row
        rows.append((loc, row))

    print(f"\n### Calibration residual per localizer ({summ['n_obs']} observations, {summ['n_views']} views x 2 cameras), RMS px per point\n")
    print("| localizer | free fit / CV | rig fit / CV | stage fit / pred |\n|---|---|---|---|")
    for loc, r in rows:
        stg = f"{r['stage_fit']:.3f} / {r['stage_pred']:.3f}" if r["stage_fit"] is not None else "-"
        print(f"| {loc} | {r['free_fit']:.4f} / {r['free_cv']:.4f} | {r['rig_fit']:.4f} / {r['rig_cv']:.4f} | {stg} |")
    print(f"\n### Decomposition of the rig residual (smooth basis {nb}), RMS px\n")
    print("| localizer | total | smooth | board | remainder | remainder x-fold | remainder of CV resid | in-plane-only board: board / rem / rem x-fold | board-first: board / smooth / rem | board in-plane / out-of-plane um |\n|---|---|---|---|---|---|---|---|---|---|")
    for loc, r in rows:
        print(
            f"| {loc} | {r['rig_fit']:.4f} | {r['smooth']:.4f} | {r['board']:.4f} | {r['rem']:.4f} | {r['rem_xfold']:.4f} | {r['rem_cv']:.4f} | "
            f"{r['board_2d']:.4f} / {r['rem_2d']:.4f} / {r['rem_2d_xfold']:.4f} | "
            f"{r['bf_board']:.4f} / {r['bf_smooth']:.4f} / {r['bf_rem']:.4f} | {r['beta_um']:.2f} / {r['beta_z_um']:.2f} |"
        )
    # Common vs localizer-specific remainder (N-cornered hat over localizers):
    # mean |e_a - e_b|^2 = v_a + v_b if the localizer-specific parts are
    # independent; common = mean |e_a|^2 - v_a.
    if len(rems) >= 3:
        names = [n for n in rems if n not in HAT_EXCLUDE]
        hat = {}
        for k, lab in ((0, "in-sample"), (1, "x-fold"), (2, "local")):
            E = {n: rems[n][k] for n in names}
            ok = np.all([np.isfinite(E[n][:, 0]) for n in names], axis=0)
            A, y = [], []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    row_ = np.zeros(len(names))
                    row_[[i, j]] = 1
                    A.append(row_)
                    y.append(((E[names[i]][ok] - E[names[j]][ok]) ** 2).sum(1).mean())
            v = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
            hat[lab] = {n: (float(np.sqrt(max(v[i], 0))), float(np.sqrt(max((E[n][ok] ** 2).sum(1).mean() - v[i], 0)))) for i, n in enumerate(names)}
        summ["remainder_hat"] = hat
        print("\nRemainder split into a part common to all localizers and a localizer-specific part (N-cornered hat), px per point\n")
        print("| localizer | specific / common (in-sample) | specific / common (x-fold) | specific / common (local) |\n|---|---|---|---|")
        for n in names:
            print("| " + n + " | " + " | ".join(f"{hat[lab][n][0]:.4f} / {hat[lab][n][1]:.4f}" for lab in ("in-sample", "x-fold", "local")) + " |")
        print("\nPairwise RMS difference of the remainders, px per point (an upper bound on either localizer's specific error, if not anti-correlated)\n")
        print("| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        pw = {}
        for a_ in names:
            pw[a_] = {b_: rms(rems[a_][0] - rems[b_][0]) for b_ in names}
            print(f"| {a_} | " + " | ".join(f"{pw[a_][b_]:.4f}" for b_ in names) + " |")
        summ["remainder_pairwise"] = pw
        for n in HAT_EXCLUDE & set(rems):
            dd_ = rems[n][0] - rems["SaddlePoint"][0]
            print(f"\n{n} vs SaddlePoint: RMS remainder difference {rms(dd_):.4f} px per point")
    ref = dict(rows)["SaddlePoint"]
    print("\nImplied extra error over SaddlePoint, sqrt(rms^2 - rms_SP^2), px: raw rig residual vs remainder\n")
    print("| localizer | raw | remainder | remainder x-fold |\n|---|---|---|---|")
    for loc, r in rows:
        ex = lambda k: np.sqrt(max(r[k] ** 2 - ref[k] ** 2, 0.0))  # noqa: E731
        print(f"| {loc} | {ex('rig_fit'):.4f} | {ex('rem'):.4f} | {ex('rem_xfold'):.4f} |")
    print("\n### Board component across localizers\n")
    b0, ok0 = betas["SaddlePoint"]
    print("| localizer | RMS beta um | beta fold A vs B: var. explained (in-plane) / corr (out-of-plane) | beta cam1 vs cam2: var. explained | RMS (beta - beta_SP) um | var. explained by beta_SP | nbhd R^2 (r=0.5/0.75/1.0 sq) | (beta - beta_SP) nbhd R^2 |\n|---|---|---|---|---|---|---|---|")
    for loc, r in rows:
        b, ok = betas[loc]
        m = ok & ok0
        diff = b[m] - b0[m]
        ve = corr2(b[m], b0[m])
        dn = [corr2(diff, mom[m, k]) for k in range(mom.shape[1])] if loc != "SaddlePoint" else [float("nan")] * 3
        r["beta_vs_sp"] = dict(diff_um=float(np.sqrt((diff**2).sum(1).mean()) * 1e3), ve=ve, diff_nbhd_r2=dn)
        print(f"| {loc} | {r['beta_um']:.2f} | {r['beta_xfold_ve']:.3f} / {r['betaz_xfold_r']:.3f} | {r['beta_cam_ve']:.3f} | {r['beta_vs_sp']['diff_um']:.2f} | {ve:.3f} | {' / '.join(f'{x:.3f}' for x in r['nbhd_r2'])} | {' / '.join(f'{x:.3f}' for x in dn)} |")
    print("\n### Structure of the remainder\n")
    keys = list(rows[0][1]["rem_by_tilt"])
    print("| localizer | remainder | per-view cubic fit: RMS / white-noise expectation | local remainder | " + " | ".join(f"rem {k}" for k in keys) + " |")
    print("|---|---|---|---|" + "---|" * len(keys))
    for loc, r in rows:
        pv = r["per_view_smooth"]
        print(f"| {loc} | {r['rem']:.4f} | {pv['fitted']:.4f} / {pv['white_expected']:.4f} | {r['local']:.4f} | " + " | ".join(f"{r['rem_by_tilt'][k]:.4f}" for k in keys) + " |")
    print("\n### Stereo consistency: cam2<-cam1 from independent per-view PnP (rig-fit intrinsics)\n")
    print("| localizer | views | rot scatter mrad | trans scatter um | trans um x / y / z | after removing smooth + board: rot mrad / trans um |\n|---|---|---|---|---|---|")
    for loc, r in rows:
        s, c = r["stereo"], r["stereo_clean"]
        print(f"| {loc} | {s['n']} | {s['rot_mrad']:.3f} | {s['t_um']:.1f} | {' / '.join(f'{x:.1f}' for x in s['t_um_xyz'])} | {c['rot_mrad']:.3f} / {c['t_um']:.1f} |")
    print("\n### Covariance validation\n")
    print(
        "| localizer | cov frac | median pred sigma px | sigma0^2 raw / rem / rem x-fold / local | sigma0^2 of the specific part: hat (local) / pairwise bound |"
        " Spearman(pred sigma, abs e) raw / rem / rem x-fold / local | rig CV uniform / Sigma-weighted | x-fold remainder uniform / Sigma-weighted |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for loc, r in rows:
        if "cov" not in r:
            continue
        c = r["cov"]
        w = r.get("weighted", {})
        # local remainder: the per-view cubic fit took a fraction dof_frac of the degrees of freedom
        s0_local = c["local"]["s0"] / (1 - r["per_view_smooth"]["dof_frac"])
        base = "SaddlePoint" if loc in HAT_EXCLUDE else loc
        hs = summ.get("remainder_hat", {}).get("local", {}).get(base)
        pw = summ.get("remainder_pairwise", {}).get(base, {})
        bound = min((v for k_, v in pw.items() if k_ != base), default=float("nan"))
        s0h = (hs[0] ** 2 / 2) / r["pred_var_mean"] if hs else float("nan")
        s0b = (bound**2 / 2) / r["pred_var_mean"]
        r["s0_specific"] = dict(hat=s0h, pairwise_bound=s0b, local=s0_local)
        print(
            f"| {loc} | {c['raw']['frac']:.2f} | {c['raw']['pred_sigma_med']:.4f} | "
            f"{c['raw']['s0']:.0f} / {c['rem']['s0']:.0f} / {c['rem_xfold']['s0']:.0f} / {s0_local:.0f} | {s0h:.1f} / {s0b:.1f} | "
            f"{c['raw']['spearman']:.3f} / {c['rem']['spearman']:.3f} / {c['rem_xfold']['spearman']:.3f} / {c['local']['spearman']:.3f} | "
            f"{r['rig_cv']:.4f} / {w.get('cv', float('nan')):.4f} | {r['rem_xfold']:.4f} / {w.get('rem_xfold', float('nan')):.4f} |"
        )
    with open(C.OUT / f"summary_{a.set}.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
