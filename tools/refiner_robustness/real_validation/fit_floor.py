"""Real-data floor of each localizer's systematic covariance term.

Usage (after ``detect.py``, ``calibrate.py --set <set>``):

    python fit_floor.py [--set all|chess]

For every localizer whose detections carry a covariance ``Sigma`` it takes
the *local remainder* of the rig calibration residual (lens field, per-corner
target term and the per-view smooth pattern common to all localizers
removed; ``analyze.py``) and finds the isotropic floor ``f`` (px per axis)
for which ``Sigma + f^2 I`` whitens it to unit variance:
``mean(e' (Sigma + f^2 I)^-1 e) / 2 / (1 - dof) = 1`` (``dof``: the fraction
of the remainder the per-view fit absorbs). Also prints sigma0^2 with the
covariance as reported and the Spearman correlation of the predicted sigma
with |e|. Aggregates only.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
from scipy.stats import spearmanr

import analyze as A
import common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--knots", default="5,4")
    a = ap.parse_args()
    nb = tuple(int(x) for x in a.knots.split(","))
    d = pickle.load(open(C.OUT / f"calib_{a.set}.pkl", "rb"))
    o = d["obs"]
    allm = np.ones(len(o["cam"]), bool)
    print(f"set {a.set}: {len(o['cam'])} observations")
    print("| localizer | local remainder RMS/axis | median pred sigma | sigma0^2 as reported | floor f px | sigma0^2 with floor | Spearman(pred sigma, abs e) |")
    print("|---|---|---|---|---|---|---|")
    for loc in d["locs"]:
        S = o["cov"].get(loc)
        if S is None or not np.isfinite(S).all(axis=(1, 2)).any():
            continue
        L = d["locs"][loc]
        P = o["xy"][loc]
        oo = dict(P=P, cam=o["cam"], id=o["id"], shape=o["shape"])
        e, dof_row = A.local_remainder(oo, o, L["rig"]["res"], L["rig"]["Jb"], allm, nb)
        dof = float(dof_row.mean())
        ok = np.isfinite(S).all(axis=(1, 2))
        e, S = e[ok], S[ok]
        s_rep = C.sigma0_sq(e, S, dof)
        f = C.fit_floor(e, S, dof)
        ps = np.sqrt(0.5 * (S[:, 0, 0] + S[:, 1, 1]))
        rho = spearmanr(ps, np.linalg.norm(e, axis=1))[0]
        print(f"| {loc} | {np.sqrt((e ** 2).mean() / (1 - dof)):.4f} | {np.median(ps):.4f} | {s_rep:.2f} | {f:.4f} | {C.sigma0_sq(e, S + f * f * np.eye(2)[None], dof):.2f} | {rho:.3f} |")


if __name__ == "__main__":
    main()
