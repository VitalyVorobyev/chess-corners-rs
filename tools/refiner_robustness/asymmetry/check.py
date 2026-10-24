"""Checks of the numpy prototype against the bindings and Monte Carlo.

1. numpy SaddlePoint (radius 2) reproduces the bindings' positions and
   (supplied-sigma) covariances.
2. Jacobian maps vs Monte Carlo scatter: SaddlePoint, asymmetry beta (9x9, (g.d)S ramp), corrected
   position, for radii 1/2/3.
3. Noise-free: how much of SaddlePoint's per-corner bias the asymmetry fit
   explains, per condition (the core question of the prototype).

Run: .venv-uq26/bin/python tools/refiner_robustness/asymmetry/check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import core  # noqa: E402
import estimators as est  # noqa: E402
import run_synth as rs  # noqa: E402
import synth  # noqa: E402


def setup(cond="baseline", gi=0):
    spec = rs.conditions()[cond]
    o = spec["optics"]
    optics = synth.Optics(psf=rs.make_psf(o["psf"]))
    if "illum" in o:
        optics.illum, optics.illum_amp = o["illum"]
    optics.vignette = o.get("vignette", 0.0)
    sen = synth.Sensor(**spec["sensor"])
    geom = rs.instances(spec["persp"])[gi]
    lin = synth.render(geom, optics, s=rs.S)
    truth = geom.corners(rs.MARGIN)
    return lin, sen, truth


def main():
    lin, sen, truth = setup()
    rng = np.random.default_rng(1)
    img = synth.sensor(lin, sen, rng)
    clean = synth.sensor(lin, sen, None)
    sig = float(np.std(img.astype(float) - clean.astype(float)))
    xy, cov = est.saddle_with_sigma(img, sig)
    j = est.match(xy, truth)
    m = j >= 0
    c0, cv = xy[j[m]], cov[j[m]]
    c, ok, L = core.saddle(img.astype(float), c0, 2)
    C = core.cov_of(L, max(sig**2, core.MIN_VAR)) + core.floor_cov(len(c))
    has = ~np.isnan(cv).any(axis=(1, 2)) & ok
    print(f"1. n={has.sum()}  max |c_np - c_bind| = {np.abs(c - c0)[has].max():.2e} px")
    rel = np.abs(C[has] - cv[has]).max(axis=(1, 2)) / np.abs(cv[has]).max(axis=(1, 2))
    print(f"   covariance rel diff median {np.median(rel):.2e}, max {rel.max():.2e}")

    # 2. Monte Carlo on a fixed lin image: maps evaluated on the clean image.
    cl = clean.astype(float)
    cc, okc, Lc = core.saddle(cl, c0, 2)
    a = core.asym_full(cl, cc, Lc, 4, "mul")
    reps = 300
    pos, bs, cor = [], [], []
    rad = {1: [], 3: []}
    for _ in range(reps):
        im = (lin + rng.normal(0, 2.0, lin.shape))  # no rounding: linear check
        ci, oki, _ = core.saddle(im, cc, 2)
        bi = np.einsum("nij,nj->ni", np.linalg.inv(a["G"]), core.asym_fit(im, ci, 4, "mul")[0])
        pos.append(ci)
        bs.append(bi)
        cor.append(ci - bi)
        for r in (1, 3):
            rad[r].append(core.saddle(im, cc, r)[0])
    clin, _, Llin = core.saddle(lin, cc, 2)
    alin = core.asym_full(lin, clin, Llin, 4, "mul")

    def ratio(samples, Lmap):
        s = np.stack(samples)
        emp = np.einsum("rni,rnj->nij", s - s.mean(0), s - s.mean(0)) / (reps - 1)
        pred = core.cov_of(Lmap, 4.0)
        return np.median(np.trace(emp, axis1=1, axis2=2) / np.trace(pred, axis1=1, axis2=2)), \
            np.sqrt(np.median(np.trace(pred, axis1=1, axis2=2) / 2))

    print("2. MC/predicted variance ratio (median over corners), predicted sigma/axis at noise 2:")
    print("   saddle r2 %.3f (%.4f px)" % ratio(pos, Llin))
    print("   beta (Lc + G^-1 Lb) %.3f (%.4f px)" % ratio(bs, alin["Lbeta"]))
    print("   corrected %.3f (%.4f px)" % ratio(cor, alin["Lcorr"]))
    for r in (1, 3):
        Lr_lin = core.saddle(lin, cc, r)[2]
        print(f"   saddle r{r} %.3f (%.4f px)" % ratio(rad[r], Lr_lin))
    print("   gain G median diag %.3f offdiag %.3f" % (np.median(alin["G"][:, [0, 1], [0, 1]]), np.median(np.abs(alin["G"][:, 0, 1]))))
    print("   cond(XtX) median %.1e" % np.median(alin["cond"]))


if __name__ == "__main__":
    main()
