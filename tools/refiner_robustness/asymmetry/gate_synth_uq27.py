"""UQ-27 synthetic check: asymmetry term against the shipped ``Propagated``.

Usage: python gate_synth_uq27.py [--draws 8] [--fA 0.0xx] [--fP 0.0xx]

Independent generator (``../run_synth.py`` conditions, as ``gate_synth.py``),
3 boards x ``--draws`` noisy images per condition; noise sigma supplied (the
realised std), so ``Propagated`` comes from the bindings with
``noise_sigma`` set. Per corner: ``P`` (bindings), ``T`` (``export_uq27.term``
at the numpy SaddlePoint estimate and the same sigma), and
``A = P - 0.053^2 I + T + fA^2 I``, ``P* = P - 0.053^2 I + fP^2 I`` with the
floors of the real-data gate. Prints sigma0^2 and RMS error per condition.
Raw arrays: ``../out/asym_synth_uq27.npz`` (gitignored).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_synth as rs  # noqa: E402
import synth  # noqa: E402

F_SHIPPED = 0.053
JOBS = [  # (condition, cell, noise or None = condition's own)
    ("baseline", 10.0, None), ("gauss 0.5", 10.0, None), ("gauss 2.0", 10.0, None), ("gamma 1/2.2", 10.0, None),
    ("Bayer+demosaic", 10.0, None), ("decentred defocus", 10.0, None), ("vignetting 40%", 10.0, None),
    ("ramp +-20% / image", 10.0, None), ("illum wave +-5%", 10.0, None), ("illum wave +-15%", 10.0, None),
    ("illum wave +-15%", 8.0, None), ("illum wave +-15%", 10.0, 1.0), ("gauss 2.0", 10.0, 1.0),
    ("coma", 10.0, None), ("combined A", 10.0, None), ("combined B", 10.0, None),
]


def run_job(args):
    import estimators as est
    from export_uq27 import term
    from gate_synth import instances

    name, spec, cell, noise, draws, seed = args
    o = spec["optics"]
    optics = synth.Optics(psf=rs.make_psf(o["psf"]))
    if "illum" in o:
        optics.illum, optics.illum_amp = o["illum"]
    optics.vignette = o.get("vignette", 0.0)
    sp = dict(spec["sensor"])
    if noise is not None:
        sp["noise"] = noise
    sen = synth.Sensor(**sp)
    rng = np.random.default_rng(seed)
    E, S, T = [], [], []
    for geom in instances(spec["persp"], cell):
        lin = synth.render(geom, optics, s=rs.S)
        truth = geom.corners(rs.MARGIN)
        clean = synth.sensor(lin, sen, None)
        for _ in range(draws):
            img = synth.sensor(lin, sen, rng)
            sig = float(np.std(img.astype(float) - clean.astype(float)))
            xy, cov = est.saddle_with_sigma(img, sig)
            j = est.match(xy, truth)
            m = j >= 0
            p, c = xy[j[m]], cov[j[m]]
            t, _, valid, _ = term(img, p, sig)
            E.append(p[valid] - truth[m][valid])
            S.append(c[valid])
            T.append(t[valid])
    return (name, cell, noise), np.concatenate(E), np.concatenate(S), np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--fA", type=float, required=True)
    ap.add_argument("--fP", type=float, required=True)
    a = ap.parse_args()
    C = rs.conditions()
    jobs = [(n, C[n], cell, noise, a.draws, 7000 + i) for i, (n, cell, noise) in enumerate(JOBS)]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_job, jobs))
    od = HERE.parent / "out"
    od.mkdir(exist_ok=True)
    raw = {}
    for (name, cell, noise), E, S, T in res:
        key = f"{name}|{cell:g}|{noise}"
        raw.update({f"E|{key}": E, f"S|{key}": S, f"T|{key}": T})
    np.savez(od / "asym_synth_uq27.npz", **raw)
    I2 = np.eye(2)[None]

    def s0(e, Sx):
        return float(np.einsum("ni,nij,nj->n", e, np.linalg.inv(Sx), e).mean() / 2)

    print(f"\nfloors: P* {a.fP:.4f}, A {a.fA:.4f}\n")
    print("| condition | RMS err px/axis | median sqrt(tr T/2) | P (shipped) | P* | A | P+T | Spearman P / A |")
    print("|---|---|---|---|---|---|---|---|")
    from scipy.stats import spearmanr

    for (name, cell, noise), E, S, T in res:
        lab = name if (cell == 10.0 and noise is None) else f"{name}, cell {cell:g}, noise {noise if noise is not None else 2:g}"
        Sb = S - F_SHIPPED**2 * I2
        covs = dict(P=S, Ps=Sb + a.fP**2 * I2, A=Sb + T + a.fA**2 * I2, PT=S + T)
        sps = [spearmanr(np.sqrt(0.5 * (c[:, 0, 0] + c[:, 1, 1])), np.linalg.norm(E, axis=1))[0] for c in (covs["P"], covs["A"])]
        print(f"| {lab} | {np.sqrt((E ** 2).mean()):.4f} | {np.median(np.sqrt(0.5 * np.trace(T, axis1=1, axis2=2))):.4f} | "
              + " | ".join(f"{s0(E, covs[k]):.2f}" for k in ("P", "Ps", "A", "PT")) + f" | {sps[0]:.2f} / {sps[1]:.2f} |")


if __name__ == "__main__":
    main()
