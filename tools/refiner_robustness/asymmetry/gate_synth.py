"""UQ-26 gate on the independent generator.

Conditions are those of ``../run_synth.py`` (imported as ``rs``) (cell 10, noise 2), plus a grid
(cells 8 / 12 at noise 2, noise 1 / 8 at cell 10) on the symmetric and
asymmetric subsets. Per condition: 3 boards, a noise-free image each and
``--draws`` noisy images each. SaddlePoint positions come from the bindings;
everything else from ``core.methods``. The noise sigma is the realised std of
(noisy - clean) of the final 8-bit image (supplied, so the noise estimator
is not under test).

Per condition and method: s0 = mean(e' Sigma^-1 e) / 2 over the noisy draws,
total RMS (radial) error; noise-free: SaddlePoint bias RMS, and the bias left
after subtracting the asymmetry estimate.

Writes ``../out/asym_synth.json`` (gitignored) and prints markdown tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synth  # noqa: E402
import run_synth as rs  # noqa: E402

ASYM = (("r2", 2, False), ("r3+add", 3, True), ("mul2", 2, "mul"), ("mul3", 3, "mul"), ("mul4", 4, "mul"), ("mulB4", 4, "mulB"), ("mulB3", 3, "mulB"))
SYMMETRIC = ["baseline", "gauss 0.5", "gauss 2.0", "defocus disk r1.5", "perspective+distortion",
             "Poisson-Gaussian", "gamma 1/2.2", "Bayer+demosaic", "JPEG q90", "JPEG q75"]
ASYMMETRIC = ["decentred defocus", "vignetting 40%", "ramp +-20% / image", "illum wave +-5%", "illum wave +-15%"]
COMA = ["coma mild", "coma", "combined A", "combined B"]
GRID = ["baseline", "gauss 2.0", "decentred defocus", "vignetting 40%", "ramp +-20% / image",
        "illum wave +-5%", "illum wave +-15%"]


def instances(persp: bool, cell: float):
    if persp:
        return rs.instances(True)
    out = []
    for g in rs.instances(False):
        g.cell = cell
        out.append(g)
    return out


def run_job(args):
    import estimators as est
    import core

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
    acc: dict = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)

    nvalid, ntot = 0, 0
    for geom in instances(spec["persp"], cell):
        lin = synth.render(geom, optics, s=rs.S)
        truth = geom.corners(rs.MARGIN)
        clean = synth.sensor(lin, sen, None)
        xy0, _ = est.chess(clean, "SaddlePoint")
        j0 = est.match(xy0, truth)
        m0 = j0 >= 0
        r0, v0 = core.methods(clean.astype(float), xy0[j0[m0]], 0.0, ASYM)
        t0 = truth[m0][v0]
        e0 = r0["c"][v0] - t0
        add("bias", e0)
        add("bias_dm", e0 - e0.mean(0))
        for lab, _, _ in ASYM:
            eb = r0[f"c_b_{lab}"][v0] - t0
            add(f"bias_b_{lab}", eb)
            add(f"bias_bdm_{lab}", eb - eb.mean(0))
        dc = r0["c_r"][:, v0] - t0[None]
        for i, r in enumerate((1, 2, 3)):
            add(f"bias_r{r}", dc[i])
        for _ in range(draws):
            img = synth.sensor(lin, sen, rng)
            sig = float(np.std(img.astype(float) - clean.astype(float)))
            xy, _ = est.chess(img, "SaddlePoint")
            j = est.match(xy, truth)
            m = j >= 0
            r, v = core.methods(img.astype(float), xy[j[m]], sig, ASYM)
            t = truth[m][v]
            nvalid += v.sum()
            ntot += len(truth)
            e = r["c"][v] - t
            add("e_c", e)
            for key in ["S_150", "S_fl10", "S_fl15", "S_fl20", "S_c", "S_craw"] + [f"S_a_{lab}" for lab, _, _ in ASYM] + [f"S_araw_{lab}" for lab, _, _ in ASYM]:
                add("q_" + key, core.whiten_q(e, r[key][v]))
            for lab, _, _ in ASYM:
                add(f"Gdiag_{lab}", r[f"G_{lab}"][v][:, [0, 1], [0, 1]])
            add("psig_150", np.sqrt(0.5 * np.trace(r["S_150"][v], axis1=1, axis2=2)))
            for lab, _, _ in ASYM:
                eb = r[f"c_b_{lab}"][v] - t
                add(f"e_b_{lab}", eb)
                add(f"e_bit_{lab}", r[f"c_bit_{lab}"][v] - t)
                add(f"q_S_b_{lab}", core.whiten_q(eb, r[f"S_b_{lab}"][v]))
                ebg = r[f"c_bg_{lab}"][v] - t
                add(f"e_bg_{lab}", ebg)
                add(f"q_S_bg_{lab}", core.whiten_q(ebg, r[f"S_bg_{lab}"][v]))
                add(f"gfrac_{lab}", (r[f"zg_{lab}"][v] > 9.21)[:, None].astype(float))
                add(f"psig_b_{lab}", np.sqrt(0.5 * np.trace(r[f"S_b_{lab}"][v], axis1=1, axis2=2)))
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    rms = lambda a: float(np.sqrt((a**2).sum(1).mean()))  # noqa: E731
    out = dict(valid=float(nvalid / max(ntot, 1)), n=int(len(cat["e_c"])))
    for k, a in cat.items():
        if k.startswith("q_"):
            out["s0_" + k[2:]] = float(a.mean() / 2)
            out["s0med_" + k[2:]] = float(np.median(a) / (2 * np.log(2)))
        elif k.startswith("Gdiag"):
            out[k] = float(np.median(a))
        elif k.startswith("gfrac"):
            out[k] = float(a.mean())
        elif k.startswith("psig"):
            out[k] = float(np.sqrt((a**2).mean()))
        else:
            out[k] = rms(a)
    return (name, cell, noise), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    C = rs.conditions()
    jobs = []
    i = 0
    for name, spec in C.items():
        if args.only and name not in args.only:
            continue
        jobs.append((name, spec, 10.0, None, args.draws, 5000 + i))
        i += 1
    if not args.no_grid:
        for name in GRID:
            if args.only and name not in args.only:
                continue
            for cell, noise in [(8.0, None), (12.0, None), (10.0, 1.0), (10.0, 8.0)]:
                jobs.append((name, C[name], cell, noise, args.draws, 5000 + i))
                i += 1
    res = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for key, out in ex.map(run_job, jobs):
            res[key] = out
            print("done", key, flush=True)
    od = HERE.parent / "out"
    od.mkdir(exist_ok=True)
    (od / "asym_synth.json").write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, indent=1))
    tables(res)


def tables(res):
    def row(key, r):
        name, cell, noise = key
        lab = name if (cell == 10.0 and noise is None) else f"{name} c{cell:g} n{noise if noise is not None else 2:g}"
        return lab

    print("\n### sigma0^2 (mean e'S^-1 e / 2)\n")
    cols = ["S_150", "S_c", "S_craw"] + [f"S_a_{l}" for l, _, _ in ASYM] + [f"S_b_{l}" for l, _, _ in ASYM if "mul" in l] + [f"S_bg_{l}" for l, _, _ in ASYM if "mul" in l]
    print("| condition | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k, r in res.items():
        print(f"| {row(k, r)} | " + " | ".join(f"{r['s0_' + c]:.2f}" for c in cols) + " |")
    print("\n### Total RMS error px (noisy) and noise-free bias px\n")
    cols2 = ["e_c"] + [f"e_b_{l}" for l, _, _ in ASYM] + [f"e_bg_{l}" for l, _, _ in ASYM if "mul" in l] + [f"gfrac_{l}" for l, _, _ in ASYM if "mul" in l] + ["bias", "bias_dm"] + [f"bias_b_{l}" for l, _, _ in ASYM] + ["bias_r1", "bias_r3"]
    print("| condition | " + " | ".join(cols2) + " | valid |")
    print("|---|" + "---|" * (len(cols2) + 1))
    for k, r in res.items():
        print(f"| {row(k, r)} | " + " | ".join(f"{r[c]:.3f}" for c in cols2) + f" | {r['valid']:.2f} |")


if __name__ == "__main__":
    main()
