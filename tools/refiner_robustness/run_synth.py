"""One-factor-at-a-time robustness sweep on the independent generator.

Per condition: 3 board instances (192 x 192 px, cell ~10 px, rolls 5/17/38
deg with a mild tilt), one noise-free image each (bias) and ``--draws`` noisy
images each. Every estimator in ``estimators.ALL`` runs on identical images.

Metrics per (condition, estimator), errors = estimate - true corner, over the
*common set* (truth corners matched within 1 px by every PRIMARY estimator in
that image; DIAGNOSTIC estimators are scored on the common set intersected
with their own matches):
  bias   = RMS noise-free error (noise-free image through the same pipeline)
  sigma  = pooled per-corner scatter over the noisy draws (radial)
  total  = RMS error over all noisy draws
  match  = fraction of truth corners each estimator matches (own set)
  out    = fraction of its own matched corners with |e| > 0.5 px
SaddlePoint only: s0 = sigma0^2 = mean(e' Sigma^-1 e)/2 over its own matched,
covariance-bearing corners in the noisy draws (noise sigma estimated per image
by the detector), and the fraction carrying a covariance.

Writes tools/refiner_robustness/out/synth.json (gitignored, local) and prints
markdown tables.
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

import synth  # noqa: E402

S = 16
W = H = 192
MARGIN = 16


def instances(persp: bool = False):
    rolls = [5.0, 17.0, 38.0]
    phases = [(0.37, 0.61), (0.12, 0.83), (0.71, 0.29)]
    out = []
    for i, (r, p) in enumerate(zip(rolls, phases)):
        if persp:
            tilts = [(40.0, -25.0), (-35.0, 30.0), (30.0, 40.0)]
            out.append(synth.Geometry(W, H, cell=14.0, roll=r, tilt=tilts[i], focal=400.0, phase=p, k1=0.15))
        else:
            out.append(synth.Geometry(W, H, cell=10.0, roll=r, tilt=(8.0, -6.0), focal=2000.0, phase=p))
    return out


def conditions():
    g = lambda s: synth.psf_gaussian(s, S)  # noqa: E731
    base_o = dict(psf=("gauss", 1.0))
    base_s = dict(noise=2.0)
    C = {}

    def add(name, optics=None, sen=None, persp=False):
        o = dict(base_o)
        o.update(optics or {})
        s = dict(base_s)
        s.update(sen or {})
        C[name] = dict(optics=o, sensor=s, persp=persp)

    add("baseline")
    add("gauss 0.5", {"psf": ("gauss", 0.5)})
    add("gauss 2.0", {"psf": ("gauss", 2.0)})
    add("defocus disk r1.5", {"psf": ("disk", 1.5, 0.0)})
    add("coma mild", {"psf": ("coma", 0.8, 1.0, 30.0)})
    add("coma", {"psf": ("coma", 0.6, 2.0, 30.0)})
    add("decentred defocus", {"psf": ("disk", 1.5, 0.6)})
    add("vignetting 40%", {"vignette": 0.4})
    add("ramp +-20% / image", {"illum": ("ramp", 0.2)})
    add("illum wave +-5%", {"illum": ("wave", 0.05)})
    add("illum wave +-15%", {"illum": ("wave", 0.15)})
    add("perspective+distortion", persp=True)
    add("Poisson-Gaussian", sen={"noise": 1.0, "poisson_b": 0.075})
    add("Bayer+demosaic", sen={"bayer": True})
    add("JPEG q90", sen={"jpeg": 90})
    add("JPEG q75", sen={"jpeg": 75})
    add("gamma 1/2.2", sen={"gamma": 1 / 2.2})
    add(
        "combined A",
        {"psf": ("coma", 0.7, 2.0, 30.0), "vignette": 0.3},
        {"noise": 1.0, "poisson_b": 0.075, "bayer": True, "gamma": 1 / 2.2, "jpeg": 90},
        persp=True,
    )
    add(
        "combined B",
        {"psf": ("disk", 1.5, 0.6), "illum": ("ramp", 0.2)},
        {"noise": 1.0, "poisson_b": 0.075, "jpeg": 75},
    )
    return C


def make_psf(spec):
    kind = spec[0]
    if kind == "gauss":
        return synth.psf_gaussian(spec[1], S)
    if kind == "disk":
        return synth.psf_disk(spec[1], S, apod=spec[2], apod_angle=-50.0)
    if kind == "coma":
        return synth.psf_coma(spec[1], spec[2], spec[3], S)
    raise ValueError(kind)


def run_condition(args):
    name, spec, draws, seed = args
    import estimators as est

    o = spec["optics"]
    optics = synth.Optics(psf=make_psf(o["psf"]))
    if "illum" in o:
        optics.illum, optics.illum_amp = o["illum"]
    optics.vignette = o.get("vignette", 0.0)
    sen = synth.Sensor(**spec["sensor"])
    names = est.ALL
    rec = {n: dict(bias=[], bias_dm=[], noisy=[], dev=[], matched=0, out=0, total_truth=0) for n in names}
    s0_terms, cov_frac, s0_sup = [], [], []
    rng = np.random.default_rng(seed)
    for gi, geom in enumerate(instances(spec["persp"])):
        lin = synth.render(geom, optics, s=S)
        truth = geom.corners(MARGIN)
        clean = synth.sensor(lin, sen, None)
        res0 = est.run_all(clean, names)
        j0 = {n: est.match(res0[n][0], truth) for n in names}
        common0 = np.all([j0[n] >= 0 for n in est.PRIMARY], axis=0)
        for n in names:
            c = common0 & (j0[n] >= 0)
            e0 = res0[n][0][j0[n][c]] - truth[c]
            rec[n]["bias"].append(e0)
            rec[n]["bias_dm"].append(e0 - e0.mean(0))
        # Noisy draws: per-corner errors keyed by truth index for scatter.
        per = {n: [] for n in names}
        for _ in range(draws):
            img = synth.sensor(lin, sen, rng)
            res = est.run_all(img, names)
            jj = {n: est.match(res[n][0], truth) for n in names}
            common = np.all([jj[n] >= 0 for n in est.PRIMARY], axis=0)
            for n in names:
                own = jj[n] >= 0
                e_own = res[n][0][jj[n][own]] - truth[own]
                rec[n]["matched"] += own.sum()
                rec[n]["total_truth"] += len(truth)
                rec[n]["out"] += (np.linalg.norm(e_own, axis=1) > 0.5).sum()
                e = np.full((len(truth), 2), np.nan)
                c = common & own
                e[c] = res[n][0][jj[n][c]] - truth[c]
                per[n].append(e)
            # SaddlePoint sigma0^2 on its own matched set.
            xy, cov = res["SaddlePoint"]
            js = jj["SaddlePoint"]
            m = js >= 0
            cv = cov[js[m]]
            has = ~np.isnan(cv).any(axis=(1, 2))
            cov_frac.append(has.mean() if m.any() else 0.0)
            e = (xy[js[m]] - truth[m])[has]
            if len(e):
                q = np.einsum("ni,nij,nj->n", e, np.linalg.inv(cv[has]), e)
                s0_terms.append(q)
            # Same, with the realised noise sigma of the final 8-bit image supplied.
            sig = float(np.std(img.astype(float) - clean.astype(float)))
            xy2, cov2 = est.saddle_with_sigma(img, sig)
            j2 = est.match(xy2, truth)
            m2 = j2 >= 0
            cv2_ = cov2[j2[m2]]
            h2 = ~np.isnan(cv2_).any(axis=(1, 2))
            e2 = (xy2[j2[m2]] - truth[m2])[h2]
            if len(e2):
                s0_sup.append(np.einsum("ni,nij,nj->n", e2, np.linalg.inv(cv2_[h2]), e2))
        for n in names:
            P = np.stack(per[n])  # (draws, ntruth, 2)
            rec[n]["noisy"].append(P[~np.isnan(P[..., 0])])
            mean = np.nanmean(P, axis=0)
            cnt = (~np.isnan(P[..., 0])).sum(0)
            d = P - mean[None]
            ss = np.nansum((d**2).sum(-1), axis=0)
            rec[n]["dev"].append((ss[cnt > 1].sum(), (cnt[cnt > 1] - 1).sum()))
    out = {}
    for n in names:
        b = np.concatenate(rec[n]["bias"])
        z = np.concatenate(rec[n]["noisy"])
        ss = sum(a for a, _ in rec[n]["dev"])
        dof = sum(c for _, c in rec[n]["dev"])
        out[n] = dict(
            bias=float(np.sqrt((b**2).sum(1).mean())),
            bias_dm=float(np.sqrt((np.concatenate(rec[n]["bias_dm"]) ** 2).sum(1).mean())),
            sigma=float(np.sqrt(ss / dof)) if dof else float("nan"),
            total=float(np.sqrt((z**2).sum(1).mean())),
            match=float(rec[n]["matched"] / rec[n]["total_truth"]),
            out=float(rec[n]["out"] / max(rec[n]["matched"], 1)),
            n_common=int(len(z)),
        )
    q = np.concatenate(s0_terms) if s0_terms else np.array([np.nan])
    out["SaddlePoint"]["s0"] = float(q.mean() / 2)
    out["SaddlePoint"]["s0_trim"] = float(np.mean(np.sort(q)[: int(0.99 * len(q))]) / 2)
    out["SaddlePoint"]["cov_frac"] = float(np.mean(cov_frac))
    q2 = np.concatenate(s0_sup) if s0_sup else np.array([np.nan])
    out["SaddlePoint"]["s0_sup"] = float(q2.mean() / 2)
    return name, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    C = conditions()
    if args.only:
        C = {k: v for k, v in C.items() if k in args.only}
    jobs = [(k, v, args.draws, 1000 + i) for i, (k, v) in enumerate(C.items())]
    results = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for name, out in ex.map(run_condition, jobs):
            results[name] = out
            print(f"done {name}", flush=True)
    od = HERE / "out"
    od.mkdir(exist_ok=True)
    (od / "synth.json").write_text(json.dumps(results, indent=1))
    print_tables(results)


def print_tables(results) -> None:
    import estimators as est

    names = est.ALL
    print("\n### Bias with the per-image mean error removed (px)\n")
    print("| condition | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c, r in results.items():
        print(f"| {c} | " + " | ".join(f"{r[n]['bias_dm']:.3f}" for n in names) + " |")
    print("\n### Total RMS error (px) -- bias (noise-free RMS) in parentheses\n")
    print("| condition | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c, r in results.items():
        print(f"| {c} | " + " | ".join(f"{r[n]['total']:.3f} ({r[n]['bias']:.3f})" for n in names) + " |")
    print("\n### Scatter sigma (px, radial) / match rate / outliers > 0.5 px\n")
    print("| condition | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c, r in results.items():
        print(
            f"| {c} | "
            + " | ".join(f"{r[n]['sigma']:.3f} / {r[n]['match']:.2f} / {r[n]['out']:.3f}" for n in names)
            + " |"
        )
    print("\n### SaddlePoint sigma0^2 (estimated noise), 99%-trimmed, covariance fraction\n")
    print("| condition | s0 | s0 trimmed | cov frac | s0 supplied sigma | common n |")
    print("|---|---|---|---|---|---|")
    for c, r in results.items():
        sp = r["SaddlePoint"]
        print(f"| {c} | {sp['s0']:.2f} | {sp['s0_trim']:.2f} | {sp['cov_frac']:.2f} | {sp['s0_sup']:.2f} | {sp['n_common']} |")


if __name__ == "__main__":
    main()
