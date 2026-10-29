"""Cross-camera floor (UQ-44). Pre-registered definition and decision rule:
docs/reference/uncertainty-model.md, "Cross-camera floor (UQ-44)".

Same inputs as ``ablation.py`` (``calibrate.py`` outputs in ``$RV_OUT`` and
the core dumps). Usage::

    python crossfloor.py --real ablation_real.csv --systematic systematic.csv \\
        [--synth ablation_synth.csv] [--boot 200]

Per localizer and corner set, with model (a) (shipped knots):

- ``e_in``: the shipped local remainder (all terms fitted on both cameras);
- ``e_x``: per held-out camera B, target term from the other camera, B's
  lens field fitted on B, per-view cubic per view of B;
- ``f_in`` / ``f_x``: sigma0^2 = 1 on ``e_in`` / on ``e_x`` pooled over both
  directions; ``f_m``: the mean of the two sigma0^2 equal to 1 (candidate).

Prints markdown tables and writes ``crossfloor.json``. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import ablation as AB
import analyze as A
import calibrate as K
import common as C

EYE = np.eye(2)[None]
# Harness localizer -> systematic table whose floor it sets (max rule).
TABLE_OF = {"CoM": "CoM", "SaddlePoint": "SaddlePoint", "Radon+SP": "SaddlePoint", "Radon": "RadonPeak"}
BAND = (0.67, 1.5)


def remainders(P, cam, ids, view, r, Jb, shape, nb):
    """``e_in`` and ``e_x`` (rows of camera B carry B's cross-camera
    remainder) with each row's per-view dof share, and the per-view cubic's
    RMS on each."""
    oo = dict(P=P, cam=cam, id=ids, shape=shape)
    o = dict(cam=cam, view=view)
    allm = np.ones(len(P), bool)
    dec = A.decompose(oo, r, Jb, allm, nb)
    pv = A.per_view_smooth(o, P, dec["rem"])
    e_in, dof_in = dec["rem"] - pv["fit"], pv["dof_row"]
    rem_x = np.full_like(r, np.nan)
    for b in (0, 1):
        test = cam == b
        dx = A.decompose(oo, r, Jb, ~test, nb)
        pred = A.Smooth(shape, nb).fit(P, cam, r - dx["board"], test)
        rem_x[test] = (r - pred(P, cam) - dx["board"])[test]
    px = A.per_view_smooth(o, P, rem_x)
    return dict(e_in=e_in, dof_in=dof_in, e_x=rem_x - px["fit"], dof_x=px["dof_row"], cubic_in=A.rms(pv["fit"]), cubic_x=A.rms(px["fit"]))


def s0(e, base, dof_row, m, f):
    return C.sigma0_sq(e[m], base[m] + f * f * EYE, float(dof_row[m].mean()))


def floors(R, base, ok):
    """f_in, f_x and f_m."""
    e_in, e_x, d_in, d_x = R["e_in"], R["e_x"], R["dof_in"], R["dof_x"]

    def solve(g):
        return brentq(lambda f: g(f) - 1.0, 0.0, 1.0) if g(0.0) > 1 else 0.0

    f_in = solve(lambda f: s0(e_in, base, d_in, ok, f))
    f_x = solve(lambda f: s0(e_x, base, d_x, ok, f))
    f_m = solve(lambda f: 0.5 * (s0(e_in, base, d_in, ok, f) + s0(e_x, base, d_x, ok, f)))
    return dict(f_in=f_in, f_x=f_x, f_m=f_m)


def per_camera(R, base, ok, cam, f_x, f_m):
    out = {}
    for b in (0, 1):
        m = ok & (cam == b)
        sx = s0(R["e_x"], base, R["dof_x"], m, f_x)
        mid = 0.5 * (s0(R["e_in"], base, R["dof_in"], m, f_m) + s0(R["e_x"], base, R["dof_x"], m, f_m))
        out[C.CAMS[b]] = dict(s0_x_fx=sx, mid_fm=mid, need_x=floors_one(R["e_x"], base, R["dof_x"], m))
    return out


def floors_one(e, base, dof_row, m):
    g = lambda f: s0(e, base, dof_row, m, f)  # noqa: E731
    return brentq(lambda f: g(f) - 1.0, 0.0, 1.0) if g(0.0) > 1 else 0.0


class Loc:
    """One localizer on one corner set: the arrays a refit needs."""

    def __init__(self, rs: AB.RealSet, loc: str):
        o = rs.o
        L = rs.locs[loc]
        rig = rs.d["locs"][loc]["rig"]
        self.P, self.r, self.Jb = o["xy"][loc], rig["res"], rig["Jb"]
        self.cam, self.ids, self.view = o["cam"], o["id"], o["view"]
        self.shape = o["shape"]
        self.ok = L["ok"]
        self.base = np.full((len(self.ok), 2, 2), np.nan)
        self.base[self.ok] = L["rec"].base("a")
        self.base[~self.ok] = np.eye(2)  # never used (masked), keeps inv finite

    def run(self, nb, rows=None, view=None):
        if rows is None:
            rows, view = np.arange(len(self.P)), self.view
        R = remainders(self.P[rows], self.cam[rows], self.ids[rows], view, self.r[rows], self.Jb[rows], self.shape, nb)
        return R, floors(R, self.base[rows], self.ok[rows])


def boot_one(args):
    locs, nb, seed, nviews = args
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nviews, nviews)
    out = {}
    for key, L in locs.items():
        rows, view = [], []
        for k, v in enumerate(pick):
            s = np.where(L.view == v)[0]
            rows.append(s)
            view.append(np.full(len(s), k))
        rows, view = np.concatenate(rows), np.concatenate(view)
        out[key] = L.run(nb, rows, view)[1]
    return out


def delta_beta(L: Loc, nb):
    """Per-id target terms from each camera alone: RMS of their in-plane
    difference (um) and its expectation from the remainder's scatter."""
    oo = dict(P=L.P, cam=L.cam, id=L.ids, shape=L.shape)
    nid = (C.NSQ - 1) ** 2
    bs, Ainv = [], []
    for c in (0, 1):
        dec = A.decompose(oo, L.r, L.Jb, L.cam == c, nb)
        bs.append(dec)
        m = L.cam == c
        var = (dec["rem"][m] ** 2).sum(1).mean() / 2
        Am = np.zeros((nid, 3, 3))
        np.add.at(Am, L.ids[m], np.einsum("nki,nkj->nij", L.Jb[m], L.Jb[m]))
        Ai = np.full((nid, 3, 3), np.nan)
        Ai[dec["beta_ok"]] = np.linalg.inv(Am[dec["beta_ok"]]) * var
        Ainv.append(Ai)
    okb = bs[0]["beta_ok"] & bs[1]["beta_ok"]
    d = bs[0]["beta"][okb] - bs[1]["beta"][okb]
    exp = (Ainv[0][okb] + Ainv[1][okb])[:, :2, :2]
    return dict(
        d=d, okb=okb,
        rms_um=float(np.sqrt((d[:, :2] ** 2).sum(1).mean()) * 1e3),
        noise_um=float(np.sqrt(np.trace(exp, axis1=1, axis2=2).mean()) * 1e3),
        beta_um=float(np.sqrt((bs[0]["beta"][okb, :2] ** 2).sum(1).mean()) * 1e3),
    )


def leakage(L: Loc, R: dict, nb):
    """Per held-out camera B: B's in-sample floor, and the per-axis RMS of the
    training camera's random error that its per-id target term carries into
    B (``J_B A^-1 J_B' sigma_A^2``) against what the term absorbs on its own
    camera; with the 3-D and the in-plane-only target term."""
    nid = (C.NSQ - 1) ** 2
    oo = dict(P=L.P, cam=L.cam, id=L.ids, shape=L.shape)
    out = {}
    for b in (0, 1):
        ma, mb = L.cam != b, L.cam == b
        dx = A.decompose(oo, L.r, L.Jb, ma, nb)
        var = (dx["rem"][ma] ** 2).sum(1).mean() / 2
        row = dict(f_in=floors_one(R["e_in"], L.base, R["dof_in"], L.ok & mb))
        for dims in (3, 2):
            J = L.Jb[:, :, :dims]
            Am = np.zeros((nid, dims, dims))
            np.add.at(Am, L.ids[ma], np.einsum("nki,nkj->nij", J[ma], J[ma]))
            Ai = np.zeros((nid, dims, dims))
            Ai[dx["beta_ok"]] = np.linalg.inv(Am[dx["beta_ok"]])
            q = lambda m: float(np.sqrt(np.einsum("nki,nij,nkj->n", J[m], Ai[L.ids[m]], J[m]).mean() * var / 2))  # noqa: E731
            row[f"leak{dims}"], row[f"own{dims}"] = q(mb), q(ma)
        out[C.CAMS[b]] = row
    R2 = remainders(L.P, L.cam, L.ids, L.view, L.r, L.Jb[:, :, :2], L.shape, nb)
    f2 = floors(R2, L.base, L.ok)
    return out, f2, per_camera(R2, L.base, L.ok, L.cam, f2["f_x"], f2["f_m"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--systematic", required=True)
    ap.add_argument("--synth")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    nb = (5, 4)
    const = AB.constant_models(Path(a.systematic))
    real = pd.read_csv(a.real)
    sets = {s: AB.RealSet(s, real, const, nb) for s in ("all", "chess")}
    locs = {(s, loc): Loc(rs, loc) for s, rs in sets.items() for loc in rs.locs}
    summ = dict(point={}, cams={}, cubic={}, nb108={}, boot={}, dbeta={}, rank={}, calib={}, synth={})

    print("\n### Floors (px per axis), 5x4 lens basis\n")
    print("| set | localizer | f_in | f_x | f_m | per-view cubic RMS e_in / e_x | cam1: need_x, s0_x(f_x), mid(f_m) | cam2: need_x, s0_x(f_x), mid(f_m) |\n|---|---|---|---|---|---|---|---|")
    point = {}
    for (s, loc), L in locs.items():
        R, f = L.run(nb)
        cams = per_camera(R, L.base, L.ok, L.cam, f["f_x"], f["f_m"])
        point[(s, loc)] = f
        summ["point"][f"{s}|{loc}"] = f
        summ["cams"][f"{s}|{loc}"] = cams
        summ["cubic"][f"{s}|{loc}"] = (R["cubic_in"], R["cubic_x"])
        cc = " | ".join(f"{x['need_x']:.4f}, {x['s0_x_fx']:.2f}, {x['mid_fm']:.2f}" for x in cams.values())
        print(f"| {s} | {loc} | {f['f_in']:.4f} | {f['f_x']:.4f} | {f['f_m']:.4f} | {R['cubic_in']:.4f} / {R['cubic_x']:.4f} | {cc} |", flush=True)
        lk, f2, c2 = leakage(L, R, nb)
        summ.setdefault("leak", {})[f"{s}|{loc}"] = dict(cams=lk, inplane=f2, inplane_cams=c2)
        print(
            "    per held-out camera: "
            + "; ".join(f"{k} f_in {x['f_in']:.4f}, leak 3-D {x['leak3']:.4f} (own {x['own3']:.4f}), in-plane {x['leak2']:.4f} (own {x['own2']:.4f})" for k, x in lk.items())
            + f"; in-plane target term: f_x {f2['f_x']:.4f} f_m {f2['f_m']:.4f}, s0_x(f_x) "
            + " / ".join(f"{x['s0_x_fx']:.2f}" for x in c2.values()),
            flush=True,
        )

    # Lens-structure check: every lens field on a 10x8 basis.
    print("\n### 10x8 lens basis: excess f_x^2 - f_in^2 kept\n")
    for (s, loc), L in locs.items():
        f8 = L.run((10, 8))[1]
        f5 = point[(s, loc)]
        ex5 = f5["f_x"] ** 2 - f5["f_in"] ** 2
        ex8 = f8["f_x"] ** 2 - f8["f_in"] ** 2
        summ["nb108"][f"{s}|{loc}"] = dict(**f8, kept=ex8 / ex5 if ex5 > 0 else float("nan"))
        print(f"[{s}] {loc:11s} f_in {f8['f_in']:.4f} f_x {f8['f_x']:.4f} f_m {f8['f_m']:.4f}; excess kept {ex8 / ex5:.2f}", flush=True)

    # Table floors (max rule).
    def table(fl: dict, key: str):
        out = {}
        for (s, loc), f in fl.items():
            t = TABLE_OF[loc]
            out[t] = max(out.get(t, 0.0), f[key])
        return out

    tab = {k: table(point, k) for k in ("f_in", "f_x", "f_m")}
    summ["table"] = tab
    print("\nper table:", json.dumps(tab))

    # View bootstrap.
    if a.boot:
        nviews = len(sets["all"].o["tokens"])
        jobs = [(locs, nb, 1000 + i, nviews) for i in range(a.boot)]
        reps = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, res in enumerate(ex.map(boot_one, jobs, chunksize=2)):
                reps.append(res)
                if (i + 1) % 20 == 0:
                    print(f"  bootstrap {i + 1}/{a.boot}", flush=True)
        print("\n### View bootstrap: 90% interval of the table floors\n")
        print("| table | f | point | 5% | 95% | half-width / point |\n|---|---|---|---|---|---|")
        for k in ("f_in", "f_x", "f_m"):
            tb = [table(rep, k) for rep in reps]
            for t in tab[k]:
                x = np.array([r_[t] for r_ in tb])
                lo, hi = np.percentile(x, [5, 95])
                hw = (hi - lo) / 2 / tab[k][t]
                summ["boot"][f"{t}|{k}"] = dict(lo=lo, hi=hi, hw=hw, sd=float(x.std()))
                print(f"| {t} | {k} | {tab[k][t]:.4f} | {lo:.4f} | {hi:.4f} | {hw:.2f} |")
        # per (set, localizer) too, for the report
        for key in locs:
            x = np.array([rep[key]["f_m"] for rep in reps])
            summ["boot"][f"{key[0]}|{key[1]}|f_m"] = np.percentile(x, [5, 95]).tolist()

    # Camera-specific target term.
    print("\n### Target term from each camera alone (in-plane, um)\n")
    print("| set | localizer | RMS beta | RMS (beta_cam1 - beta_cam2) | noise expectation | var. of SP's difference explained by this one's |\n|---|---|---|---|---|---|")
    for s in sets:
        db = {loc: delta_beta(locs[(s, loc)], nb) for loc in sets[s].locs}
        ref = db["SaddlePoint"]
        for loc, x in db.items():
            m = x["okb"] & ref["okb"]
            ia = np.flatnonzero(m[x["okb"]])
            ib = np.flatnonzero(m[ref["okb"]])
            ve = A.corr2(ref["d"][ib, :2], x["d"][ia, :2]) if loc != "SaddlePoint" else float("nan")
            summ["dbeta"][f"{s}|{loc}"] = dict(rms_um=x["rms_um"], noise_um=x["noise_um"], beta_um=x["beta_um"], ve_sp=ve)
            print(f"| {s} | {loc} | {x['beta_um']:.2f} | {x['rms_um']:.2f} | {x['noise_um']:.2f} | {ve:.2f} |")

    # Ranking and weighting with the candidate floors.
    ship = {t: AB.shipped_table(t)[1] for t in AB.TABLES}
    new = tab["f_m"]
    print("\n### Ranking and weighting: shipped vs f_m\n")
    print("| set | localizer | Spearman local ship / new | Spearman x-fold ship / new | CV ship / new | x-fold rem ship / new |\n|---|---|---|---|---|---|")
    for s, rs in sets.items():
        for loc, L in rs.locs.items():
            S0 = rs.full_cov(loc, "a", ship)
            S1 = rs.full_cov(loc, "a", new)
            sp = [(AB.stats(L["local"], S, L["dof"])["rho"], AB.stats(L["xrem"], S)["rho"]) for S in (S0, S1)]
            summ["rank"][f"{s}|{loc}"] = sp
            row = f"| {s} | {loc} | {sp[0][0]:.3f} / {sp[1][0]:.3f} | {sp[0][1]:.3f} / {sp[1][1]:.3f} |"
            if not a.skip_calib:
                Ld = rs.d["locs"][loc]
                P = rs.o["xy"][loc]
                cm = []
                for tag, S in (("ship", S0), ("new", S1)):
                    cache = C.OUT / f"crossfloor_rig_{s}_{loc}_{tag}.pkl"
                    if cache.exists():
                        fit = pd.read_pickle(cache)
                    else:
                        fit = K.rig_fit_cv(rs.o, P, Ld["free"], K.whitening(S)[0])
                        pd.to_pickle(fit, cache)
                    cm.append(AB.calib_metrics(rs, loc, fit, S))
                summ["calib"][f"{s}|{loc}"] = cm
                row += f" {cm[0]['cv']:.4f} / {cm[1]['cv']:.4f} | {cm[0]['xrem']:.4f} / {cm[1]['xrem']:.4f} |"
            print(row, flush=True)

    # Synthetic grid with the candidate floors.
    if a.synth:
        df = pd.read_csv(a.synth)
        rec = AB.Records(df, const)
        e = df[["ex", "ey"]].to_numpy()
        keys = df[["generator", "condition", "localizer", "sigma"]].astype(str).agg("|".join, axis=1).to_numpy()
        res = {}
        for tag, fl in (("ship", ship), ("f_m", new), ("f_x", tab["f_x"])):
            S = rec.cov("a", fl)
            q = np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e) / 2
            res[tag] = pd.Series(q).groupby(keys).mean()
        sc = pd.DataFrame(res)
        loc_of = np.array([k.split("|")[2] for k in sc.index])
        print("\n### Synthetic sigma0^2 grid: median (min-max) per localizer\n")
        for loc in AB.DUMP.values():
            d = sc[loc_of == loc]
            summ["synth"][loc] = {k: [float(d[k].median()), float(d[k].min()), float(d[k].max()), int((d[k] > 2).sum())] for k in d}
            print(f"| {loc} | " + " | ".join(f"{k} {d[k].median():.2f} ({d[k].min():.2f}-{d[k].max():.2f}, >2: {(d[k] > 2).sum()})" for k in d) + " |")

    with open(C.OUT / "crossfloor.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
