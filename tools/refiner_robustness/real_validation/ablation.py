"""Systematic-model ablation (UQ-36), held-out floor (UQ-37) and the weighted
calibration's sigma0 (UQ-38). Pre-registered criteria and results:
docs/reference/uncertainty-model.md, "Systematic-model ablation and held-out
floor".

Inputs (all local, never committed):

- ``detect.py`` and ``calibrate.py --set all`` / ``--set chess`` outputs in
  ``$RV_OUT`` (or ``out/``);
- per-corner records from the core localizers (``uncertainty_eval.rs``,
  ``--ignored``): ``dump_ablation_real`` on the same images (written as
  ``<cam>__<file stem>.u8``), ``dump_ablation_synth`` (the ``sigma0_sweep``
  grid) and ``dump_systematic`` (the synthetic calibration records).

Usage::

    python ablation.py --real ablation_real.csv --synth ablation_synth.csv \\
        --systematic systematic.csv

Each record carries the noise term ``N`` at the image's sigma, the shipped
systematic term ``S``, the window offset, the edge width and the window grid.
The variants re-score the same corners:

- ``a``: ``N + S`` (shipped; ``ship`` rows use the shipped floors, ``a``
  rows the floors refitted here, to check they reproduce);
- ``b``: ``N + R + f^2 I`` (``R``: the Radon peak fit's 8-bit rounding term,
  the part of ``S`` outside the knot model);
- ``c``: ``b`` plus the window-offset model with one ``(g, rho)`` per
  localizer, fitted on the pooled synthetic records (``fit_systematic.fit``).

Prints markdown tables and writes ``ablation.json`` to the output directory.
Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analyze as A
import calibrate as K
import common as C

sys.path.insert(0, str(C.HERE.parent))
import fit_systematic as FS  # noqa: E402

CORE = C.HERE.parents[2] / "crates" / "chess-corners-core" / "src"
# Systematic tables, by the dump's localizer names (the Rust sources are the
# single definition of the knots and floors).
TABLES = {
    "CoM": "refine/center_of_mass.rs",
    "Forstner": "refine/forstner.rs",
    "SaddlePoint": "refine/saddle_point.rs",
    "RadonPeak": "detect/radon/detect.rs",
}
# Harness localizer -> dump localizer.
DUMP = {"CoM": "CoM", "Forstner": "Forstner", "SaddlePoint": "SaddlePoint", "Radon": "RadonPeak", "Radon+SP": "RadonSaddle"}
VARIANTS = ("a", "b", "c")
EYE = np.eye(2)[None]


def shipped_table(name: str):
    src = (CORE / TABLES[name]).read_text()
    block = src[src.index("const SYSTEMATIC"):]
    block = block[: block.index("};")]
    knots = np.array([[float(v) for v in m] for m in re.findall(r"knot\(([\d.]+),\s*(-?[\d.]+),\s*([\d.]+)\)", block)])
    floor = float(re.search(r"floor:\s*([\d.]+)", block).group(1))
    return knots, floor


def sys_cov(u, w, h, knots, floor):
    """``SystematicModel::covariance`` for rows of offsets ``u`` (N, 2),
    widths ``w`` (NaN = unknown) and grids ``h``."""
    W = np.where(np.isfinite(w), w, knots[0, 0])
    g = np.interp(W, knots[:, 0], knots[:, 1])
    rho = np.interp(W, knots[:, 0], knots[:, 2])
    P = h * h / 12
    den = g * g * P + rho * rho
    pos = den > 0
    shrink = np.where(pos, g * P / np.where(pos, den, 1), 0.0)
    spread = np.where(pos, P * rho * rho / np.where(pos, den, 1), P) + floor * floor
    e = u * (1 - shrink)[:, None]
    S = e[:, :, None] * e[:, None, :]
    S[:, 0, 0] += spread
    S[:, 1, 1] += spread
    return S


def cov3(df, p):
    S = np.empty((len(df), 2, 2))
    S[:, 0, 0] = df[p + "xx"].to_numpy()
    S[:, 0, 1] = S[:, 1, 0] = df[p + "xy"].to_numpy()
    S[:, 1, 1] = df[p + "yy"].to_numpy()
    return S


class Records:
    """Per-corner model ingredients, with each row's table and the parts the
    variants recombine."""

    def __init__(self, df: pd.DataFrame, const: dict):
        self.N = cov3(df, "n")
        self.S = cov3(df, "s")
        self.u = df[["ux", "uy"]].to_numpy()
        self.w = df["width"].to_numpy()
        self.h = df["grid"].to_numpy()
        loc = df["localizer"].to_numpy()
        n = len(df)
        self.table = np.empty(n, object)
        for name in TABLES:
            self.table[loc == name] = name
        # Radon + SaddlePoint: the saddle-point model where the refinement
        # moved the peak (offset 0, exact reproduction), else the peak fit's.
        rs = loc == "RadonSaddle"
        if rs.any():
            kn, fl = shipped_table("SaddlePoint")
            sp = sys_cov(np.zeros((rs.sum(), 2)), self.w[rs], self.h[rs], kn, fl)
            self.table[rs] = np.where(np.abs(sp - self.S[rs]).max((1, 2)) < 1e-5, "SaddlePoint", "RadonPeak")
        self.ship_floor = np.zeros(n)
        self.knot_part = np.zeros((n, 2, 2))  # S without floor and rounding
        self.const_part = np.zeros((n, 2, 2))
        for name in TABLES:
            m = self.table == name
            if not m.any():
                continue
            kn, fl = shipped_table(name)
            # The saddle-point fit re-centres its window: no offset.
            u = np.zeros((m.sum(), 2)) if name == "SaddlePoint" else self.u[m]
            self.ship_floor[m] = fl
            self.knot_part[m] = sys_cov(u, self.w[m], self.h[m], kn, 0.0)
            g, rho = const[name]
            self.const_part[m] = sys_cov(u, np.full(m.sum(), np.nan), self.h[m], np.array([[1.0, g, rho]]), 0.0)
        # Everything in S that is neither the knot model nor the floor: the
        # Radon peak fit's rounding of its upsampled samples.
        self.R = self.S - self.knot_part - (self.ship_floor**2)[:, None, None] * EYE
        self.repro = np.abs(self.R).max((1, 2))

    def base(self, v: str):
        """The variant's covariance without its floor."""
        if v == "a":
            return self.N + self.knot_part + self.R
        if v == "b":
            return self.N + self.R
        return self.N + self.R + self.const_part

    def cov(self, v: str, floor: dict | float):
        f = floor if np.isscalar(floor) else np.array([floor[t] for t in self.table])
        return self.base(v) + (np.asarray(f) ** 2).reshape(-1, 1, 1) * EYE


def constant_models(path: Path) -> dict:
    """One ``(g, rho)`` per table from all edge widths pooled."""
    df = pd.read_csv(path).dropna(subset=["width"])
    out = {}
    for loc, h, fit_gain in FS.MODELS.values():
        g, rho, rms, n = FS.fit(df[df.localizer == loc], h, fit_gain)
        out[loc] = (g, rho)
        print(f"constant model {loc:12s}: g {g:.3f} rho {rho:.4f} (noise-free rms {rms:.4f}, n {n})")
    return out


# ---------------------------------------------------------------------------
# Real set
# ---------------------------------------------------------------------------


def align(o: dict, loc: str, real: pd.DataFrame, const: dict):
    """Records of the core dump in observation order (NaN rows unmatched)."""
    P = o["xy"][loc]
    d = real[real.localizer == DUMP[loc]]
    rows = np.full(len(P), -1)
    groups = {k: g.index.to_numpy() for k, g in d.groupby("condition")}
    for c in (0, 1):
        for v, tok in enumerate(o["tokens"]):
            s = np.where((o["cam"] == c) & (o["view"] == v))[0]
            key = f"{C.CAMS[c]}__Cam{c + 1}_{tok}"
            if len(s) == 0 or key not in groups:
                continue
            gi = groups[key]
            j = C.match(d.loc[gi, ["x", "y"]].to_numpy(), P[s], radius=1e-3)
            rows[s[j >= 0]] = gi[j[j >= 0]]
    ok = rows >= 0
    full = real.loc[rows[ok]].reset_index(drop=True)
    rec = Records(full, const)
    return rec, ok


class RealSet:
    def __init__(self, name: str, real: pd.DataFrame, const: dict, nb):
        d = pickle.load(open(C.OUT / f"calib_{name}.pkl", "rb"))
        self.name, self.d, self.o, self.nb = name, d, d["obs"], nb
        o = self.o
        self.fold = o["fold"][o["view"]]
        self.locs = {}
        for loc in DUMP:
            if loc not in d["locs"]:
                continue
            rec, ok = align(o, loc, real, const)
            cov_b = o["cov"][loc][ok]
            dev = np.abs(rec.N + rec.S - cov_b).max((1, 2)) / np.abs(cov_b).max((1, 2))
            P = o["xy"][loc]
            oo = dict(P=P, cam=o["cam"], id=o["id"], shape=o["shape"])
            rig = d["locs"][loc]["rig"]
            allm = np.ones(len(P), bool)
            local, dof_row = A.local_remainder(oo, o, rig["res"], rig["Jb"], allm, nb)
            dec = A.decompose(oo, rig["res"], rig["Jb"], allm, nb)
            xrem = A.xfold_remainder(oo, rig["res"], rig["Jb"], self.fold, nb)
            self.locs[loc] = dict(rec=rec, ok=ok, oo=oo, rem=dec["rem"], xrem=xrem, local=local, dof=float(dof_row.mean()))
            print(
                f"[{name}] {loc:9s} matched {ok.mean():.4f} of {len(P)}; |Sigma_a - bindings| / |Sigma| max {np.nanmax(dev):.1e}; "
                f"model reproduction max {rec.repro.max():.1e} (Radon rounding rows: {(rec.repro > 1e-5).sum()})"
            )

    def full_cov(self, loc, v, floor):
        """(N_obs, 2, 2) covariance in observation order (NaN unmatched)."""
        L = self.locs[loc]
        S = np.full((len(L["ok"]), 2, 2), np.nan)
        S[L["ok"]] = L["rec"].cov(v, floor)
        return S

    def fit_floor(self, loc, v):
        L = self.locs[loc]
        ok = L["ok"]
        return C.fit_floor(L["local"][ok], L["rec"].base(v), L["dof"])


def stats(e, S, dof=0.0):
    ok = np.isfinite(S).all(axis=(1, 2)) & np.isfinite(e).all(1)
    ps = np.sqrt(0.5 * (S[ok, 0, 0] + S[ok, 1, 1]))
    return dict(s0=C.sigma0_sq(e[ok], S[ok], dof), rho=float(spearmanr(ps, np.linalg.norm(e[ok], axis=1))[0]), med=float(np.median(ps)))


def calib_metrics(rs: RealSet, loc: str, fit: dict, S: np.ndarray | None):
    """Held-out CV RMS, cross-fold remainder RMS, fold-to-fold |dfx| per
    camera, and sigma0^2 of the fit and of the CV residuals under ``S``."""
    o = rs.o
    L = rs.locs[loc]
    xw = A.xfold_remainder(L["oo"], fit["res"], fit["Jb"], rs.fold, rs.nb)
    n = len(fit["res"])
    npar = 24 + 6 * len(o["tokens"])
    out = dict(cv=A.rms(fit["cv"]), xrem=A.rms(xw))
    fp = fit["fold_p"]
    out["dfx"] = [abs(fp[0][9 * c] - fp[1][9 * c]) for c in (0, 1)]
    out["fx"] = [float(fit["p"][9 * c]) for c in (0, 1)]
    if S is None:
        out["s0_fit"] = float((fit["res"] ** 2).sum() / (2 * n - npar))  # px^2 per axis
        out["s0_cv"] = float((fit["cv"] ** 2).sum(1).mean() / 2)
    else:
        # Rows without a covariance (a few unmatched corners) are left out.
        ok = np.isfinite(S).all(axis=(1, 2))
        q = lambda r: np.einsum("ni,nij,nj->n", r[ok], np.linalg.inv(S[ok]), r[ok])  # noqa: E731
        out["s0_fit"] = float(q(fit["res"]).sum() / (2 * ok.sum() - npar))
        out["s0_cv"] = float(q(fit["cv"]).mean() / 2)
    return out


def held_out(rs: RealSet, loc: str, v: str):
    """Floor fitted on a training part, sigma0^2 of the held-out part's local
    remainder with the smooth and board terms from the training part."""
    o = rs.o
    L = rs.locs[loc]
    oo, ok, rec = L["oo"], L["ok"], L["rec"]
    rig = rs.d["locs"][loc]["rig"]
    r, Jb = rig["res"], rig["Jb"]
    base = np.full((len(ok), 2, 2), np.nan)
    base[ok] = rec.base(v)
    st = o["stage"][o["view"]]
    splits = {"cam1 -> cam2": o["cam"] == 0, "cam2 -> cam1": o["cam"] == 1}
    for f in (0, 1):
        splits[f"fold {1 - f} -> {f}"] = rs.fold != f
    for lv in C.LEVELS:
        splits[f"Rx {lv:+.0f} out"] = st[:, 0] != lv
    out = {}
    for name, train in splits.items():
        test = ~train
        e_tr, dof_tr = A.local_remainder(oo, o, r, Jb, train, rs.nb)
        m = train & ok
        f = C.fit_floor(e_tr[m], base[m], float(dof_tr[train].mean()))
        dec = A.decompose(oo, r, Jb, train, rs.nb)
        smooth = dec["smooth"]
        if name.startswith("cam"):
            # The held-out camera's own lens field (its smooth term is not
            # predictable from the other camera); target term from the other.
            pred = A.Smooth(oo["shape"], rs.nb).fit(oo["P"], o["cam"], r - dec["board"], test)
            smooth = np.where(test[:, None], pred(oo["P"], o["cam"]), smooth)
        rem = r - smooth - dec["board"]
        pvs = A.per_view_smooth(o, oo["P"], rem)
        e = rem - pvs["fit"]
        m = test & ok
        S = base[m] + f * f * EYE
        dof = float(pvs["dof_row"][test].mean())
        # need: the floor the held-out part itself would take.
        out[name] = dict(floor=f, s0=C.sigma0_sq(e[m], S, dof), need=C.fit_floor(e[m], base[m], dof))
    return out


# ---------------------------------------------------------------------------
# Synthetic grid
# ---------------------------------------------------------------------------


def synth_scores(path: Path, const: dict, floors: dict):
    df = pd.read_csv(path)
    rec = Records(df, const)
    print(f"synthetic records: {len(df)}; model reproduction max {rec.repro.max():.1e}")
    e = df[["ex", "ey"]].to_numpy()
    keys = df[["generator", "condition", "localizer", "sigma"]].astype(str).agg("|".join, axis=1).to_numpy()
    out = {}
    ship = {t: shipped_table(t)[1] for t in TABLES}
    for v in VARIANTS + ("ship",):
        S = rec.cov("a", ship) if v == "ship" else rec.cov(v, floors[v])
        q = np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e) / 2
        out[v] = pd.Series(q).groupby(keys).mean()
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--systematic", required=True)
    ap.add_argument("--knots", default="5,4")
    ap.add_argument("--skip-calib", action="store_true", help="skip the weighted calibrations")
    a = ap.parse_args()
    nb = tuple(int(x) for x in a.knots.split(","))
    const = constant_models(Path(a.systematic))
    real = pd.read_csv(a.real)
    sets = {s: RealSet(s, real, const, nb) for s in ("all", "chess")}
    summ = dict(const={k: list(v) for k, v in const.items()}, floors={}, real={}, heldout={}, calib={}, synth={}, width={})

    # Floors: local-remainder sigma0^2 = 1, max over the two sets.
    floors = {v: {} for v in VARIANTS}
    print("\n### Floors (px per axis): all / chess -> max\n")
    print("| localizer | " + " | ".join(VARIANTS) + " |\n|---|---|---|---|")
    per = {}
    for loc in DUMP:
        cells = []
        for v in VARIANTS:
            fs = [rs.fit_floor(loc, v) for rs in sets.values() if loc in rs.locs]
            per[(loc, v)] = max(fs)
            cells.append(" / ".join(f"{x:.4f}" for x in fs) + f" -> **{max(fs):.4f}**")
        print(f"| {loc} | " + " | ".join(cells) + " |")
    # A table's floor: the max over the harness localizers that use it
    # (SaddlePoint's from both seeds, the Radon peak fit's from the peak fit).
    for v in VARIANTS:
        for t, locs in {"CoM": ["CoM"], "Forstner": ["Forstner"], "SaddlePoint": ["SaddlePoint", "Radon+SP"], "RadonPeak": ["Radon"]}.items():
            floors[v][t] = max(per[(loc, v)] for loc in locs)
    summ["floors"] = floors
    print("\nper table:", json.dumps(floors))

    # Ranking and sigma0^2 on the real set.
    print("\n### Real set: sigma0^2 rem / x-fold / local; Spearman rem / x-fold / local; median sigma\n")
    print("| set | localizer | variant | sigma0^2 | Spearman | median sigma px |\n|---|---|---|---|---|---|")
    for sname, rs in sets.items():
        for loc, L in rs.locs.items():
            for v in VARIANTS + ("ship",):
                S = rs.full_cov(loc, "a", {t: shipped_table(t)[1] for t in TABLES}) if v == "ship" else rs.full_cov(loc, v, floors[v])
                st = [stats(L["rem"], S), stats(L["xrem"], S), stats(L["local"], S, L["dof"])]
                summ["real"][f"{sname}|{loc}|{v}"] = st
                print(
                    f"| {sname} | {loc} | {v} | {' / '.join(f'{x['s0']:.2f}' for x in st)} | "
                    f"{' / '.join(f'{x['rho']:.3f}' for x in st)} | {st[0]['med']:.4f} |"
                )

    # Edge width of the real corners against the knot range.
    print("\n### Edge width of the real corners (SaddlePoint windows), px\n")
    print("| set | camera | n | p5 / p25 / p50 / p75 / p95 | below 1.09 | above 1.68 |\n|---|---|---|---|---|---|")
    for sname, rs in sets.items():
        L = rs.locs["SaddlePoint"]
        cam = rs.o["cam"][L["ok"]]
        for c in (0, 1):
            w = L["rec"].w[cam == c]
            w = w[np.isfinite(w)]
            pc = np.percentile(w, [5, 25, 50, 75, 95])
            summ["width"][f"{sname}|cam{c + 1}"] = pc.tolist()
            print(f"| {sname} | cam{c + 1} | {len(w)} | {' / '.join(f'{x:.2f}' for x in pc)} | {(w < 1.09).mean():.1%} | {(w > 1.68).mean():.1%} |")

    # Held-out floor.
    print("\n### Held-out floor: floor fitted on the training part -> held-out local-remainder sigma0^2\n")
    for sname, rs in sets.items():
        for loc in rs.locs:
            for v in VARIANTS:
                h = held_out(rs, loc, v)
                summ["heldout"][f"{sname}|{loc}|{v}"] = h
                s0 = np.array([x["s0"] for x in h.values()])
                fl = np.array([x["floor"] for x in h.values()])
                print(
                    f"[{sname}] {loc:9s} {v}: floor {fl.min():.4f}-{fl.max():.4f}; held-out sigma0^2 {s0.min():.2f}-{s0.max():.2f} "
                    f"(" + ", ".join(f"{k} {x['s0']:.2f} [needs {x['need']:.4f}]" for k, x in h.items()) + ")"
                )

    # Weighted calibrations.
    if not a.skip_calib:
        print("\n### Rig calibration: uniform vs Sigma-weighted\n")
        print("| set | localizer | weights | CV RMS | x-fold rem RMS | fx cam1 / cam2 px | fold |dfx| cam1 / cam2 px | sigma0^2 fit / CV |\n|---|---|---|---|---|---|---|---|")
        for sname, rs in sets.items():
            for loc in rs.locs:
                Ld = rs.d["locs"][loc]
                P = rs.o["xy"][loc]
                ship = rs.full_cov(loc, "a", {t: shipped_table(t)[1] for t in TABLES})
                rows = {"uniform": calib_metrics(rs, loc, Ld["rig"], None)}
                if "rig_w" in Ld:
                    rows["a"] = calib_metrics(rs, loc, Ld["rig_w"], ship)
                for v in ("b", "c"):
                    S = rs.full_cov(loc, v, floors[v])
                    cache = C.OUT / f"ablation_rig_{sname}_{loc}_{v}.pkl"
                    if cache.exists():
                        fit = pickle.load(open(cache, "rb"))
                    else:
                        fit = K.rig_fit_cv(rs.o, P, Ld["free"], K.whitening(S)[0])
                        pickle.dump(fit, open(cache, "wb"))
                    rows[v] = calib_metrics(rs, loc, fit, S)
                summ["calib"][f"{sname}|{loc}"] = rows
                for k, m in rows.items():
                    print(
                        f"| {sname} | {loc} | {k} | {m['cv']:.4f} | {m['xrem']:.4f} | {m['fx'][0]:.0f} / {m['fx'][1]:.0f} | {m['dfx'][0]:.1f} / {m['dfx'][1]:.1f} | "
                        f"{m['s0_fit']:.3f} / {m['s0_cv']:.3f} |",
                        flush=True,
                    )

    # Synthetic grid.
    sc = synth_scores(Path(a.synth), const, floors)
    print("\n### Synthetic sigma0^2 grid (per localizer: median / p90 / max; worst |ln|; cells > 2; new cells > 2)\n")
    print("| localizer | variant | cells | median | p90 | max | worst abs ln | > 2 | new > 2 |\n|---|---|---|---|---|---|---|---|---|")
    loc_of = np.array([k.split("|")[2] for k in sc.index])
    for loc in DUMP.values():
        d = sc[loc_of == loc]
        for v in ("ship",) + VARIANTS:
            x = d[v]
            new = int(((x > 2) & (d["ship"] <= 2)).sum())
            row = dict(n=len(x), median=x.median(), p90=x.quantile(0.9), max=x.max(), worst_ln=float(np.abs(np.log(x)).max()), over2=int((x > 2).sum()), new_over2=new)
            summ["synth"][f"{loc}|{v}"] = row
            if new:
                print(f"  new > 2 under {v}: " + ", ".join(f"{k.split('|')[1]} s{k.split('|')[3]} {d.loc[k, 'ship']:.2f}->{x[k]:.2f}" for k in x.index[(x > 2) & (d["ship"] <= 2)]))
            print(f"| {loc} | {v} | {row['n']} | {row['median']:.2f} | {row['p90']:.2f} | {row['max']:.2f} | {row['worst_ln']:.2f} | {row['over2']} | {new} |")
        for gen in ("optical", "synth"):
            g = d[[k.split("|")[0] == gen for k in d.index]]
            summ["synth"][f"{loc}|ship|{gen}"] = dict(n=len(g), median=g["ship"].median())
            print(f"  {loc} shipped, {gen} generator only: {len(g)} cells, median {g['ship'].median():.2f}")
    with open(C.OUT / "ablation.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
