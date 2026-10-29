"""Score candidate orientation-σ definitions from the study harness CSVs.

Usage: python analyze.py CSV [CSV ...] [--fit-gen optical]

Each CSV comes from ``cargo run --release -p chess-corners --example
orientation_sigma_study`` (``optical`` or ``files`` mode). Candidates:

- ``cur``     shipped ``AxisEstimate::sigma``;
- ``cur*k``   shipped value times one constant per method;
- ``b``       residual-scaled covariance ``sqrt(c_unit·s2)`` (no lookup table);
- ``a``       noise-propagated ``sqrt(c_pix·σ_pix²)``, σ_pix = true noise
              (with 1/12 quantization) -- an oracle;
- ``a_est``   the same with the detector's own noise estimate;
- ``X+f``     ``sqrt(X² + f²)`` with a per-method floor ``f``.

The constants (``k``, ``f``) are fitted on ``--fit-gen`` only (pooled
coverage(|z|<1) = 0.683) and applied unchanged to the other generators.
DiskFit's inflated-fallback rows keep the 10° minimum on every candidate.
Prints markdown tables. z = err / σ̂, err = undirected angle error.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

COND = ["gen", "cell", "blur", "rot_deg", "shear_deg", "noise", "shading"]
INFLATED = np.radians(10.0)
SENTINEL = np.pi - 1e-3


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["cond"] = d[COND].apply(lambda c: c.map(str)).agg("|".join, axis=1)
    d["abs_err"] = d.err.abs()
    infl = d.path.eq("inflated").to_numpy()
    q = 1.0 / 12.0

    def keep_inflation(v):
        v = np.asarray(v, float)
        return np.where(infl, np.maximum(v, INFLATED), v)

    d["cur"] = d.sigma_cur
    d["b"] = keep_inflation(np.sqrt(d.c_unit * d.s2))
    d["a"] = keep_inflation(np.sqrt(d.c_pix * (d.noise.astype(float) ** 2 + q)))
    d["a_est"] = keep_inflation(np.sqrt(d.c_pix * d.noise_est**2))
    return d


def valid(d, col):
    v = d[col].to_numpy()
    return np.isfinite(v) & (v > 0) & (v < SENTINEL) & np.isfinite(d.err.to_numpy())


def cov1(err, sig):
    return float(np.mean(np.abs(err) < sig))


def fit_scale(d, col):
    m = valid(d, col)
    e, s = d.err[m].to_numpy(), d[col][m].to_numpy()
    return brentq(lambda k: cov1(e, k * s) - 0.683, 1e-3, 1e3)


def fit_floor(d, col):
    m = valid(d, col)
    e, s = d.err[m].to_numpy(), d[col][m].to_numpy()
    g = lambda f: cov1(e, np.sqrt(s * s + f * f)) - 0.683  # noqa: E731
    return 0.0 if g(0.0) >= 0 else brentq(g, 0.0, 1.0)


def stats(d, col):
    m = valid(d, col)
    x = d[m]
    e, s = x.err.to_numpy(), x[col].to_numpy()
    z = e / s
    out = dict(
        n=int(m.sum()),
        dropped=int((~m).sum()),
        std_z=float(np.std(z)),
        rstd_z=float(1.4826 * np.median(np.abs(z))),
        cov1=float(np.mean(np.abs(z) < 1)),
        cov2=float(np.mean(np.abs(z) < 2)),
        rho=float(spearmanr(s, np.abs(e)).statistic),
        med_sig_deg=float(np.degrees(np.median(s))),
        med_err_deg=float(np.degrees(np.median(np.abs(e)))),
    )
    rhos, cells_ok, ncell = [], 0, 0
    for _, g in x.groupby("cond"):
        if len(g) < 20:
            continue
        ncell += 1
        c = cov1(g.err.to_numpy(), g[col].to_numpy())
        cells_ok += 0.45 <= c <= 0.90
        if g[col].nunique() > 2:
            r = spearmanr(g[col], g.abs_err).statistic
            if np.isfinite(r):
                rhos.append(r)
    out["rho_cell"] = float(np.mean(rhos)) if rhos else np.nan
    out["cells_ok"] = cells_ok / ncell if ncell else np.nan
    return out


def r1_pass(s):
    return 0.55 <= s["cov1"] <= 0.80 and s["cov2"] >= 0.88 and 0.75 <= s["rstd_z"] <= 1.33 and s["cells_ok"] >= 0.75


def breakdown(d, col, by):
    rows = {}
    for key, g in d.groupby(by):
        m = valid(g, col)
        rows[key] = cov1(g.err[m].to_numpy(), g[col][m].to_numpy()) if m.any() else np.nan
    return rows


def gate_table(d, gens, cands, gate=np.radians(15.0), gross=np.radians(5.0)):
    """What a downstream gate `σ̂ <= gate` does with each candidate: rejected
    share, gross-error (|err| > gross) share among kept and rejected, the
    AUC of σ̂ as a gross-error detector, and the gate that rejects the same
    share as the shipped σ̂ does at `gate`."""
    from scipy.stats import mannwhitneyu

    print(f"\n| generator | method | σ̂ | rejected at {np.degrees(gate):.0f}° | gross among kept | gross among rejected | AUC(gross) | equivalent gate (°) |")
    print("|---|---|---|---|---|---|---|---|")
    for gen in gens:
        for meth in ("ring", "disk"):
            x = d[(d.gen == gen) & (d.method == meth)]
            if x.empty:
                continue
            ref = x.cur.to_numpy()
            ref_rej = np.mean(~(ref <= gate))
            for c in cands:
                if c not in x:
                    continue
                s = x[c].to_numpy()
                ok = np.isfinite(s)
                if not ok.any():
                    continue
                s, e = s[ok], x.abs_err.to_numpy()[ok]
                rej = ~(s <= gate)
                g = e > gross
                auc = np.nan
                if g.any() and (~g).any():
                    auc = mannwhitneyu(s[g], s[~g]).statistic / (g.sum() * (~g).sum())
                eq = np.degrees(np.quantile(s, 1 - ref_rej)) if 0 < ref_rej < 1 else np.nan
                kept_g = g[~rej].mean() if (~rej).any() else np.nan
                rej_g = g[rej].mean() if rej.any() else np.nan
                print(f"| {gen} | {meth} | {c} | {rej.mean():.2%} | {kept_g:.2%} | {rej_g:.0%} | {auc:.3f} | {eq:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--fit-gen", default="optical")
    ap.add_argument("--breakdown", action="store_true")
    a = ap.parse_args()
    d = load(a.csv)
    gens = list(dict.fromkeys(d.gen))
    cands = ["cur", "cur*k", "b", "b+f", "a", "a+f", "a_est", "a_est+f"]
    consts = {}
    for meth in ("ring", "disk"):
        fit = d[(d.gen == a.fit_gen) & (d.method == meth)]
        if fit.empty:
            continue
        k = fit_scale(fit, "cur")
        sel = d.method == meth
        d.loc[sel, "cur*k"] = d.loc[sel, "cur"] * k
        consts[(meth, "cur*k")] = f"k = {k:.3f}"
        for base in ("b", "a", "a_est"):
            f = fit_floor(fit, base)
            d.loc[sel, f"{base}+f"] = np.sqrt(d.loc[sel, base] ** 2 + f * f)
            consts[(meth, f"{base}+f")] = f"f = {np.degrees(f):.2f}°"
    print(f"Constants fitted on `{a.fit_gen}`: " + "; ".join(f"{m} {c}: {v}" for (m, c), v in consts.items()))
    print()
    print("| generator | method | σ̂ | N | std(z) | robust std(z) | P(|z|<1) | P(|z|<2) | cells in band | Spearman pooled | Spearman in-cell | median σ̂ (°) | R1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for gen in gens:
        for meth in ("ring", "disk"):
            x = d[(d.gen == gen) & (d.method == meth)]
            if x.empty:
                continue
            for c in cands:
                if c not in x or not valid(x, c).any():
                    continue
                s = stats(x, c)
                print(f"| {gen} | {meth} | {c} | {s['n']} | {s['std_z']:.2f} | {s['rstd_z']:.2f} | {s['cov1']:.2f} | "
                      f"{s['cov2']:.2f} | {s['cells_ok']:.0%} | {s['rho']:.3f} | {s['rho_cell']:.3f} | "
                      f"{s['med_sig_deg']:.2f} | {'pass' if r1_pass(s) else 'fail'} |")
    gate_table(d, gens, ["cur", "b", "a+f", "a_est"])
    if a.breakdown:
        for gen in gens:
            for by in ("noise", "cell", "blur", "rot_deg", "shear_deg", "shading"):
                x = d[(d.gen == gen)]
                if not 2 <= x[by].nunique() <= 8:
                    continue
                print(f"\nP(|z|<1) by {by}, {gen}")
                keys = sorted(x[by].unique(), key=lambda v: (isinstance(v, str), v))
                print("| method | σ̂ | " + " | ".join(str(k) for k in keys) + " |")
                print("|---|---|" + "---|" * len(keys))
                for meth in ("ring", "disk"):
                    for c in ("cur", "b", "a+f", "b+f"):
                        xm = x[x.method == meth]
                        if c not in xm:
                            continue
                        r = breakdown(xm, c, by)
                        print(f"| {meth} | {c} | " + " | ".join(f"{r.get(k, np.nan):.2f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
