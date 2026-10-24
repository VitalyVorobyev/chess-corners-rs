"""UQ-27 real-data gate: asymmetry term vs the shipped ``Propagated`` model.

Usage (after ``../real_validation/detect.py <root>`` and
``export_uq27.py <root>``):

    python gate_uq27.py <dataset_root> [--boot 1000]

Criteria are pre-registered in ``docs/reference/asymmetry-term.md``
("Real-data gate"). Per corner set (``all``, ``chess``) and localizer:

1. Uniform rig calibration and the UQ-30 decomposition of its residual
   (``analyze.decompose``): in-sample remainder, cross-fold remainder (lens
   field and target term from the other fold of views), local remainder (the
   per-view cubic removed as well; sigma0^2 divided by ``1 - dof``).
2. Covariances: ``P`` = bindings (``Propagated``, floor 0.053 px);
   ``P*`` = ``P`` with its floor refitted; ``A`` = ``P - f_P^2 I + T + f_A^2 I``
   (floor refitted with ``T`` in place); ``P+T``. Floors: the value making the
   local-remainder sigma0^2 1, max over the two sets (the shipped rule),
   fitted on SaddlePoint and used for both SaddlePoint refiners.
3. sigma0^2, Spearman(sqrt(tr S / 2), |e|), Gaussian NLL per point on the
   local remainder.
4. Sigma-whitened rig calibration per covariance: held-out (2-fold CV)
   residual RMS and the cross-fold remainder of the weighted fit's residual,
   with a view-bootstrap 95% CI of the difference to the ``P``-weighted fit.
5. beta against the decomposition: variance of the board term / per-view
   smooth part / smooth field / remainder explained by a linear map of beta
   (and the reverse), and the slope of the remainder on beta.

Writes ``../real_validation/out/gate_uq27.json``; prints markdown tables.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RV = HERE.parent / "real_validation"
sys.path.insert(0, str(RV))

import analyze as A  # noqa: E402
import calibrate as K  # noqa: E402
import common as C  # noqa: E402

F_SHIPPED = 0.053
NB = (5, 4)
SETS = {"all": ("SaddlePoint", "Radon+SP"), "chess": ("SaddlePoint",)}
I2 = np.eye(2)[None]


def rms(r):
    r = r[np.isfinite(r[:, 0])]
    return float(np.sqrt((r**2).sum(1).mean()))


def attach(obs, loc):
    """Asymmetry term and beta per observation of ``obs`` (T = 0 where the
    fit is invalid or missing)."""
    z = np.load(RV / "out" / f"asym_{loc.replace('+', '_')}.npz", allow_pickle=True)
    key = {(str(c), str(t), int(i)): k for k, (c, t, i) in enumerate(zip(z["cam"], z["token"], z["id"]))}
    n = len(obs["cam"])
    T = np.zeros((n, 2, 2))
    beta = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    for j in range(n):
        k = key.get((C.CAMS[obs["cam"][j]], obs["tokens"][obs["view"][j]], int(obs["id"][j])))
        if k is not None and z["valid"][k]:
            T[j], beta[j], valid[j] = z["T"][k], z["beta"][k], True
    return T, beta, valid


def decomp(obs, P, res, Jb):
    oo = dict(P=P, cam=obs["cam"], id=obs["id"], shape=obs["shape"])
    allm = np.ones(len(P), bool)
    fold = obs["fold"][obs["view"]]
    dec = A.decompose(oo, res, Jb, allm, NB)
    x = np.full_like(res, np.nan)
    for f in (0, 1):
        x[fold == f] = A.decompose(oo, res, Jb, fold != f, NB)["rem"][fold == f]
    pvs = A.per_view_smooth(obs, P, dec["rem"])
    return dict(rem=dec["rem"], x=x, local=dec["rem"] - pvs["fit"], dof=pvs["dof_frac"], board=dec["board"], smooth=dec["smooth"], pv=pvs["fit"])


def s0(e, S, dof=0.0):
    q = np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e)
    return float(q.mean() / 2 / (1 - dof))


def nll(e, S, dof=0.0):
    q = np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e) / (1 - dof)
    return float(np.mean(0.5 * q + 0.5 * np.log(np.linalg.det(S)) + np.log(2 * np.pi)))


def sp(e, S):
    ps = np.sqrt(0.5 * (S[:, 0, 0] + S[:, 1, 1]))
    return float(spearmanr(ps, np.linalg.norm(e, axis=1))[0])


def floor_for(e, S0, dof):
    """Isotropic floor f with sigma0^2(S0 + f^2 I) = 1 (0 if already <= 1)."""
    g = lambda f: s0(e, S0 + f * f * I2, dof) - 1.0  # noqa: E731
    return brentq(g, 0.0, 1.0) if g(0.0) > 0 else 0.0


def ve(a, b):
    """Fraction of the variance of a (n, 2) explained by a linear map of b."""
    return A.corr2(a, b)


def boot_ci(view, a, b, nboot, rng):
    """95% view-bootstrap CI of rms(a) - rms(b) (rows NaN-free in both)."""
    ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    v = view[ok]
    qa = (a[ok] ** 2).sum(1)
    qb = (b[ok] ** 2).sum(1)
    uv = np.unique(v)
    ix = np.searchsorted(uv, v)
    sa = np.bincount(ix, qa, len(uv))
    sb = np.bincount(ix, qb, len(uv))
    cnt = np.bincount(ix, minlength=len(uv)).astype(float)
    d = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, len(uv), len(uv)), minlength=len(uv)).astype(float)
        n = (w * cnt).sum()
        d.append(np.sqrt((w * sa).sum() / n) - np.sqrt((w * sb).sum() / n))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    prior = json.load(open(Path(a.root) / "calibration.json"))["intrinsic"]
    K0s = [np.array(prior[f"camera{c}"]["matrix"], float) for c in (0, 1)]
    rng = np.random.default_rng(27)

    # 1. Uniform fits and decompositions (cached).
    cache = RV / "out" / "gate_uq27_uniform.pkl"
    if cache.exists():
        U = pickle.load(open(cache, "rb"))
    else:
        U = {}
        for sname, locs in SETS.items():
            obs = K.load_obs(sname, {})
            for loc in locs:
                P = obs["xy"][loc]
                free = K.free_fit(obs, P, K0s)
                rig = K.rig_fit_cv(obs, P, free)
                T, beta, valid = attach(obs, loc)
                U[(sname, loc)] = dict(obs=obs, P=P, free=free, rig=rig, d=decomp(obs, P, rig["res"], rig["Jb"]), S=obs["cov"][loc], T=T, beta=beta, valid=valid)
                print(f"uniform {sname} {loc}: rig {rms(rig['res']):.4f} / CV {rms(rig['cv']):.4f}", flush=True)
        pickle.dump(U, open(cache, "wb"))

    # 2. Floors (SaddlePoint, max over sets).
    fl = {"P*": [], "A": []}
    for sname in SETS:
        u = U[(sname, "SaddlePoint")]
        d, S, T = u["d"], u["S"], u["T"]
        ok = np.isfinite(S).all(axis=(1, 2))
        Sb = S - F_SHIPPED**2 * I2
        fl["P*"].append(floor_for(d["local"][ok], Sb[ok], d["dof"]))
        fl["A"].append(floor_for(d["local"][ok], (Sb + T)[ok], d["dof"]))
    floors = {k: max(v) for k, v in fl.items()}
    print(f"\nfloors per set (all, chess): P* {fl['P*']}, A {fl['A']} -> {floors}")
    out = dict(floors=floors, floors_per_set=fl, sets={})

    for (sname, loc), u in U.items():
        obs, P, d, S, T, beta, valid = u["obs"], u["P"], u["d"], u["S"], u["T"], u["beta"], u["valid"]
        ok = np.isfinite(S).all(axis=(1, 2))
        Sb = S - F_SHIPPED**2 * I2
        covs = {"P": S, "P*": Sb + floors["P*"] ** 2 * I2, "A": Sb + T + floors["A"] ** 2 * I2, "P+T": S + T}
        res = dict(n=int(len(P)), cov_frac=float(ok.mean()), asym_valid=float(valid.mean()),
                   T_sigma_med=float(np.median(np.sqrt(0.5 * np.trace(T[valid], axis1=1, axis2=2)))),
                   T_pos_frac=float((np.trace(T[valid], axis1=1, axis2=2) > 0).mean()),
                   beta_med=float(np.median(np.linalg.norm(beta[valid], axis=1))),
                   rms=dict(rem=rms(d["rem"]), x=rms(d["x"]), local=rms(d["local"])), cov={})
        for name, Sx in covs.items():
            r = {}
            for ek in ("rem", "x", "local"):
                e = d[ek][ok]
                m = np.isfinite(e[:, 0])
                dof = d["dof"] if ek == "local" else 0.0
                r[ek] = dict(s0=s0(e[m], Sx[ok][m], dof), sp=sp(e[m], Sx[ok][m]), nll=nll(e[m], Sx[ok][m], dof))
            r["sigma_med"] = float(np.median(np.sqrt(0.5 * (Sx[ok, 0, 0] + Sx[ok, 1, 1]))))
            # Spearman restricted to corners with a valid asymmetry fit
            m = ok & valid
            r["local_sp_valid"] = sp(d["local"][m], Sx[m])
            res["cov"][name] = r
        # 4. Weighted calibrations.
        fold = obs["fold"][obs["view"]]
        oo = dict(P=P, cam=obs["cam"], id=obs["id"], shape=obs["shape"])
        W = {"uniform": u["rig"]}
        for name in ("P", "P*", "A", "P+T"):
            Wm, _ = K.whitening(covs[name])
            W[name] = K.rig_fit_cv(obs, P, u["free"], Wm)
        wt = {}
        xw = {}
        for name, rw in W.items():
            x = np.full_like(P, np.nan)
            for f in (0, 1):
                x[fold == f] = A.decompose(oo, rw["res"], rw["Jb"], fold != f, NB)["rem"][fold == f]
            xw[name] = x
            wt[name] = dict(fit=rms(rw["res"]), cv=rms(rw["cv"]), x=rms(x))
        for name in W:
            if name != "P":
                wt[name]["ci_cv_vs_P"] = boot_ci(obs["view"], W[name]["cv"], W["P"]["cv"], a.boot, rng)
                wt[name]["ci_x_vs_P"] = boot_ci(obs["view"], xw[name], xw["P"], a.boot, rng)
        res["weighted"] = wt
        # 5. beta against the decomposition components.
        m = valid & np.isfinite(d["rem"][:, 0])
        b = beta[m]
        comp = {k: d[k][m] for k in ("board", "pv", "smooth", "rem", "local", "x")}
        res["beta_vs"] = {k: dict(ve_comp_by_beta=ve(v, b), ve_beta_by_comp=ve(b, v)) for k, v in comp.items() if np.isfinite(v).all()}
        # beta's own structure: per-view cubic share, per-id repeatability
        sub = dict(cam=obs["cam"][m], view=obs["view"][m])
        pvb = A.per_view_smooth(sub, P[m], b)
        res["beta_pv_share"] = float((pvb["fit"] ** 2).sum() / ((b - b.mean(0)) ** 2).sum())
        res["beta_pv_share_white"] = float(pvb["dof_frac"])
        ids = obs["id"][m]
        mu = np.zeros((ids.max() + 1, 2))
        np.add.at(mu, ids, b)
        cnt = np.bincount(ids, minlength=ids.max() + 1)[:, None]
        mu = mu / np.maximum(cnt, 1)
        res["beta_id_share"] = float(((mu[ids] - b.mean(0)) ** 2).sum() / ((b - b.mean(0)) ** 2).sum())
        # slope of remainder on beta (1 if beta were the bias, 0 if unrelated)
        for k in ("rem", "local", "x"):
            e = d[k][m]
            mm = np.isfinite(e[:, 0])
            res.setdefault("slope", {})[k] = float((b[mm] * e[mm]).sum() / (b[mm] ** 2).sum())
        out["sets"][f"{sname}|{loc}"] = res
        print(f"done {sname} {loc}", flush=True)
    (RV / "out" / "gate_uq27.json").write_text(json.dumps(out, indent=1))
    tables(out)


def tables(out):
    print(f"\nfloors: {out['floors']} (per set all/chess: {out['floors_per_set']})")
    print("\n| set | localizer | cov | median sigma | s0^2 rem / x-fold / local | Spearman rem / x-fold / local (local, valid fits) | NLL local |")
    print("|---|---|---|---|---|---|---|")
    for k, r in out["sets"].items():
        s, loc = k.split("|")
        for name, c in r["cov"].items():
            print(f"| {s} | {loc} | {name} | {c['sigma_med']:.4f} | {c['rem']['s0']:.2f} / {c['x']['s0']:.2f} / {c['local']['s0']:.2f} | "
                  f"{c['rem']['sp']:.3f} / {c['x']['sp']:.3f} / {c['local']['sp']:.3f} ({c['local_sp_valid']:.3f}) | {c['local']['nll']:.3f} |")
    print("\n| set | localizer | weighting | fit | CV | x-fold rem | CV - P [95% CI] | x-fold - P [95% CI] |")
    print("|---|---|---|---|---|---|---|---|")
    for k, r in out["sets"].items():
        s, loc = k.split("|")
        for name, w in r["weighted"].items():
            ci = lambda key: f"{w[key][0]:+.4f}, {w[key][1]:+.4f}" if key in w else "-"  # noqa: E731
            print(f"| {s} | {loc} | {name} | {w['fit']:.4f} | {w['cv']:.4f} | {w['x']:.4f} | [{ci('ci_cv_vs_P')}] | [{ci('ci_x_vs_P')}] |")
    print("\n| set | localizer | valid | median abs beta | median sqrt(tr T/2) | T>0 | beta per-view share | beta per-id share | var. explained by beta: board / per-view / field / rem / local / x-fold | beta explained by each | slope rem / local / x-fold on beta |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in out["sets"].items():
        s, loc = k.split("|")
        bv = r["beta_vs"]
        ks = ("board", "pv", "smooth", "rem", "local", "x")
        print(f"| {s} | {loc} | {r['asym_valid']:.3f} | {r['beta_med']:.4f} | {r['T_sigma_med']:.4f} | {r['T_pos_frac']:.2f} | {r['beta_pv_share']:.3f} | {r['beta_id_share']:.3f} | "
              + " / ".join(f"{bv[x]['ve_comp_by_beta']:.3f}" for x in ks if x in bv) + " | "
              + " / ".join(f"{bv[x]['ve_beta_by_comp']:.3f}" for x in ks if x in bv) + " | "
              + " / ".join(f"{r['slope'][x]:+.3f}" for x in ("rem", "local", "x")) + " |")


if __name__ == "__main__":
    main()
