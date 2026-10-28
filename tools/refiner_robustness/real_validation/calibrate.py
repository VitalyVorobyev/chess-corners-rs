"""Stage 2: calibrations per localizer.

Usage: python calibrate.py <dataset_root> [--set all|chess] [--cov LOC=file.npz ...]
                           [--locs a,b] [--no-stage]

Pose models, all with the OpenCV k1 k2 p1 p2 k3 camera per camera:

- ``free``  : ``cv2.calibrateCamera`` per camera, a free pose per view (the
              robustness audit's setting). CV: train on one fold, PnP per
              held-out view and camera.
- ``rig``   : both cameras jointly, one board pose per view (cam1 frame) and
              one shared cam1 -> cam2 transform. CV: train on one fold, then
              one 6-dof pose per held-out view fitted to both cameras.
- ``stage`` : ``rig`` with the per-view pose replaced by a rigid two-axis
              stage, ``T_c1_b(v) = T0 . Tz(z d) . Rot_A(a) . Rot_B(b)``, each
              Rot about an unknown line (4 parameters), ``d`` the unknown Z
              direction and scale, ``a, b, z`` from the filename (17 pose
              parameters instead of 6 per view). A (Rx) is the outer axis: the
              composed rotations show it, and B-outer fits ~4x worse. Reported as a
              model-fit check; ``cv_pred`` predicts held-out views with no
              per-view freedom.

``stage_check`` (SaddlePoint) fits stage variants to the rig poses in pose
space (pixel metric: 25 board points through cam1) and prints the per-view
discrepancy grouped by the Rx level.

Weighting: a localizer whose detections carry a covariance (the bindings'
``position_covariance``, or ``--cov``) also gets the rig fit and its CV
whitened by that covariance (``rig_w``; residuals reported unweighted).
``--cov LOC=file.npz`` supplies the covariance for LOC: arrays ``cam`` (str,
'cam1'/'cam2'), ``token`` (str), ``id`` (int), ``cov`` (N, 2, 2) px^2.

Stereo consistency: per view, independent PnP on each camera with that
localizer's rig-fit intrinsics; ``T_c2_c1(v)`` stored for ``analyze.py``.

Writes ``out/calib_<set>.pkl``.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import common as C

LEVELS = C.LEVELS

# ---------------------------------------------------------------------------
# Observation table
# ---------------------------------------------------------------------------


def load_obs(set_name: str, cov_files: dict):
    recs = pickle.load(open(C.OUT / "detections.pkl", "rb"))
    for loc, path in cov_files.items():
        z = np.load(path, allow_pickle=True)
        key = {(str(c), str(t), int(i)): k for k, (c, t, i) in enumerate(zip(z["cam"], z["token"], z["id"]))}
        n = 0
        for r in recs:
            S = np.full((len(r["ids"]), 2, 2), np.nan)
            for j, i in enumerate(r["ids"]):
                k = key.get((r["cam"], r["token"], int(i)))
                if k is not None:
                    S[j] = z["cov"][k]
                    n += 1
            S[~np.isfinite(r["xy"][loc][:, 0])] = np.nan
            r["cov"][loc] = S
        print(f"--cov {loc}: {n} covariances from {path}")
    locs = C.LOCALIZERS if set_name == "all" else [n for n in C.LOCALIZERS if not n.startswith("Radon")]
    tokens = sorted({r["token"] for r in recs})
    vix = {t: i for i, t in enumerate(tokens)}
    stage = np.zeros((len(tokens), 5))
    Xb = C.board_points()
    cols = dict(cam=[], view=[], id=[], X=[], ref=[])
    xy = {n: [] for n in locs}
    cov = {n: [] for n in locs}
    for r in recs:
        stage[vix[r["token"]]] = r["stage"]
        keep = np.all([np.isfinite(r["xy"][n][:, 0]) for n in locs], axis=0)
        m = int(keep.sum())
        cols["cam"].append(np.full(m, C.CAMS.index(r["cam"])))
        cols["view"].append(np.full(m, vix[r["token"]]))
        cols["id"].append(r["ids"][keep])
        cols["X"].append(Xb[r["ids"][keep]])
        cols["ref"].append(r["ref"][keep])
        for n in locs:
            xy[n].append(r["xy"][n][keep])
            cov[n].append(r["cov"][n][keep])
    obs = {k: np.concatenate(v) for k, v in cols.items()}
    obs["xy"] = {n: np.concatenate(xy[n]) for n in locs}
    obs["cov"] = {n: np.concatenate(cov[n]) for n in locs}
    obs["tokens"] = tokens
    obs["stage"] = stage
    obs["fold"] = C.folds(stage)
    obs["locs"] = locs
    obs["shape"] = recs[0]["shape"]
    return obs


def rms(r):
    r = r[np.isfinite(r[:, 0])]
    return float(np.sqrt((r**2).sum(1).mean()))


# ---------------------------------------------------------------------------
# Free per-camera calibration (OpenCV)
# ---------------------------------------------------------------------------

CRIT = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)


def cv_calib(objs, imgs, size, K0):
    _, K, D, rv, tv = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objs],
        [p.astype(np.float32) for p in imgs],
        size,
        K0.copy(),
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=CRIT,
    )
    return K, D, rv, tv


def pnp(obj, img, K, D):
    _, r, t = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    r, t = cv2.solvePnPRefineLM(obj, img, K, D, r, t)
    return r, t


def reproj(obj, r, t, K, D):
    p, _ = cv2.projectPoints(obj.astype(np.float64), r, t, K, D)
    return p.reshape(-1, 2)


def free_fit(obs, P, K0s):
    h, w = obs["shape"]
    res = np.full_like(P, np.nan)
    rcv = np.full_like(P, np.nan)
    intr, poses = [], {}
    for c in (0, 1):
        vs = sorted(set(obs["view"][obs["cam"] == c]))
        sel = [np.where((obs["cam"] == c) & (obs["view"] == v))[0] for v in vs]
        K, D, rv, tv = cv_calib([obs["X"][s] for s in sel], [P[s] for s in sel], (w, h), K0s[c])
        intr.append(C.intr_from_cv(K, D))
        for v, s, r, t in zip(vs, sel, rv, tv):
            res[s] = P[s] - reproj(obs["X"][s], r, t, K, D)
            poses[(c, v)] = C.rt_to_T(r, t)
        for fold in (0, 1):
            tr = [k for k, v in enumerate(vs) if obs["fold"][v] != fold]
            te = [k for k, v in enumerate(vs) if obs["fold"][v] == fold]
            Kt, Dt, _, _ = cv_calib([obs["X"][sel[k]] for k in tr], [P[sel[k]] for k in tr], (w, h), K0s[c])
            for k in te:
                s = sel[k]
                r, t = pnp(obs["X"][s], P[s], Kt, Dt)
                rcv[s] = P[s] - reproj(obs["X"][s], r, t, Kt, Dt)
    return dict(res=res, cv=rcv, intr=intr, poses=poses)


# ---------------------------------------------------------------------------
# Joint models (scipy). Parameters: intr cam1 (9) | intr cam2 (9) | rig (6) |
# pose block (rig: 6 per view in ``views``; stage: 17).
# ---------------------------------------------------------------------------


def _axis(par, k, sgn):
    v = np.empty(3)
    o = [i for i in range(3) if i != k]
    v[k] = sgn
    v[o] = par[:2]
    q = np.zeros(3)
    q[o] = par[2:4]
    return v / np.linalg.norm(v), q


def stage_motion(pb, meta, stage):
    """Board-at-zero <- board motion per stage row; ``pb[6:17]`` = axis A (4),
    axis B (4), Z direction (3). A is the outer axis."""
    kA, sA, kB, sB = meta
    omA, qA = _axis(pb[6:10], kA, sA)
    omB, qB = _axis(pb[10:14], kB, sB)
    out = np.empty((len(stage), 4, 4))
    for i, (a, b, _x, _y, z) in enumerate(stage):
        Tz = np.eye(4)
        Tz[:3, 3] = z * pb[14:17]
        out[i] = Tz @ C.axis_rotation(omA, qA, np.deg2rad(a)) @ C.axis_rotation(omB, qB, np.deg2rad(b))
    return out


def view_poses(p, mode, views, stage, meta):
    """cam1 <- board for each of the (unique) ``views``; in ``rig`` mode
    ``views`` must be the model's own view list (one 6-dof block each)."""
    pb = p[24:]
    if mode == "rig":
        return np.stack([C.rt_to_T(pb[6 * k : 6 * k + 3], pb[6 * k + 3 : 6 * k + 6]) for k in range(len(views))])
    T0 = C.rt_to_T(pb[:3], pb[3:6])
    return np.einsum("ij,vjk->vik", T0, stage_motion(pb, meta, stage[np.asarray(views)]))


def project_obs(p, Tv, cam, X):
    """Tv (N, 4, 4) cam1 <- board per observation."""
    rig = C.rt_to_T(p[18:21], p[21:24])
    T = np.where((cam == 1)[:, None, None], np.einsum("ij,njk->nik", rig, Tv), Tv)
    Xc = np.einsum("nij,nj->ni", T[:, :3, :3], X) + T[:, :3, 3]
    out = np.empty((len(X), 2))
    for c in (0, 1):
        m = cam == c
        out[m] = C.project(Xc[m], p[9 * c : 9 * c + 9])
    return out, T


class Joint:
    """Least squares over observations ``idx``; grouped central-difference
    Jacobian (each observation depends on one view block)."""

    def __init__(self, obs, P, idx, mode, W=None, meta=None):
        self.cam, self.view, self.X, self.P = obs["cam"][idx], obs["view"][idx], obs["X"][idx], P[idx]
        self.W = None if W is None else W[idx]
        self.stage, self.mode, self.meta = obs["stage"], mode, meta
        self.views = sorted(set(self.view))
        self.vk = np.searchsorted(self.views, self.view)

    def predict(self, p, view=None, cam=None, X=None):
        """Predicted pixels for this model's observations, or for other
        observations (stage mode only)."""
        if view is None:
            Tv = view_poses(p, self.mode, self.views, self.stage, self.meta)[self.vk]
            return project_obs(p, Tv, self.cam, self.X)
        u, k = np.unique(view, return_inverse=True)
        Tv = view_poses(p, self.mode, u, self.stage, self.meta)[k]
        return project_obs(p, Tv, cam, X)

    def resid(self, p):
        r = self.P - self.predict(p)[0]
        if self.W is not None:
            r = np.einsum("nij,nj->ni", self.W, r)
        return r.ravel()

    def jac(self, p):
        n = len(p)
        J = np.zeros((2 * len(self.cam), n))
        h = 1e-6 * np.maximum(np.abs(p), 1e-3)
        glob = range(n) if self.mode != "rig" else range(24)
        for j in glob:
            e = np.zeros(n)
            e[j] = h[j]
            J[:, j] = (self.resid(p + e) - self.resid(p - e)) / (2 * h[j])
        if self.mode == "rig":
            rows = np.repeat(self.vk, 2)
            ar = np.arange(2 * len(self.cam))
            for j in range(6):
                cols = 24 + 6 * np.arange(len(self.views)) + j
                e = np.zeros(n)
                e[cols] = h[cols]
                d = (self.resid(p + e) - self.resid(p - e)) / (2 * h[cols][rows])
                J[ar, cols[rows]] = d
        return J

    def fit(self, p0):
        return least_squares(self.resid, p0, jac=self.jac, method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=100).x


def rig_init(free, views):
    intr = np.concatenate(free["intr"])
    rels, pose = [], []
    for v in views:
        T1 = free["poses"].get((0, v))
        if T1 is None:
            T1 = C.inv(rels[0]) @ free["poses"][(1, v)] if rels else free["poses"][(1, v)]
        pose.append(np.r_[C.T_to_rt(T1)])
        if (0, v) in free["poses"] and (1, v) in free["poses"]:
            rels.append(free["poses"][(1, v)] @ C.inv(free["poses"][(0, v)]))
    R = Rotation.from_matrix(np.stack([x[:3, :3] for x in rels])).mean().as_rotvec()
    t = np.median(np.stack([x[:3, 3] for x in rels]), 0)
    return np.r_[intr, R, t, np.concatenate(pose)]


def fit_view_pose(p, cam, X, P, W, T_init):
    """One cam1 <- board pose for a held-out view, both cameras, intrinsics and
    rig fixed."""

    def f(x):
        Tv = np.repeat(C.rt_to_T(x[:3], x[3:])[None], len(X), 0)
        r = P - project_obs(p, Tv, cam, X)[0]
        if W is not None:
            r = np.einsum("nij,nj->ni", W, r)
        return r.ravel()

    x = least_squares(f, np.r_[C.T_to_rt(T_init)], x_scale="jac", xtol=1e-12, ftol=1e-12).x
    return C.rt_to_T(x[:3], x[3:])


def rig_fit_cv(obs, P, free, W=None):
    n = len(P)
    allv = sorted(set(obs["view"]))
    p0 = rig_init(free, allv)
    J = Joint(obs, P, np.arange(n), "rig", W)
    p = J.fit(p0)
    fitted, T = J.predict(p)
    fitted = P - fitted
    Jb = np.zeros((n, 2, 3))
    for c in (0, 1):
        m = obs["cam"] == c
        Jb[m] = C.board_jacobian(T[m, :3, :3], T[m, :3, 3], obs["X"][m], p[9 * c : 9 * c + 9])
    cv = np.full_like(P, np.nan)
    fold_p = []
    for fold in (0, 1):
        tr = np.where(obs["fold"][obs["view"]] != fold)[0]
        te_views = [v for v in allv if obs["fold"][v] == fold]
        trv = sorted(set(obs["view"][tr]))
        pt0 = np.r_[p[:24], np.concatenate([p[24 + 6 * allv.index(v) : 30 + 6 * allv.index(v)] for v in trv])]
        pt = Joint(obs, P, tr, "rig", W).fit(pt0)
        fold_p.append(pt[:24])
        for v in te_views:
            s = np.where(obs["view"] == v)[0]
            T0 = C.rt_to_T(p[24 + 6 * allv.index(v) : 27 + 6 * allv.index(v)], p[27 + 6 * allv.index(v) : 30 + 6 * allv.index(v)])
            Tv = fit_view_pose(pt, obs["cam"][s], obs["X"][s], P[s], None if W is None else W[s], T0)
            cv[s] = P[s] - project_obs(pt, np.repeat(Tv[None], len(s), 0), obs["cam"][s], obs["X"][s])[0]
    return dict(p=p, res=fitted, cv=cv, Jb=Jb, fold_p=fold_p)


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------


def stage_init(p_rig, stage):
    V = len(stage)
    T = [C.rt_to_T(p_rig[24 + 6 * v : 27 + 6 * v], p_rig[27 + 6 * v : 30 + 6 * v]) for v in range(V)]
    T0 = T[[v for v in range(V) if not stage[v].any()][0]]
    M = [C.inv(T0) @ T[v] for v in range(V)]
    meta, par = [], []
    for ax in (0, 1):
        vs = [v for v in range(V) if stage[v, ax] != 0 and stage[v, 1 - ax] == 0 and stage[v, 4] == 0]
        om = np.mean([Rotation.from_matrix(M[v][:3, :3]).as_rotvec() / np.deg2rad(stage[v, ax]) for v in vs], 0)
        om /= np.linalg.norm(om)
        A = np.concatenate([np.eye(3) - M[v][:3, :3] for v in vs])
        q = np.linalg.lstsq(A, np.concatenate([M[v][:3, 3] for v in vs]), rcond=None)[0]
        k = int(np.argmax(np.abs(om)))
        q = q - q[k] / om[k] * om
        o = [i for i in range(3) if i != k]
        meta += [k, float(np.sign(om[k]))]
        par += list(om[o] / abs(om[k])) + list(q[o])
    d = np.mean([M[v][:3, 3] / stage[v, 4] for v in range(V) if stage[v, 4] != 0], 0)
    return np.r_[p_rig[:24], C.T_to_rt(T0)[0], C.T_to_rt(T0)[1], par, d], tuple(meta)


def stage_fit(obs, P, p_rig):
    pinit, meta = stage_init(p_rig, obs["stage"])
    J = Joint(obs, P, np.arange(len(P)), "stage", None, meta)
    p = J.fit(pinit)
    res = P - J.predict(p)[0]
    cvp = np.full_like(P, np.nan)
    for fold in (0, 1):
        tr = np.where(obs["fold"][obs["view"]] != fold)[0]
        te = np.where(obs["fold"][obs["view"]] == fold)[0]
        Jt = Joint(obs, P, tr, "stage", None, meta)
        pt = Jt.fit(p)
        cvp[te] = P[te] - Jt.predict(pt, obs["view"][te], obs["cam"][te], obs["X"][te])[0]
    return dict(p=p, meta=meta, res=res, cv_pred=cvp)


def stage_check(obs, p_rig):
    """Pose-space fit of stage variants to the rig poses. Returns a dict of
    model -> (n_pose_params, rms px) and per-Rx-level mean discrepancy."""
    st = obs["stage"]
    V = len(st)
    intr = p_rig[:9]
    Tr = [C.rt_to_T(p_rig[24 + 6 * v : 27 + 6 * v], p_rig[27 + 6 * v : 30 + 6 * v]) for v in range(V)]
    g = np.linspace(C.SQUARE, (C.NSQ - 1) * C.SQUARE, 5)
    G = np.array([[x, y, 0] for x in g for y in g])
    target = np.stack([C.project((T[:3, :3] @ G.T).T + T[:3, 3], intr) for T in Tr])
    pinit, meta = stage_init(p_rig, st)
    ia = np.searchsorted(LEVELS, st[:, 0])
    ib = np.searchsorted(LEVELS, st[:, 1])

    def motions(pb, model):
        kA, sA, kB, sB = meta
        omA, qA = _axis(pb[6:10], kA, sA)
        omB, qB = _axis(pb[10:14], kB, sB)
        out = []
        for v, (a, b, _, _, z) in enumerate(st):
            sa = pb[17] if model == "scale" else 1.0
            sb = pb[18] if model == "scale" else 1.0
            RA = C.axis_rotation(omA, qA, np.deg2rad(a) * sa)
            RB = C.axis_rotation(omB, qB, np.deg2rad(b) * sb)
            Tz = np.eye(4)
            Tz[:3, 3] = z * pb[14:17]
            if model == "rx_level_free":  # per-Rx-level free 6-dof offset
                k = 19 + 6 * ia[v]
                RA = C.rt_to_T(pb[k : k + 3] * 1e-3, pb[k + 3 : k + 6] * 1e-3) @ RA
            if model == "ry_level_free":
                k = 19 + 6 * ib[v]
                RB = RB @ C.rt_to_T(pb[k : k + 3] * 1e-3, pb[k + 3 : k + 6] * 1e-3)
            out.append(Tz @ RA @ RB)
        return out

    def resid(pb, model):
        T0 = C.rt_to_T(pb[:3], pb[3:6])
        Ms = motions(pb, model)
        return np.concatenate([C.project(((T0 @ M)[:3, :3] @ G.T).T + (T0 @ M)[:3, 3], intr) - target[v] for v, M in enumerate(Ms)]).ravel()

    pb0 = np.r_[pinit[24:], 1.0, 1.0, np.zeros(30)]
    npar = {"rigid": 17, "scale": 19, "rx_level_free": 17 + 24, "ry_level_free": 17 + 24}
    out = {}
    for model in npar:
        r = least_squares(resid, pb0, args=(model,), x_scale="jac", max_nfev=500)
        rr = r.fun.reshape(V, -1, 2)
        out[model] = (npar[model], float(np.sqrt((rr**2).sum(2).mean())))
        if model == "rigid":
            pb = r.x
            T0 = C.rt_to_T(pb[:3], pb[3:6])
            Ms = motions(pb, "rigid")
            Cc = np.eye(4)
            Cc[:3, 3] = [C.NSQ * C.SQUARE / 2, C.NSQ * C.SQUARE / 2, 0]
            disc = []
            for v in range(V):
                D = C.inv(Cc) @ C.inv(Tr[v]) @ (T0 @ Ms[v]) @ Cc
                disc.append(np.r_[Rotation.from_matrix(D[:3, :3]).as_rotvec() * 1e3, D[:3, 3] * 1e3])
            out["discrepancy"] = np.array(disc)  # mrad (3), um (3), board-centre frame
    out["rx_level"] = ia
    out["is_z"] = st[:, 4] != 0
    return out


def whitening(cov):
    """Per-observation W with W'W proportional to Sigma^-1 (identity where Sigma
    is missing), normalised to unit mean scale."""
    W = np.tile(np.eye(2), (len(cov), 1, 1))
    ok = np.isfinite(cov).all(axis=(1, 2))
    W[ok] = np.linalg.inv(np.linalg.cholesky(cov[ok]))
    s = np.sqrt(np.mean(np.einsum("nij,nij->n", W[ok], W[ok]) / 2))
    W[ok] /= s
    return W, ok


def stereo_poses(obs, P, p):
    """Per view: cam2 <- cam1 from independent PnP on each camera."""
    out = {}
    Ks = [C.intr_to_cv(p[9 * c : 9 * c + 9]) for c in (0, 1)]
    for v in sorted(set(obs["view"])):
        T = []
        for c in (0, 1):
            s = np.where((obs["view"] == v) & (obs["cam"] == c))[0]
            if len(s) < 20:
                break
            r, t = pnp(obs["X"][s], P[s], *Ks[c])
            T.append(C.rt_to_T(r, t))
        if len(T) == 2:
            out[v] = T[1] @ C.inv(T[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--set", default="all", choices=["all", "chess"])
    ap.add_argument("--cov", action="append", default=[], help="LOC=file.npz")
    ap.add_argument("--locs", default=None, help="comma-separated subset")
    ap.add_argument("--no-stage", action="store_true")
    a = ap.parse_args()
    obs = load_obs(a.set, dict(s.split("=", 1) for s in a.cov))
    prior = json.load(open(Path(a.root) / "calibration.json"))["intrinsic"]
    K0s = [np.array(prior[f"camera{c}"]["matrix"], float) for c in (0, 1)]
    locs = obs["locs"] if a.locs is None else a.locs.split(",")
    print(f"set={a.set}: {len(obs['cam'])} observations, {len(obs['tokens'])} views, localizers {locs}")
    out = dict(obs={k: v for k, v in obs.items() if k not in ("xy",)}, locs={})
    out["obs"]["xy"] = {n: obs["xy"][n] for n in locs}
    for loc in locs:
        P = obs["xy"][loc]
        free = free_fit(obs, P, K0s)
        rig = rig_fit_cv(obs, P, free)
        res = dict(free=free, rig=rig)
        res["stereo"] = stereo_poses(obs, P, rig["p"])
        if not a.no_stage:
            res["stage"] = stage_fit(obs, P, rig["p"])
            if loc == "SaddlePoint":
                out["stage_check"] = stage_check(obs, rig["p"])
        covs = obs["cov"][loc]
        if np.isfinite(covs).all(axis=(1, 2)).mean() > 0.5:
            W, ok = whitening(covs)
            res["rig_w"] = rig_fit_cv(obs, P, free, W)
            res["cov_frac"] = float(ok.mean())
        out["locs"][loc] = res
        line = f"{loc:12s} free {rms(free['res']):.4f}/{rms(free['cv']):.4f}  rig {rms(rig['res']):.4f}/{rms(rig['cv']):.4f}"
        if "stage" in res:
            line += f"  stage {rms(res['stage']['res']):.4f}/pred {rms(res['stage']['cv_pred']):.4f}"
        if "rig_w" in res:
            line += f"  rig weighted {rms(res['rig_w']['res']):.4f}/{rms(res['rig_w']['cv']):.4f}"
        print(line, flush=True)
    with open(C.OUT / f"calib_{a.set}.pkl", "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
