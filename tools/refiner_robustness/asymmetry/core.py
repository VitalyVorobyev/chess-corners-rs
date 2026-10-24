"""Per-corner asymmetry term (UQ-26 prototype), vectorised over corners.

Everything is expressed as a linear map from a fixed pixel *block* around each
corner, so the noise covariance of every estimate is ``sigma^2 L L^T``.

- ``saddle``: the shipped re-centred SaddlePoint fit (quadratic on a
  ``(2r+1)^2`` grid resampled bilinearly at the estimate, iterated until the
  fitted saddle is on the window centre) for any radius, with its
  implicit-function Jacobian ``L_c = -M^-1 J R W`` (book section 8.3).
- ``asymmetry``: at the estimate ``c``, ``P(d) = I(c + d)`` on an integer grid,
  ``S = (P(d) + P(-d)) / 2``, ``A = (P(d) - P(-d)) / 2``. ``A`` on a half-plane
  of the ``(2Ra+1)^2`` grid is fitted by least squares with
  ``b . grad S`` (a mislocation: ``P(d) = S0(d + b)`` gives ``A ~ b . grad S0``,
  so ``b`` estimates ``c - c*``) and ``(g . d)(S - m)`` (a multiplicative
  illumination ramp), optionally ``g' . d`` (additive ramp). The design matrix
  is built from the data and treated as fixed for the noise propagation (its
  noise enters multiplied by the small coefficients, second order).
  ``L_b = X^+ Q W`` is the direct pixel map; ``b`` also moves with ``c``
  (gain ``G = db/dc``, close to ``I``), so ``Var(b) = (G L_c + L_b)(...)^T``
  and the corrected position ``c - b`` has map ``(I - G) L_c - L_b``.

Coordinates: pixel ``i`` is centred at coordinate ``i`` (as the detector).
"""
from __future__ import annotations

import numpy as np

KB = 6  # block half-size: block pixels floor(c) + e, e in [-KB, KB + 1]^2
NB = 2 * KB + 2
FLOOR = 0.006  # 1.5.0 position floor, px per axis
MIN_VAR = 1.0 / 12.0


# ---------------------------------------------------------------------------
# Sampling as a linear operator on the block
# ---------------------------------------------------------------------------

_SEL: dict = {}


def _sel(k: int) -> np.ndarray:
    """(4, n_k, NB^2) 0/1 selectors: sample d in [-k, k]^2 (row-major, dy
    outer) of the bilinear corner (i, j) -> block pixel KB + d + (i, j)."""
    if k not in _SEL:
        assert k + 1 <= KB
        side = 2 * k + 1
        s = np.zeros((4, side * side, NB * NB))
        for q, (i, j) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            n = 0
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    s[q, n, (KB + dy + j) * NB + (KB + dx + i)] = 1.0
                    n += 1
        _SEL[k] = s
    return _SEL[k]


def block(img: np.ndarray, c: np.ndarray):
    """Block pixels (N, NB^2) and bilinear fractions t (N, 2) for positions
    c (N, 2). Positions whose block leaves the image are clamped (callers
    keep corners >= KB + 2 px from the border)."""
    f = np.floor(c).astype(int)
    t = c - f
    ey, ex = np.meshgrid(np.arange(-KB, KB + 2), np.arange(-KB, KB + 2), indexing="ij")
    ys = np.clip(f[:, 1, None, None] + ey[None], 0, img.shape[0] - 1)
    xs = np.clip(f[:, 0, None, None] + ex[None], 0, img.shape[1] - 1)
    return img[ys, xs].reshape(len(c), -1).astype(np.float64), t


def wmat(t: np.ndarray, k: int) -> np.ndarray:
    """(N, n_k, NB^2) bilinear sampling operator for fractions t."""
    tx, ty = t[:, 0], t[:, 1]
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1)
    return np.einsum("nq,qkp->nkp", w, _sel(k))


def dwmat(t: np.ndarray, k: int):
    """d wmat / d tx, d wmat / d ty."""
    tx, ty = t[:, 0], t[:, 1]
    wx = np.stack([-(1 - ty), (1 - ty), -ty, ty], 1)
    wy = np.stack([-(1 - tx), -tx, (1 - tx), tx], 1)
    s = _sel(k)
    return np.einsum("nq,qkp->nkp", wx, s), np.einsum("nq,qkp->nkp", wy, s)


def sample(img: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Bilinear patch (N, (2k+1)^2) at c + d."""
    b, t = block(img, c)
    return np.einsum("nkp,np->nk", wmat(t, k), b)


def inv(A: np.ndarray) -> np.ndarray:
    """Batched inverse; NaN for singular / non-finite members."""
    out = np.full_like(A, np.nan)
    fin = np.isfinite(A).all(axis=(1, 2))
    if fin.any():
        c = np.linalg.cond(A[fin])
        g = np.where(fin)[0][c < 1e12]
        if len(g):
            out[g] = np.linalg.inv(A[g])
    return out


# ---------------------------------------------------------------------------
# SaddlePoint
# ---------------------------------------------------------------------------

_FIT: dict = {}


def fit_op(r: int) -> np.ndarray:
    """Rows a..e of (Phi^T Phi)^-1 Phi^T for [x^2, xy, y^2, x, y, 1]."""
    if r not in _FIT:
        dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        x, y = dx.ravel().astype(float), dy.ravel().astype(float)
        phi = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
        _FIT[r] = np.linalg.solve(phi.T @ phi, phi.T)[:5]
    return _FIT[r]


def _solve(th):
    a, b, c, d, e = th.T
    det = 4 * a * c - b * b
    ux = -(2 * c * d - b * e) / det
    uy = (b * d - 2 * a * e) / det
    return np.stack([ux, uy], 1), det


def _jac(th, u, det):
    a, b, c = th[:, 0], th[:, 1], th[:, 2]
    nh = np.stack([np.stack([-2 * c, b], 1), np.stack([b, -2 * a], 1)], 1) / det[:, None, None]
    z = np.zeros_like(a)
    du = np.stack(
        [np.stack([2 * u[:, 0], u[:, 1], z, z + 1, z], 1), np.stack([z, u[:, 0], 2 * u[:, 1], z, z + 1], 1)], 1
    )
    return nh @ du  # (N, 2, 5)


def saddle(img: np.ndarray, c0: np.ndarray, r: int = 2, iters: int = 30, tol: float = 1e-5):
    """Re-centred SaddlePoint from c0. Returns (c, ok, L) with L (N, 2, NB^2)
    the noise map of the converged position."""
    R = fit_op(r)
    c = c0.astype(np.float64).copy()
    n = len(c)
    ok = np.ones(n, bool)

    def evalu(cc):
        b, t = block(img, cc)
        W = wmat(t, r)
        th = np.einsum("ik,nkp,np->ni", R, W, b)
        u, det = _solve(th)
        return u, det, th, b, t, W

    u, det, th, b, t, W = evalu(c)
    ok &= det < -1e-3
    for _ in range(iters):
        nu = np.linalg.norm(u, axis=1)
        act = ok & (nu > tol)
        if not act.any():
            break
        a = np.where(act)[0]
        Wx, Wy = dwmat(t[a], r)
        dth = np.stack([np.einsum("ik,nkp,np->ni", R, Wx, b[a]), np.einsum("ik,nkp,np->ni", R, Wy, b[a])], 2)
        J = _jac(th[a], u[a], det[a])
        M = J @ dth  # du/dc
        dm = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        step = u[a].copy()  # plain step if M is degenerate
        g = np.abs(dm) > 1e-3
        step[g] = -np.einsum("nij,nj->ni", inv(M[g]), u[a][g])
        step[~np.isfinite(step).all(1)] = u[a][~np.isfinite(step).all(1)]
        lam = np.ones(len(a))
        done = np.zeros(len(a), bool)
        for _h in range(6):
            tr = ~done
            if not tr.any():
                break
            cn = c[a[tr]] + lam[tr, None] * step[tr]
            un, dn, thn, bn, tn, Wn = evalu(cn)
            better = (dn < -1e-3) & (np.linalg.norm(un, axis=1) < nu[a[tr]])
            idx = a[tr][better]
            c[idx], u[idx], det[idx], th[idx], b[idx], t[idx] = cn[better], un[better], dn[better], thn[better], bn[better], tn[better]
            done[np.where(tr)[0][better]] = True
            lam[tr] *= 0.5
        ok[a[~done]] &= np.linalg.norm(u[a[~done]], axis=1) < 1e-3
        # a corner that cannot improve but is already within 1e-3 stays ok
        if (~done).all():
            break
    ok &= np.linalg.norm(u, axis=1) < 1e-3
    c = c + u
    # Noise map at the fixed point.
    u, det, th, b, t, W = evalu(c)
    Wx, Wy = dwmat(t, r)
    dth = np.stack([np.einsum("ik,nkp,np->ni", R, Wx, b), np.einsum("ik,nkp,np->ni", R, Wy, b)], 2)
    J = _jac(th, u, det)
    M = J @ dth
    L = -inv(M) @ (J @ np.einsum("ik,nkp->nip", R, W))
    ok &= np.isfinite(L).all(axis=(1, 2)) & (det < -1e-3)
    return c, ok, L


# ---------------------------------------------------------------------------
# Asymmetry fit
# ---------------------------------------------------------------------------

_Q: dict = {}


def _half(ra: int):
    """Indices of the inner grid [-ra, ra]^2 in the half plane (dy > 0 or
    dy == 0, dx > 0) and of their mirrors, plus offsets."""
    if ra not in _Q:
        side = 2 * ra + 1
        dy, dx = np.meshgrid(np.arange(-ra, ra + 1), np.arange(-ra, ra + 1), indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()
        h = (dy > 0) | ((dy == 0) & (dx > 0))
        idx = np.where(h)[0]
        mir = (side * side - 1) - idx
        _Q[ra] = (idx, mir, dx[idx].astype(float), dy[idx].astype(float))
    return _Q[ra]


def asym_fit(img: np.ndarray, c: np.ndarray, ra: int = 2, additive: bool = False, weight: str | None = None):
    """Fit A at c. Returns (b (N,2), coefs (N,p), Lb (N,2,NB^2), cond)."""
    k = ra + 1
    blk, t = block(img, c)
    W = wmat(t, k)  # (N, (2k+1)^2, NB^2)
    P = np.einsum("nkp,np->nk", W, blk)
    side = 2 * k + 1
    Pg = P.reshape(-1, side, side)
    S = 0.5 * (Pg + Pg[:, ::-1, ::-1])
    gx = 0.5 * (S[:, 1:-1, 2:] - S[:, 1:-1, :-2])
    gy = 0.5 * (S[:, 2:, 1:-1] - S[:, :-2, 1:-1])
    Si = S[:, 1:-1, 1:-1].reshape(len(c), -1)
    gx, gy = gx.reshape(len(c), -1), gy.reshape(len(c), -1)
    idx, mir, dx, dy = _half(ra)
    m = Si.mean(1, keepdims=True)
    # additive: False -> [grad S, (g.d)(S - m)]; True -> plus g'.d;
    # "mul" -> [grad S, (g.d) S] (a pure multiplicative ramp, 4 unknowns).
    if additive == "mulB":
        # Regressors = dA/dc of the bilinear sampler itself (so db/dc ~ I and
        # no gain correction is needed): dA/dc(d) = (dP(d) - dP(-d)) / 2.
        Wx, Wy = dwmat(t, k)
        ddx = np.einsum("nkp,np->nk", Wx, blk).reshape(-1, side, side)
        ddy = np.einsum("nkp,np->nk", Wy, blk).reshape(-1, side, side)
        ax = (0.5 * (ddx - ddx[:, ::-1, ::-1]))[:, 1:-1, 1:-1].reshape(len(c), -1)
        ay = (0.5 * (ddy - ddy[:, ::-1, ::-1]))[:, 1:-1, 1:-1].reshape(len(c), -1)
        cols = [ax[:, idx], ay[:, idx], dx * Si[:, idx], dy * Si[:, idx]]
    elif additive == "mul":
        cols = [gx[:, idx], gy[:, idx], dx * Si[:, idx], dy * Si[:, idx]]
    else:
        cols = [gx[:, idx], gy[:, idx], dx * (Si[:, idx] - m), dy * (Si[:, idx] - m)]
    if additive is True:
        cols += [np.broadcast_to(dx, gx[:, idx].shape), np.broadcast_to(dy, gx[:, idx].shape)]
    X = np.stack(cols, 2)  # (N, nh, p)
    # A on the inner grid: inner index -> outer index map.
    inner = (np.arange(1, side - 1)[:, None] * side + np.arange(1, side - 1)[None]).ravel()
    Wi = W[:, inner]  # (N, ni, NB^2)
    Qw = 0.5 * (Wi[:, idx] - Wi[:, mir])  # A rows as pixel maps
    A = np.einsum("nhp,np->nh", Qw, blk)
    if weight == "gauss":
        wv = np.exp(-0.5 * (dx * dx + dy * dy) / (0.6 * ra + 0.6) ** 2)
        Xw, Qww = X * np.sqrt(wv)[None, :, None], Qw * np.sqrt(wv)[None, :, None]
    else:
        Xw, Qww = X, Qw
    XtX = np.einsum("nhi,nhj->nij", Xw, Xw)
    pinv = inv(XtX) @ np.swapaxes(Xw, 1, 2)  # (N, p, nh)
    Lfull = pinv @ Qww
    coef = np.einsum("nph,nh->np", pinv, A if weight is None else A * np.sqrt(wv)[None])
    cond = np.full(len(c), np.nan)
    fin = np.isfinite(XtX).all(axis=(1, 2))
    if fin.any():
        ev = np.linalg.eigvalsh(XtX[fin])
        cond[fin] = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
    return coef[:, :2], coef, Lfull, cond


def asym_full(img, c, Lc, ra=2, additive=False, weight=None, h=0.01):
    """b, Var map of b (G Lc + Lb), corrected-position map ((I-G) Lc - Lb),
    gain G, coefs, cond."""
    b, coef, Lfull, cond = asym_fit(img, c, ra, additive, weight)
    Lb = Lfull[:, :2]
    bx = asym_fit(img, c + [h, 0], ra, additive, weight)[0]
    bxm = asym_fit(img, c - [h, 0], ra, additive, weight)[0]
    by = asym_fit(img, c + [0, h], ra, additive, weight)[0]
    bym = asym_fit(img, c - [0, h], ra, additive, weight)[0]
    G = np.stack([(bx - bxm) / (2 * h), (by - bym) / (2 * h)], 2)  # (N, 2, 2) db/dc
    # Newton step to the fixed point b(c) = 0: beta = G^-1 b estimates c - c*.
    Gi = inv(G)
    beta = np.einsum("nij,nj->ni", Gi, b)
    Lbeta = Lc + Gi @ Lb  # noise map of beta (includes the noise of c)
    Lcorr = -Gi @ Lb  # noise map of c - beta
    # Ramp coefficients and their noise map (their dependence on c is
    # second order: a shift does not create a ramp-shaped asymmetry).
    return dict(b=b, beta=beta, Lbeta=Lbeta, Lcorr=Lcorr, G=G, coef=coef, cond=cond, Lg=Lfull[:, 2:4])


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------


def cov_of(L: np.ndarray, var: float) -> np.ndarray:
    return var * np.einsum("nip,njp->nij", L, L)


def floor_cov(n: int) -> np.ndarray:
    return np.broadcast_to(FLOOR**2 * np.eye(2), (n, 2, 2)).copy()


def psd_part(C: np.ndarray) -> np.ndarray:
    out = np.full_like(C, np.nan)
    fin = np.isfinite(C).all(axis=(1, 2))
    if fin.any():
        Cf = C[fin]
        w, v = np.linalg.eigh(0.5 * (Cf + np.swapaxes(Cf, 1, 2)))
        w = np.clip(w, 0, None)
        out[fin] = np.einsum("nij,nj,nkj->nik", v, w, v)
    return out


def whiten_q(e: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nij,nj->n", e, inv(C), e)


def methods(img: np.ndarray, c2: np.ndarray, sigma: float, asym_cfgs=(("r2", 2, False),),
            radii=(1, 2, 3)):
    """Per-corner positions and covariances of every option.

    c2: SaddlePoint (radius 2) estimates (e.g. from the bindings); refined
    again here to tol 1e-5. ``asym_cfgs``: (label, window radius, additive
    ramp). Returns (dict of arrays, validity mask).
    """
    var = max(sigma * sigma, MIN_VAR)
    c, ok, Lc = saddle(img, c2, 2)
    n = len(c)
    out = dict(c=c)
    base = cov_of(Lc, var) + floor_cov(n)
    out["S_150"] = base
    # Control: a larger constant floor instead of a per-corner term.
    for f in (0.010, 0.015, 0.020):
        out[f"S_fl{round(f * 1000)}"] = base + f * f * np.eye(2)[None]
    valid = ok.copy()
    for lab, ra, add in asym_cfgs:
        a = asym_full(img, c, Lc, ra, add)
        b = a["beta"]
        Cb = cov_of(a["Lbeta"], var)
        bbT = np.einsum("ni,nj->nij", b, b)
        out[f"beta_{lab}"] = b
        out[f"G_{lab}"] = a["G"]
        out[f"S_a_{lab}"] = base + psd_part(bbT - Cb)
        out[f"S_araw_{lab}"] = base + bbT
        out[f"c_b_{lab}"] = c - b
        out[f"S_b_{lab}"] = cov_of(a["Lcorr"], var) + floor_cov(n)
        # Gated correction: only where the fitted ramp is significant
        # (chi^2_2 99% = 9.21).
        g = a["coef"][:, 2:4]
        zg = whiten_q(g, cov_of(a["Lg"], var))
        gate = zg > 9.21
        out[f"zg_{lab}"] = zg
        out[f"c_bg_{lab}"] = np.where(gate[:, None], c - b, c)
        out[f"S_bg_{lab}"] = np.where(gate[:, None, None], out[f"S_b_{lab}"], base)
        cc = c - b
        Gi = inv(a["G"])
        for _ in range(2):
            cc = cc - np.einsum("nij,nj->ni", Gi, asym_fit(img, cc, ra, add)[0])
        out[f"c_bit_{lab}"] = cc
        valid &= np.isfinite(b).all(1) & (np.linalg.norm(b, axis=1) < 1.0)
    # (c) multi-radius spread
    cs, Ls = [], []
    for r in radii:
        if r == 2:
            cr, okr, Lr = c, ok, Lc
        else:
            cr, okr, Lr = saddle(img, c, r)
        cs.append(cr)
        Ls.append(Lr)
        valid &= okr & (np.linalg.norm(cr - c, axis=1) < 1.0)
    cs, Ls = np.stack(cs), np.stack(Ls)
    k = len(radii)
    dc = cs - cs.mean(0)
    D = np.einsum("kni,knj->nij", dc, dc) / (k - 1)
    dL = Ls - Ls.mean(0)
    ED = var * np.einsum("knip,knjp->nij", dL, dL) / (k - 1)
    out["S_c"] = base + psd_part(D - ED)
    out["S_craw"] = base + D
    out["c_r"] = cs
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            valid &= np.isfinite(v.reshape(n, -1)).all(1)
    valid &= np.isfinite(cs).all(axis=(0, 2))
    return out, valid
