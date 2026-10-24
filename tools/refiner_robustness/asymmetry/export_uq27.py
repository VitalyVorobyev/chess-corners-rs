"""UQ-27: per-corner asymmetry term on the UQ-30 real-data harness.

Usage (after ``../real_validation/detect.py <root>`` with the current
bindings installed):

    python export_uq27.py <dataset_root>

For every observation of ``SaddlePoint`` and ``Radon+SP`` in
``../real_validation/out/detections.pkl`` it refines the bindings' position
with the numpy SaddlePoint (``core.saddle``, radius 2), fits the asymmetry
term at it (``core.asym_full``, 9 x 9 window, multiplicative ramp) and
stores

- ``cov_prop``: the bindings' ``position_covariance`` (``Propagated``);
- ``T``: the asymmetry term ``(beta beta' - var L_beta L_beta')_+`` with
  ``var = max(sigma_n^2, 1/12)`` at the image's estimated noise sigma (the
  same sigma and floor as ``Propagated``);
- ``beta``, ``valid``, ``dpos`` (numpy minus bindings position).

Writes ``../real_validation/out/asym_<loc>.npz`` (keys ``cam``, ``token``,
``id`` as ``calibrate.py --cov`` expects) and
``cov_<loc>_prop.npz`` / ``cov_<loc>_prop_plus_T.npz`` in the ``--cov``
format. Invalid fits fall back to ``T = 0``. Aggregates are printed only.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RV = HERE.parent / "real_validation"
sys.path.insert(0, str(RV))

import core  # noqa: E402

LOCS = ("SaddlePoint", "Radon+SP")
RA = 4  # 9 x 9 fit window
MODE = "mul"


def term(img: np.ndarray, xy: np.ndarray, sigma: float):
    """(T, beta, valid, dpos) for positions ``xy`` (N, 2) of one image."""
    n = len(xy)
    T = np.zeros((n, 2, 2))
    beta = np.full((n, 2), np.nan)
    dpos = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    h, w = img.shape
    inb = (xy[:, 0] > core.KB + 3) & (xy[:, 0] < w - core.KB - 4) & (xy[:, 1] > core.KB + 3) & (xy[:, 1] < h - core.KB - 4)
    idx = np.where(inb)[0]
    if len(idx) == 0:
        return T, beta, valid, dpos
    var = max(sigma * sigma, core.MIN_VAR)
    f = img.astype(np.float64)
    c, ok, Lc = core.saddle(f, xy[idx], 2)
    a = core.asym_full(f, c, Lc, RA, MODE)
    b = a["beta"]
    Cb = core.cov_of(a["Lbeta"], var)
    Ti = core.psd_part(np.einsum("ni,nj->nij", b, b) - Cb)
    good = ok & np.isfinite(b).all(1) & (np.linalg.norm(b, axis=1) < 1.0) & np.isfinite(Ti).all(axis=(1, 2))
    good &= np.linalg.norm(c - xy[idx], axis=1) < 0.05
    T[idx[good]] = Ti[good]
    beta[idx[good]] = b[good]
    dpos[idx] = c - xy[idx]
    valid[idx[good]] = True
    return T, beta, valid, dpos


def main() -> None:
    root = Path(sys.argv[1])
    recs = pickle.load(open(RV / "out" / "detections.pkl", "rb"))
    acc = {loc: dict(cam=[], token=[], id=[], cov_prop=[], T=[], beta=[], valid=[], dpos=[], sigma=[]) for loc in LOCS}
    for r in recs:
        cam, tok = r["cam"], r["token"]
        img = cv2.imread(str(root / cam / f"{cam.capitalize()}_{tok}.png"), cv2.IMREAD_GRAYSCALE)
        sig = r["noise_sigma"]
        sig = float(sig) if sig is not None else 0.0
        for loc in LOCS:
            xy = r["xy"][loc]
            m = np.isfinite(xy[:, 0])
            T, beta, valid, dpos = term(img, xy[m], sig)
            a = acc[loc]
            k = int(m.sum())
            a["cam"].append(np.full(k, cam))
            a["token"].append(np.full(k, tok))
            a["id"].append(r["ids"][m])
            a["cov_prop"].append(r["cov"][loc][m])
            a["T"].append(T)
            a["beta"].append(beta)
            a["valid"].append(valid)
            a["dpos"].append(dpos)
            a["sigma"].append(np.full(k, sig))
        print(f"{cam} {tok} done", flush=True)
    for loc in LOCS:
        z = {k: np.concatenate(v) for k, v in acc[loc].items()}
        tag = loc.replace("+", "_")
        np.savez(RV / "out" / f"asym_{tag}.npz", **z)
        np.savez(RV / "out" / f"cov_{tag}_prop.npz", cam=z["cam"], token=z["token"], id=z["id"], cov=z["cov_prop"])
        np.savez(RV / "out" / f"cov_{tag}_prop_plus_T.npz", cam=z["cam"], token=z["token"], id=z["id"], cov=z["cov_prop"] + z["T"])
        v = z["valid"]
        dp = np.linalg.norm(z["dpos"][np.isfinite(z["dpos"][:, 0])], axis=1)
        tr = 0.5 * np.trace(z["T"][v], axis1=1, axis2=2)
        pr = 0.5 * np.trace(z["cov_prop"][v], axis1=1, axis2=2)
        print(
            f"{loc}: {len(v)} obs, valid {v.mean():.3f}; |numpy - bindings| median {np.median(dp):.1e} p99 {np.percentile(dp, 99):.1e} px; "
            f"|beta| median {np.median(np.linalg.norm(z['beta'][v], axis=1)):.4f} px; "
            f"sqrt(tr T/2) median {np.median(np.sqrt(tr)):.4f} p90 {np.percentile(np.sqrt(tr), 90):.4f}; "
            f"T>0 frac {(tr > 0).mean():.2f}; Propagated sigma median {np.median(np.sqrt(pr)):.4f}"
        )


if __name__ == "__main__":
    main()
