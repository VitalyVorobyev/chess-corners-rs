"""Board-predicted axis directions for the real-data part of the
orientation-σ study (ChArUco stereo set of ``refiner_robustness/real_validation``).

Usage: python real.py DATASET_ROOT OUT_DIR

For every image, ``cv2.aruco.CharucoDetector`` gives corner ids; for each id a
homography is fitted to the detected corners within two cells (at least 12 of
the 25), and the true local axis directions are the images of the board x and
y directions under it (its Jacobian at the corner). Writes
``OUT_DIR/truth/<n>.csv`` and ``OUT_DIR/list.tsv`` (images are read in place,
never copied) for ``orientation_sigma_study files``; feed the resulting CSV to
``analyze.py --fit-gen optical`` together with the synthetic CSVs. Report only
aggregates: the dataset is private.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "refiner_robustness" / "real_validation"))
import common as C  # noqa: E402

HALF = 2
MIN_PTS = 12


def local_axes(ids, ref, n_inner):
    ij = np.column_stack([ids % n_inner, ids // n_inner]).astype(float)
    pos = {int(i): k for k, i in enumerate(ids)}
    out = np.full((len(ids), 2), np.nan)
    resid = []
    for k, (i, j) in enumerate(ij.astype(int)):
        nb = [pos[jj * n_inner + ii] for jj in range(j - HALF, j + HALF + 1) for ii in range(i - HALF, i + HALF + 1)
              if 0 <= ii < n_inner and 0 <= jj < n_inner and (jj * n_inner + ii) in pos]
        if len(nb) < MIN_PTS:
            continue
        src = ij[nb] - ij[k]
        h, _ = cv2.findHomography(src, ref[nb], 0)
        if h is None:
            continue
        p = np.column_stack([src, np.ones(len(nb))]) @ h.T
        resid.append(np.linalg.norm(p[:, :2] / p[:, 2:3] - ref[nb], axis=1))
        # Jacobian of the homography at the corner (src = 0).
        w = h[2, 2]
        du = (h[:2, 0] * w - h[:2, 2] * h[2, 0]) / w**2
        dv = (h[:2, 1] * w - h[:2, 2] * h[2, 1]) / w**2
        out[k] = np.arctan2(du[1], du[0]), np.arctan2(dv[1], dv[0])
    return out, (np.concatenate(resid) if resid else np.zeros(0))


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "truth").mkdir(parents=True, exist_ok=True)
    det = cv2.aruco.CharucoDetector(C.board())
    n_inner = C.NSQ - 1
    rows, res_all, n = [], [], 0
    for cam in C.CAMS:
        for f in sorted((root / cam).glob("*.png")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            cc, ids, _, _ = det.detectBoard(img)
            if ids is None or len(ids) < 30:
                continue
            ids = ids.ravel().astype(int)
            ref = cc.reshape(-1, 2).astype(np.float64)
            ax, res = local_axes(ids, ref, n_inner)
            res_all.append(res)
            t = np.column_stack([ref, ax])
            tpath = out / "truth" / f"{n}.csv"
            np.savetxt(tpath, t, delimiter=",", header="x,y,a1,a2", comments="", fmt="%.6f")
            rows.append(f"{f}\t{tpath}\treal,nan,nan,nan,nan,nan,img{n},{cam[-1]}")
            n += 1
    (out / "list.tsv").write_text("\n".join(rows) + "\n")
    r = np.concatenate(res_all)
    print(f"{n} images; local-homography fit residual: median {np.median(r):.3f} px, p90 {np.quantile(r, 0.9):.3f} px")


if __name__ == "__main__":
    main()
