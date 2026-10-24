"""Stage 1: ChArUco reference IDs + every localizer on every image.

Usage: python detect.py <dataset_root>

Writes ``out/detections.pkl``: a list of per-image records ``{cam, token,
stage (a, b, x, y, z), ids (M,), ref (M, 2), xy {loc: (M, 2) NaN if
unmatched}, cov {loc: (M, 2, 2) NaN if unavailable}, noise_sigma}``. Corner
identity comes from ``cv2.aruco.CharucoDetector``; a localizer's detection is
assigned to a reference corner within 2 px.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

import common as C


def main() -> None:
    import chess_corners as cc

    root = Path(sys.argv[1])
    det = cv2.aruco.CharucoDetector(C.board())
    recs = []
    for cam in C.CAMS:
        for f in sorted((root / cam).glob("*.png")):
            p = C.parse_token(f.name)
            if p is None:
                continue
            tok, stage = p
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            cc_, ids, _, _ = det.detectBoard(img)
            if ids is None or len(ids) < 30:
                print(f"skip {cam} {tok}: no board")
                continue
            ref = cc_.reshape(-1, 2).astype(np.float64)
            r = dict(cam=cam, token=tok, stage=stage, ids=ids.ravel().astype(int), ref=ref, xy={}, cov={})
            r["noise_sigma"] = cc.estimate_noise_sigma(img)
            r["shape"] = img.shape
            for loc in C.LOCALIZERS:
                xy, cov = C.localize(img, loc, ref)
                j = C.match(xy, ref)
                ok = j >= 0
                P = np.full((len(ref), 2), np.nan)
                P[ok] = xy[j[ok]]
                S = np.full((len(ref), 2, 2), np.nan)
                if cov is not None:
                    S[ok] = cov[j[ok]]
                r["xy"][loc] = P
                r["cov"][loc] = S
            recs.append(r)
            rate = " ".join(f"{loc}:{np.isfinite(r['xy'][loc][:, 0]).mean():.2f}" for loc in C.LOCALIZERS)
            print(f"{cam} {tok}: {len(ref)} ids  {rate}", flush=True)
    C.OUT.mkdir(exist_ok=True)
    with open(C.OUT / "detections.pkl", "wb") as fh:
        pickle.dump(recs, fh)


if __name__ == "__main__":
    main()
