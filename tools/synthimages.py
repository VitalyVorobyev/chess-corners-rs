#!/usr/bin/env python3
"""
Synthetic chessboard generator with ground-truth corners.

Dependencies:
  pip install numpy opencv-python

Example:
  python synth_chessboard.py \
    --out out_synth --num 200 --seed 1 \
    --inner 9 6 --square_size 0.03 --pps 90 \
    --img_size 1280 960 --fx 1100 --fy 1100 \
    --yaw 35 --pitch 30 --roll 15 \
    --z_range 0.7 1.6 \
    --noise_sigma 3.0 --blur_sigma 0.8 \
    --gamma_range 0.9 1.2 --contrast_range 0.9 1.1 --brightness_range -10 10 \
    --vignetting 0.15
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import cv2


# ----------------------------- Math helpers -----------------------------

def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)

def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)

def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, pts_XY: np.ndarray) -> np.ndarray:
    """
    pts_XY: (N,2) on plane Z=0 in board coords.
    Returns: (N,2) pixel coords.
    """
    N = pts_XY.shape[0]
    pts3 = np.zeros((N, 3), dtype=np.float64)
    pts3[:, 0:2] = pts_XY
    pts_cam = (R @ pts3.T).T + t.reshape(1, 3)  # (N,3)
    Z = pts_cam[:, 2:3]
    # avoid divide by zero
    z_ok = np.maximum(Z, 1e-9)
    pts_norm = pts_cam[:, 0:2] / z_ok
    u = K[0, 0] * pts_norm[:, 0] + K[0, 2]
    v = K[1, 1] * pts_norm[:, 1] + K[1, 2]
    return np.stack([u, v], axis=1)

def pose_to_H_tex2img(K: np.ndarray, R: np.ndarray, t: np.ndarray, square_size: float, pps: int) -> np.ndarray:
    """
    Computes homography mapping texture pixel coords (u_tex, v_tex) -> image pixels (u_img, v_img).
    Texture pixels map to plane meters via scale s = square_size/pps.
    H = K [r1 r2 t] * S, where S maps tex pixels to plane coords.
    """
    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    Rt = np.concatenate([r1, r2, t.reshape(3, 1)], axis=1)  # 3x3
    H_plane2img = K @ Rt
    s = square_size / float(pps)
    S = np.array([[s, 0.0, 0.0],
                  [0.0, s, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    H = H_plane2img @ S
    return H


# ----------------------------- Rendering -----------------------------

def make_chess_texture(inner_x: int, inner_y: int, pps: int, invert: bool = False) -> np.ndarray:
    """
    OpenCV "boardSize" is number of inner corners (inner_x, inner_y).
    Number of squares is (inner_x+1, inner_y+1).
    Returns uint8 texture image.
    """
    squares_x = inner_x + 1
    squares_y = inner_y + 1
    W = squares_x * pps
    H = squares_y * pps
    tex = np.zeros((H, W), dtype=np.uint8)

    for sy in range(squares_y):
        for sx in range(squares_x):
            white = ((sx + sy) % 2 == 0)
            if invert:
                white = not white
            val = 255 if white else 0
            x0, x1 = sx * pps, (sx + 1) * pps
            y0, y1 = sy * pps, (sy + 1) * pps
            tex[y0:y1, x0:x1] = val

    return tex

def add_vignetting(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """
    strength ~ [0..0.5] typical.
    Multiplies by (1 - strength * r^2) with small random center offset.
    """
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    cx = w * (0.5 + rng.uniform(-0.05, 0.05))
    cy = h * (0.5 + rng.uniform(-0.05, 0.05))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / max(w, 1)
    dy = (yy - cy) / max(h, 1)
    r2 = dx * dx + dy * dy
    mask = 1.0 - strength * r2
    mask = np.clip(mask, 0.0, 1.0)

    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return img
    inv = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def add_noise(img: np.ndarray, noise_sigma: float, sp_prob: float, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)

    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
        out += noise

    if sp_prob > 0:
        m = rng.random(size=out.shape)
        out[m < sp_prob * 0.5] = 0.0
        out[m > 1.0 - sp_prob * 0.5] = 255.0

    return np.clip(out, 0, 255).astype(np.uint8)

def photometric_jitter(img: np.ndarray,
                       gamma_range: Tuple[float, float],
                       contrast_range: Tuple[float, float],
                       brightness_range: Tuple[float, float],
                       rng: np.random.Generator) -> np.ndarray:
    """
    contrast: multiply
    brightness: add
    then gamma
    """
    a = rng.uniform(contrast_range[0], contrast_range[1])
    b = rng.uniform(brightness_range[0], brightness_range[1])
    out = img.astype(np.float32) * a + b
    out = np.clip(out, 0, 255).astype(np.uint8)

    g = rng.uniform(gamma_range[0], gamma_range[1])
    out = apply_gamma(out, g)
    return out


# ----------------------------- Pose sampling -----------------------------

@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray
    yaw: float
    pitch: float
    roll: float

def sample_pose(rng: np.random.Generator,
                yaw_max: float, pitch_max: float, roll_max: float,
                z_range: Tuple[float, float],
                xy_range: Tuple[float, float]) -> Pose:
    yaw = rng.uniform(-yaw_max, yaw_max)
    pitch = rng.uniform(-pitch_max, pitch_max)
    roll = rng.uniform(-roll_max, roll_max)

    # Compose rotations (camera coords: +Z forward). This is a reasonable convention for synthesis.
    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)

    z = rng.uniform(z_range[0], z_range[1])
    tx = rng.uniform(-xy_range[0], xy_range[0])
    ty = rng.uniform(-xy_range[1], xy_range[1])
    t = np.array([tx, ty, z], dtype=np.float64)

    return Pose(R=R, t=t, yaw=yaw, pitch=pitch, roll=roll)

def all_visible(pts: np.ndarray, img_w: int, img_h: int, margin: int) -> bool:
    if pts.ndim != 2 or pts.shape[1] != 2:
        return False
    x = pts[:, 0]
    y = pts[:, 1]
    return (x.min() >= margin and y.min() >= margin and
            x.max() < (img_w - margin) and y.max() < (img_h - margin))

def positive_depth(R: np.ndarray, t: np.ndarray, pts_XY: np.ndarray) -> bool:
    # Check Zc > 0 for all points
    N = pts_XY.shape[0]
    pts3 = np.zeros((N, 3), dtype=np.float64)
    pts3[:, 0:2] = pts_XY
    pts_cam = (R @ pts3.T).T + t.reshape(1, 3)
    return np.all(pts_cam[:, 2] > 1e-6)


# ----------------------------- Main generation -----------------------------

def generate_one(idx: int,
                 tex: np.ndarray,
                 inner_x: int, inner_y: int,
                 square_size: float, pps: int,
                 img_w: int, img_h: int,
                 K: np.ndarray,
                 yaw_max: float, pitch_max: float, roll_max: float,
                 z_range: Tuple[float, float],
                 margin: int,
                 noise_sigma: float,
                 sp_prob: float,
                 blur_sigma: float,
                 gamma_range: Tuple[float, float],
                 contrast_range: Tuple[float, float],
                 brightness_range: Tuple[float, float],
                 vignetting: float,
                 rng: np.random.Generator,
                 max_tries: int = 500) -> Tuple[np.ndarray, Dict[str, Any]]:
    squares_x = inner_x + 1
    squares_y = inner_y + 1
    board_w = squares_x * square_size
    board_h = squares_y * square_size

    # Outer corners in plane coords for visibility test
    outer_XY = np.array([
        [0.0, 0.0],
        [board_w, 0.0],
        [board_w, board_h],
        [0.0, board_h],
    ], dtype=np.float64)

    # Inner corners ground truth: OpenCV order is row-major (y then x)
    # We place origin at the top-left outer corner.
    inner_pts_XY = []
    for iy in range(1, inner_y + 1):
        for ix in range(1, inner_x + 1):
            inner_pts_XY.append([ix * square_size, iy * square_size])
    inner_pts_XY = np.array(inner_pts_XY, dtype=np.float64)

    # Choose translation ranges relative to board size (enough to cover various placements)
    xy_range = (0.6 * board_w, 0.6 * board_h)

    pose = None
    for _ in range(max_tries):
        p = sample_pose(rng, yaw_max, pitch_max, roll_max, z_range, xy_range)

        if not positive_depth(p.R, p.t, outer_XY):
            continue

        outer_uv = project_points(K, p.R, p.t, outer_XY)
        if not all_visible(outer_uv, img_w, img_h, margin):
            continue

        # also ensure inner corners are visible (stricter)
        inner_uv = project_points(K, p.R, p.t, inner_pts_XY)
        if not all_visible(inner_uv, img_w, img_h, margin):
            continue

        pose = p
        break

    if pose is None:
        raise RuntimeError("Failed to sample a visible pose. Try widening ranges or reducing margin.")

    # Build homography for warping texture -> image
    H = pose_to_H_tex2img(K, pose.R, pose.t, square_size, pps)

    # Random background (slightly textured)
    bg = rng.integers(100, 180, size=(img_h, img_w), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=rng.uniform(0.0, 1.2))

    # Warp texture and mask
    mask_tex = np.ones_like(tex, dtype=np.uint8) * 255
    warped = cv2.warpPerspective(tex, H, (img_w, img_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    mask = cv2.warpPerspective(mask_tex, H, (img_w, img_h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    img = bg.copy()
    img[mask > 0] = warped[mask > 0]

    # Photometric effects
    img = photometric_jitter(img, gamma_range, contrast_range, brightness_range, rng)
    img = add_vignetting(img, vignetting, rng)

    # Blur
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Noise
    img = add_noise(img, noise_sigma=noise_sigma, sp_prob=sp_prob, rng=rng)

    # Ground truth (inner corners)
    inner_uv = project_points(K, pose.R, pose.t, inner_pts_XY).astype(np.float32)

    meta = {
        "index": idx,
        "pose": {
            "yaw_deg": float(pose.yaw),
            "pitch_deg": float(pose.pitch),
            "roll_deg": float(pose.roll),
            "R": pose.R.tolist(),
            "t": pose.t.tolist(),
        },
        "H_tex2img": H.tolist(),
        "gt_corners_uv": inner_uv.reshape(-1, 2).tolist(),
    }

    return img, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--inner", type=int, nargs=2, default=[9, 6], metavar=("NX", "NY"),
                    help="Number of inner corners (OpenCV boardSize).")
    ap.add_argument("--square_size", type=float, default=0.03, help="Square size in meters (or arbitrary units).")
    ap.add_argument("--pps", type=int, default=90, help="Pixels per square in the texture.")

    ap.add_argument("--img_size", type=int, nargs=2, default=[1280, 960], metavar=("W", "H"))
    ap.add_argument("--fx", type=float, default=1100.0)
    ap.add_argument("--fy", type=float, default=1100.0)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    ap.add_argument("--yaw", type=float, default=35.0, help="Max abs yaw in degrees.")
    ap.add_argument("--pitch", type=float, default=30.0, help="Max abs pitch in degrees.")
    ap.add_argument("--roll", type=float, default=15.0, help="Max abs roll in degrees.")
    ap.add_argument("--z_range", type=float, nargs=2, default=[0.2, 0.7], metavar=("ZMIN", "ZMAX"),
                    help="Camera Z translation range (distance to board plane).")
    ap.add_argument("--margin", type=int, default=20, help="Pixel margin to keep board inside image.")

    ap.add_argument("--noise_sigma", type=float, default=2.0, help="Additive Gaussian noise sigma (pixel intensity).")
    ap.add_argument("--sp_prob", type=float, default=0.0, help="Salt/pepper probability (0..1).")
    ap.add_argument("--blur_sigma", type=float, default=0.7, help="Gaussian blur sigma (pixels).")

    ap.add_argument("--gamma_range", type=float, nargs=2, default=[0.9, 1.2])
    ap.add_argument("--contrast_range", type=float, nargs=2, default=[0.9, 1.1])
    ap.add_argument("--brightness_range", type=float, nargs=2, default=[-10.0, 10.0])

    ap.add_argument("--vignetting", type=float, default=0.0, help="Vignetting strength ~0..0.3")
    ap.add_argument("--invert", action="store_true", help="Invert square colors.")
    ap.add_argument("--jpeg_quality", type=int, default=0,
                    help="If >0, save JPG with given quality; else save PNG.")

    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    inner_x, inner_y = args.inner
    img_w, img_h = args.img_size

    cx = args.cx if args.cx is not None else (img_w * 0.5)
    cy = args.cy if args.cy is not None else (img_h * 0.5)
    K = np.array([[args.fx, 0.0, cx],
                  [0.0, args.fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    tex = make_chess_texture(inner_x, inner_y, args.pps, invert=args.invert)

    rng = np.random.default_rng(args.seed)

    dataset = {
        "generator": "synth_chessboard.py",
        "num": args.num,
        "inner_corners": [inner_x, inner_y],
        "square_size": args.square_size,
        "pps": args.pps,
        "img_size": [img_w, img_h],
        "K": K.tolist(),
        "images": []
    }

    for i in range(args.num):
        img, meta = generate_one(
            idx=i,
            tex=tex,
            inner_x=inner_x, inner_y=inner_y,
            square_size=args.square_size, pps=args.pps,
            img_w=img_w, img_h=img_h,
            K=K,
            yaw_max=args.yaw, pitch_max=args.pitch, roll_max=args.roll,
            z_range=(args.z_range[0], args.z_range[1]),
            margin=args.margin,
            noise_sigma=args.noise_sigma,
            sp_prob=args.sp_prob,
            blur_sigma=args.blur_sigma,
            gamma_range=(args.gamma_range[0], args.gamma_range[1]),
            contrast_range=(args.contrast_range[0], args.contrast_range[1]),
            brightness_range=(args.brightness_range[0], args.brightness_range[1]),
            vignetting=args.vignetting,
            rng=rng,
        )

        if args.jpeg_quality and args.jpeg_quality > 0:
            fname = f"{i:06d}.jpg"
            path = out_dir / "images" / fname
            cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        else:
            fname = f"{i:06d}.png"
            path = out_dir / "images" / fname
            cv2.imwrite(str(path), img)

        meta["file_name"] = f"images/{fname}"
        dataset["images"].append(meta)

        if (i + 1) % 25 == 0:
            print(f"Generated {i+1}/{args.num}")

    with open(out_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Done. Wrote {args.num} images + dataset.json to: {out_dir}")


if __name__ == "__main__":
    main()
