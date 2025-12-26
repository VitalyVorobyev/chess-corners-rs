#!/usr/bin/env python3
"""
Synthetic accuracy benchmark for corner detectors.

This script evaluates three detectors on a synthetic dataset produced by
`tools/synthimages.py` (a `dataset.json` file plus an `images/` folder with
rendered grayscale chessboards and per-image ground truth corner coordinates).

Detectors:
  - chess-corners (Rust CLI)
  - OpenCV Harris (+ cornerSubPix)
  - OpenCV findChessboardCornersSB

Matching rule:
  A detection is considered correct if it can be matched to an unmatched ground
  truth corner within `--match-threshold-px`. Unmatched detections are counted
  as false positives; unmatched GT corners as false negatives.

Outputs:
  - JSON report (`report.json`) with per-detector + per-image metrics
  - publication-friendly plots (hist/CDF/scatter/PRF + worst-case overlays)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from detection.chess import DEFAULT_BIN, run_chess_corners

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_DIR = ROOT / "testdata" / "synthetic"
DEFAULT_DATASET_JSON = DEFAULT_DATASET_DIR / "dataset.json"
DEFAULT_CFG = ROOT / "config" / "config_single.json"
DEFAULT_OUTDIR = ROOT / "testdata" / "out" / "synthetic_accuracy"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr.reshape(-1, 2)


def _safe_div(num: float, denom: float) -> float | None:
    if denom == 0.0:
        return None
    return num / denom


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def _summarize_1d(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


@dataclass
class MatchResult:
    gt_count: int
    det_count: int
    threshold_px: float
    matched_pairs: np.ndarray  # (tp, 2) [det_idx, gt_idx]
    unmatched_det_idx: np.ndarray
    unmatched_det_kind: List[str]  # for each unmatched det: "spurious" or "duplicate"
    unmatched_gt_idx: np.ndarray
    dx_px: np.ndarray
    dy_px: np.ndarray
    err_px: np.ndarray

    @property
    def tp(self) -> int:
        return int(self.matched_pairs.shape[0])

    @property
    def fp(self) -> int:
        return int(self.unmatched_det_idx.size)

    @property
    def fn(self) -> int:
        return int(self.unmatched_gt_idx.size)

    def fp_spurious(self) -> int:
        return int(sum(k == "spurious" for k in self.unmatched_det_kind))

    def fp_duplicate(self) -> int:
        return int(sum(k == "duplicate" for k in self.unmatched_det_kind))

    def precision(self) -> float | None:
        return _safe_div(float(self.tp), float(self.tp + self.fp))

    def recall(self) -> float | None:
        return _safe_div(float(self.tp), float(self.tp + self.fn))

    def f1(self) -> float | None:
        return _f1(self.precision(), self.recall())


def match_detections_to_gt(detected: np.ndarray, gt: np.ndarray, threshold_px: float) -> MatchResult:
    detected = _as_points(detected)
    gt = _as_points(gt)
    det_count = int(detected.shape[0])
    gt_count = int(gt.shape[0])

    if det_count == 0 and gt_count == 0:
        return MatchResult(
            gt_count=0,
            det_count=0,
            threshold_px=threshold_px,
            matched_pairs=np.empty((0, 2), dtype=np.int32),
            unmatched_det_idx=np.empty((0,), dtype=np.int32),
            unmatched_det_kind=[],
            unmatched_gt_idx=np.empty((0,), dtype=np.int32),
            dx_px=np.empty((0,), dtype=float),
            dy_px=np.empty((0,), dtype=float),
            err_px=np.empty((0,), dtype=float),
        )

    if gt_count == 0:
        unmatched_det_idx = np.arange(det_count, dtype=np.int32)
        return MatchResult(
            gt_count=0,
            det_count=det_count,
            threshold_px=threshold_px,
            matched_pairs=np.empty((0, 2), dtype=np.int32),
            unmatched_det_idx=unmatched_det_idx,
            unmatched_det_kind=["spurious"] * int(unmatched_det_idx.size),
            unmatched_gt_idx=np.empty((0,), dtype=np.int32),
            dx_px=np.empty((0,), dtype=float),
            dy_px=np.empty((0,), dtype=float),
            err_px=np.empty((0,), dtype=float),
        )

    if det_count == 0:
        unmatched_gt_idx = np.arange(gt_count, dtype=np.int32)
        return MatchResult(
            gt_count=gt_count,
            det_count=0,
            threshold_px=threshold_px,
            matched_pairs=np.empty((0, 2), dtype=np.int32),
            unmatched_det_idx=np.empty((0,), dtype=np.int32),
            unmatched_det_kind=[],
            unmatched_gt_idx=unmatched_gt_idx,
            dx_px=np.empty((0,), dtype=float),
            dy_px=np.empty((0,), dtype=float),
            err_px=np.empty((0,), dtype=float),
        )

    diff = detected[:, None, :] - gt[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (M, N)

    det_used = np.zeros((det_count,), dtype=bool)
    gt_used = np.zeros((gt_count,), dtype=bool)

    candidates = np.argwhere(dists <= threshold_px)
    if candidates.size:
        cand_dists = dists[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(cand_dists, kind="stable")
        matches: List[Tuple[int, int]] = []
        for k in order:
            det_idx = int(candidates[k, 0])
            gt_idx = int(candidates[k, 1])
            if det_used[det_idx] or gt_used[gt_idx]:
                continue
            det_used[det_idx] = True
            gt_used[gt_idx] = True
            matches.append((det_idx, gt_idx))
        matched_pairs = np.array(matches, dtype=np.int32).reshape(-1, 2)
    else:
        matched_pairs = np.empty((0, 2), dtype=np.int32)

    unmatched_det_idx = np.where(~det_used)[0].astype(np.int32)
    unmatched_gt_idx = np.where(~gt_used)[0].astype(np.int32)

    if matched_pairs.size:
        delta = detected[matched_pairs[:, 0]] - gt[matched_pairs[:, 1]]
        dx_px = delta[:, 0]
        dy_px = delta[:, 1]
        err_px = np.linalg.norm(delta, axis=1)
    else:
        dx_px = np.empty((0,), dtype=float)
        dy_px = np.empty((0,), dtype=float)
        err_px = np.empty((0,), dtype=float)

    nn_gt_dist = dists.min(axis=1)
    fp_nn = nn_gt_dist[unmatched_det_idx]
    unmatched_det_kind = ["spurious" if float(d) > threshold_px else "duplicate" for d in fp_nn]

    return MatchResult(
        gt_count=gt_count,
        det_count=det_count,
        threshold_px=threshold_px,
        matched_pairs=matched_pairs,
        unmatched_det_idx=unmatched_det_idx,
        unmatched_det_kind=unmatched_det_kind,
        unmatched_gt_idx=unmatched_gt_idx,
        dx_px=dx_px,
        dy_px=dy_px,
        err_px=err_px,
    )


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_error_hist(errors: Dict[str, np.ndarray], threshold_px: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "chess-corners": "#1d3557",
        "harris": "#e63946",
        "chessboard_sb": "#2a9d8f",
    }
    bins = np.linspace(0.0, threshold_px, 50)

    for name, arr in errors.items():
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            histtype="stepfilled",
            alpha=0.35,
            edgecolor=colors.get(name, "black"),
            color=colors.get(name, "gray"),
            label=f"{name} (n={arr.size})",
        )
        ax.axvline(float(np.median(arr)), color=colors.get(name, "black"), linewidth=1.2)

    ax.axvline(threshold_px, color="black", linestyle=":", linewidth=1.2, label="match threshold")
    ax.set_title("Matched corner error (distance to GT)")
    ax.set_xlabel("Error (px)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_error_cdf(errors: Dict[str, np.ndarray], threshold_px: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "chess-corners": "#1d3557",
        "harris": "#e63946",
        "chessboard_sb": "#2a9d8f",
    }

    for name, arr in errors.items():
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        x = np.sort(arr)
        y = (np.arange(1, x.size + 1) / x.size).astype(float)
        ax.plot(x, y, color=colors.get(name, "black"), linewidth=2.0, label=f"{name} (n={x.size})")

    ax.axvline(threshold_px, color="black", linestyle=":", linewidth=1.2, label="match threshold")
    ax.set_xlim(left=0.0, right=threshold_px)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Matched corner error CDF")
    ax.set_xlabel("Error (px)")
    ax.set_ylabel("Fraction ≤ x")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_dxdy_scatter(dxdy: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True, sharey=True)
    names = ["chess-corners", "harris", "chessboard_sb"]
    colors = {
        "chess-corners": "#1d3557",
        "harris": "#e63946",
        "chessboard_sb": "#2a9d8f",
    }

    lim = 1.5
    for ax, name in zip(axes, names):
        dx, dy = dxdy.get(name, (np.empty((0,)), np.empty((0,))))
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
        if dx.size:
            ax.scatter(
                dx,
                dy,
                s=12,
                alpha=0.35,
                color=colors.get(name, "black"),
                edgecolors="none",
            )
            lim = max(lim, float(np.max(np.abs(dx))) * 1.1, float(np.max(np.abs(dy))) * 1.1)
        ax.set_title(name)
        ax.set_xlabel("dx (px)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("dy (px)")
    lim = min(lim, 6.0)
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle("Matched residuals (detected - GT)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_prf(summary: Dict[str, Dict[str, float | None]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names = ["chess-corners", "harris", "chessboard_sb"]
    metrics = ["precision", "recall", "f1"]
    colors = {"precision": "#457b9d", "recall": "#2a9d8f", "f1": "#e63946"}

    x = np.arange(len(names))
    width = 0.22
    for i, m in enumerate(metrics):
        vals: List[float] = []
        for name in names:
            v = summary.get(name, {}).get(m)
            vals.append(0.0 if v is None else float(v))
        ax.bar(x + (i - 1) * width, vals, width=width, label=m, color=colors[m], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Precision / Recall / F1 (GT matching)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_counts(summary: Dict[str, Dict[str, float | None]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names = ["chess-corners", "harris", "chessboard_sb"]
    keys = ["tp", "fp", "fn"]
    colors = {"tp": "#2a9d8f", "fp": "#e63946", "fn": "#ffb703"}

    x = np.arange(len(names))
    width = 0.25
    for i, k in enumerate(keys):
        vals = [float(summary.get(name, {}).get(k) or 0.0) for name in names]
        ax.bar(x + (i - 1) * width, vals, width=width, label=k, color=colors[k], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Match counts across dataset")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _detect_chess_corners(img_path: Path, cfg_path: Path, bin_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    corners = run_chess_corners(img_path, cfg_path, bin_path)
    total_ms = (time.perf_counter() - t0) * 1000.0
    pts = (
        np.array([[c.x, c.y] for c in corners], dtype=float)
        if corners
        else np.empty((0, 2), dtype=float)
    )
    return pts, {"total_ms": float(total_ms)}


def _detect_harris(gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    from detection.harris import run_harris_corners

    pts, metrics = run_harris_corners(gray)
    metrics = dict(metrics)
    metrics["total_ms"] = float(metrics.get("detect_ms", 0.0)) + float(metrics.get("refine_ms", 0.0)) + float(
        metrics.get("merge_ms", 0.0)
    )
    return pts, metrics


def _detect_chessboard(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, float]]:
    from detection.chessboard import run_chessboard_corners

    pts, metrics = run_chessboard_corners(gray, pattern_size=pattern_size)
    metrics = dict(metrics)
    metrics.setdefault("total_ms", float(metrics.get("detect_ms", 0.0)))
    return pts, metrics


def _pick_worst_images(
    images: List[Dict[str, Any]],
    per_image: Dict[int, Dict[str, Any]],
    detector_key: str,
    top_k: int,
) -> List[int]:
    scored: List[Tuple[int, int, int, float]] = []
    for meta in images:
        idx = int(meta["index"])
        det = per_image[idx][detector_key]
        fn = int(det["fn"])
        fp = int(det["fp"])
        max_err = float(det.get("error_stats", {}).get("max", 0.0) or 0.0)
        scored.append((idx, fn, fp, max_err))
    scored.sort(key=lambda t: (-t[1], -t[2], -t[3], t[0]))
    return [idx for (idx, _, _, _) in scored[: max(0, top_k)]]


def _save_worst_case_grid(
    *,
    dataset_dir: Path,
    images: List[Dict[str, Any]],
    per_image: Dict[int, Dict[str, Any]],
    detector_key: str,
    out_path: Path,
    top_k: int,
    cfg_path: Path,
    bin_path: Path,
    pattern_size: Tuple[int, int],
    match_threshold_px: float,
) -> None:
    worst = _pick_worst_images(images, per_image, detector_key, top_k)
    if not worst:
        return

    cols = min(4, len(worst))
    rows = int(math.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6), squeeze=False)
    fig.suptitle(f"Worst cases: {detector_key}", fontsize=14)

    for ax in axes.ravel():
        ax.axis("off")

    # Map index -> meta for quick lookup.
    by_idx = {int(m["index"]): m for m in images}

    for plot_idx, idx in enumerate(worst):
        meta = by_idx[idx]
        img_path = dataset_dir / meta["file_name"]
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        gt = _as_points(meta["gt_corners_uv"])

        if detector_key == "chess-corners":
            det_pts, _ = _detect_chess_corners(img_path, cfg_path, bin_path)
        elif detector_key == "harris":
            det_pts, _ = _detect_harris(gray)
        elif detector_key == "chessboard_sb":
            det_pts, _ = _detect_chessboard(gray, pattern_size)
        else:
            raise ValueError(f"unknown detector: {detector_key}")

        match = match_detections_to_gt(det_pts, gt, match_threshold_px)
        matched_det = match.matched_pairs[:, 0] if match.matched_pairs.size else np.empty((0,), dtype=int)

        r = plot_idx // cols
        c = plot_idx % cols
        ax = axes[r][c]
        ax.imshow(gray, cmap="gray")
        ax.scatter(gt[:, 0], gt[:, 1], s=14, c="#2a9d8f", marker="+", linewidths=1.2, label="GT")

        if matched_det.size:
            ax.scatter(
                det_pts[matched_det, 0],
                det_pts[matched_det, 1],
                s=10,
                c="#457b9d",
                marker="o",
                linewidths=0.0,
                alpha=0.8,
                label="matched det",
            )
        if match.unmatched_det_idx.size:
            ax.scatter(
                det_pts[match.unmatched_det_idx, 0],
                det_pts[match.unmatched_det_idx, 1],
                s=16,
                facecolors="none",
                edgecolors="#e63946",
                marker="o",
                linewidths=1.0,
                alpha=0.9,
                label="FP",
            )
        if match.unmatched_gt_idx.size:
            ax.scatter(
                gt[match.unmatched_gt_idx, 0],
                gt[match.unmatched_gt_idx, 1],
                s=28,
                c="#ffb703",
                marker="x",
                linewidths=1.2,
                alpha=0.9,
                label="FN",
            )

        ax.set_title(f"{idx:06d} tp={match.tp} fp={match.fp} fn={match.fn}", fontsize=10)
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.9)

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark chess-corners, Harris, and findChessboardCornersSB on a synthetic dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help="Path to dataset.json produced by tools/synthimages.py.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional dataset root directory (defaults to dataset.json parent).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CFG,
        help="Config JSON for chess-corners (image/output fields are overridden).",
    )
    parser.add_argument(
        "--bin",
        type=Path,
        default=DEFAULT_BIN,
        help="Path to the chess-corners binary.",
    )
    parser.add_argument(
        "--match-threshold-px",
        type=float,
        default=2.5,
        help="Max distance (px) to count a detection as matching a GT corner.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images (0 = all).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for report and plots.",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=8,
        help="How many worst-case images to include in per-detector overlay grids.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (also saves them).",
    )
    args = parser.parse_args()

    if np is None or plt is None:
        raise SystemExit(
            "Missing Python dependencies.\n"
            "Install them in a virtualenv/conda env, e.g.:\n"
            "  python3.12 -m pip install numpy matplotlib\n"
            "and for the OpenCV-based detectors:\n"
            "  python3.12 -m pip install opencv-python\n"
        )
    if cv2 is None:
        raise SystemExit(
            "Missing optional dependency `cv2` (OpenCV Python bindings).\n"
            "Install it (and numpy/matplotlib) in a compatible Python env, e.g.:\n"
            "  python3.12 -m pip install opencv-python numpy matplotlib\n"
            "Note: `opencv-python` wheels may not be available for Python 3.13 yet."
        )

    if not args.dataset.exists():
        raise SystemExit(f"Dataset JSON not found: {args.dataset}")
    if not args.config.exists():
        raise SystemExit(f"Config JSON not found: {args.config}")
    if not args.bin.exists():
        raise SystemExit(
            f"Missing binary at {args.bin}; build it with `cargo build -p chess-corners --release`."
        )

    dataset = _read_json(args.dataset)
    dataset_dir = args.dataset_dir or args.dataset.parent
    images = list(dataset.get("images", []))

    if not images:
        raise SystemExit(f"No images listed in {args.dataset}")

    images.sort(key=lambda d: int(d.get("index", 0)))
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    pattern = tuple(int(x) for x in dataset.get("inner_corners", [9, 6]))
    expected_gt = int(pattern[0] * pattern[1])

    _apply_plot_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    overlays_dir = args.outdir / "overlays"

    agg: Dict[str, Dict[str, Any]] = {
        "chess-corners": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "fp_spurious": 0,
            "fp_duplicate": 0,
            "err": [],
            "dx": [],
            "dy": [],
        },
        "harris": {"tp": 0, "fp": 0, "fn": 0, "fp_spurious": 0, "fp_duplicate": 0, "err": [], "dx": [], "dy": []},
        "chessboard_sb": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "fp_spurious": 0,
            "fp_duplicate": 0,
            "err": [],
            "dx": [],
            "dy": [],
        },
    }

    per_image: Dict[int, Dict[str, Any]] = {}

    for k, meta in enumerate(images):
        idx = int(meta["index"])
        img_path = dataset_dir / meta["file_name"]
        gt = _as_points(meta["gt_corners_uv"])
        if gt.shape[0] != expected_gt:
            print(f"Warning: image {idx} GT has {gt.shape[0]} corners (expected {expected_gt}); continuing.")

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        chess_pts, chess_trace = _detect_chess_corners(img_path, args.config, args.bin)
        harris_pts, harris_trace = _detect_harris(gray)
        chessboard_pts, chessboard_trace = _detect_chessboard(gray, pattern)

        det_inputs = {
            "chess-corners": (chess_pts, chess_trace),
            "harris": (harris_pts, harris_trace),
            "chessboard_sb": (chessboard_pts, chessboard_trace),
        }

        per_image[idx] = {"file_name": meta["file_name"], "gt_count": int(gt.shape[0])}

        for det_name, (det_pts, trace) in det_inputs.items():
            match = match_detections_to_gt(det_pts, gt, args.match_threshold_px)
            entry = {
                "gt_count": match.gt_count,
                "det_count": match.det_count,
                "tp": match.tp,
                "fp": match.fp,
                "fn": match.fn,
                "fp_spurious": match.fp_spurious(),
                "fp_duplicate": match.fp_duplicate(),
                "precision": match.precision(),
                "recall": match.recall(),
                "f1": match.f1(),
                "error_stats": _summarize_1d(match.err_px),
                "trace_ms": {k: float(v) for k, v in (trace or {}).items()},
            }
            per_image[idx][det_name] = entry

            a = agg[det_name]
            a["tp"] += match.tp
            a["fp"] += match.fp
            a["fn"] += match.fn
            a["fp_spurious"] += match.fp_spurious()
            a["fp_duplicate"] += match.fp_duplicate()
            a["err"].append(match.err_px)
            a["dx"].append(match.dx_px)
            a["dy"].append(match.dy_px)

        if (k + 1) % 10 == 0 or (k + 1) == len(images):
            print(f"Processed {k+1}/{len(images)} images")

    summary: Dict[str, Dict[str, float | None]] = {}
    errors_for_plots: Dict[str, np.ndarray] = {}
    dxdy_for_plots: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, a in agg.items():
        err = np.concatenate([x for x in a["err"] if getattr(x, "size", 0)]) if a["err"] else np.array([])
        dx = np.concatenate([x for x in a["dx"] if getattr(x, "size", 0)]) if a["dx"] else np.array([])
        dy = np.concatenate([x for x in a["dy"] if getattr(x, "size", 0)]) if a["dy"] else np.array([])

        tp = float(a["tp"])
        fp = float(a["fp"])
        fn = float(a["fn"])
        precision = _safe_div(tp, tp + fp) if (tp + fp) else None
        recall = _safe_div(tp, tp + fn) if (tp + fn) else None
        f1 = _f1(precision, recall)
        summary[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fp_spurious": float(a["fp_spurious"]),
            "fp_duplicate": float(a["fp_duplicate"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "error_stats": _summarize_1d(err),
        }

        errors_for_plots[name] = err
        dxdy_for_plots[name] = (dx, dy)

    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_error_hist(errors_for_plots, args.match_threshold_px, plots_dir / "error_hist.png")
    _plot_error_cdf(errors_for_plots, args.match_threshold_px, plots_dir / "error_cdf.png")
    _plot_dxdy_scatter(dxdy_for_plots, plots_dir / "dxdy_scatter.png")
    _plot_prf(summary, plots_dir / "prf.png")
    _plot_counts(summary, plots_dir / "counts.png")

    overlays_dir.mkdir(parents=True, exist_ok=True)
    for det_key in ("chess-corners", "harris", "chessboard_sb"):
        _save_worst_case_grid(
            dataset_dir=dataset_dir,
            images=images,
            per_image=per_image,
            detector_key=det_key,
            out_path=overlays_dir / f"worst_{det_key}.png",
            top_k=max(0, int(args.worst_k)),
            cfg_path=args.config,
            bin_path=args.bin,
            pattern_size=pattern,
            match_threshold_px=float(args.match_threshold_px),
        )

    report = {
        "schema_version": 1,
        "dataset": {
            "dataset_json": str(args.dataset),
            "dataset_dir": str(dataset_dir),
            "num_images": len(images),
            "inner_corners": list(pattern),
            "expected_gt_corners_per_image": expected_gt,
        },
        "run": {
            "match_threshold_px": float(args.match_threshold_px),
            "config": str(args.config),
            "bin": str(args.bin),
        },
        "summary": summary,
        "images": [
            {
                "index": idx,
                "file_name": per_image[idx]["file_name"],
                "gt_count": per_image[idx]["gt_count"],
                "detectors": {k: v for k, v in per_image[idx].items() if k in agg},
            }
            for idx in sorted(per_image.keys())
        ],
        "outputs": {
            "outdir": str(args.outdir),
            "plots_dir": str(plots_dir),
            "overlays_dir": str(overlays_dir),
        },
    }
    report_path = args.outdir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report: {report_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
