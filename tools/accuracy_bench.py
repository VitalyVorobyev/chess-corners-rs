import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from trace.parser import parse_trace
from trace.runner import run_once, write_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "config" / "config_single.json"
BIN_PATH = ROOT / "target" / "release" / "chess-corners"
PARAMS_PATH = ROOT / "testdata" / "out" / "parameters.npz"
DEFAULT_OUTDIR = ROOT / "testdata" / "out" / "accuracy"


def image_path(camera: str, index: int) -> Path:
    camdir = "leftcamera" if camera == "l" else "rightcamera"
    suff = "L" if camera == "l" else "R"
    return ROOT / "testdata" / "imgs" / camdir / f"Im_{suff}_{index + 1}.png"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_detector(img_path: Path, base_cfg: dict) -> Tuple[np.ndarray, Dict[str, float]]:
    cfg = dict(base_cfg)
    cfg["image"] = str(img_path)
    cfg["output_json"] = None
    cfg["output_png"] = None

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cfg_path = tmpdir / "config.json"
        write_config(cfg, cfg_path)
        stdout = run_once(ROOT, BIN_PATH, cfg_path, tmpdir)

        out_path = tmpdir / "out.json"
        with out_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

    corners_raw = result.get("corners", [])
    if corners_raw:
        corners = np.array([[c["x"], c["y"]] for c in corners_raw], dtype=float)
    else:
        corners = np.empty((0, 2), dtype=float)
    metrics = parse_trace(stdout)
    return corners, metrics


def plot_corners(
    img: Image.Image,
    gt: np.ndarray,
    detected: np.ndarray,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot ground-truth and detected corners on the image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray")
    ax.scatter(gt[:, 0], gt[:, 1], s=12, c="lime", label="ground truth")
    if detected.size:
        ax.scatter(
            detected[:, 0],
            detected[:, 1],
            s=12,
            c="red",
            marker="x",
            label="chess-corners",
        )
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def nearest_errors(detected: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distance from each detected point to ground truth."""
    if detected.size == 0 or gt.size == 0:
        return np.empty((0,), dtype=float)
    diff = detected[:, None, :] - gt[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists.min(axis=1)


def summarize_errors(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {}
    return {
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
    }


def evaluate_image(
    camera: str,
    idx: int,
    gt: np.ndarray,
    base_cfg: dict,
    *,
    save_plot: Path | None = None,
    show_plot: bool = False,
) -> Tuple[dict, np.ndarray]:
    img_path = image_path(camera, idx)
    img = Image.open(img_path)

    detected, metrics = run_detector(img_path, base_cfg)
    errors = nearest_errors(detected, gt)
    acc = summarize_errors(errors)

    if save_plot or show_plot:
        plot_corners(img, gt, detected, save_path=save_plot, show=show_plot)

    return (
        {
            "camera": camera,
            "index": idx,
            "image": str(img_path.relative_to(ROOT)),
            "ground_truth_points": int(len(gt)),
            "detected_points": int(len(detected)),
            "accuracy_px": acc,
            "trace_ms": metrics,
        },
        errors,
    )


def plot_error_hist(errors: np.ndarray, title: str, path: Path) -> None:
    if errors.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Nearest GT distance (px)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_scatter(errors: np.ndarray, title: str, path: Path) -> None:
    if errors.size == 0:
        return
    x = np.arange(len(errors))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, errors, s=10, alpha=0.7, color="darkorange")
    ax.set_xlabel("Detection index")
    ax.set_ylabel("Nearest GT distance (px)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate_errors(errors: Iterable[np.ndarray]) -> dict:
    collected: List[np.ndarray] = [e for e in errors if e.size]
    if not collected:
        return {}
    combined = np.concatenate(collected)
    return summarize_errors(combined)


def run_batch(
    l_pts: np.ndarray, r_pts: np.ndarray, base_cfg: dict, outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    reports: List[dict] = []
    agg: Dict[str, List[np.ndarray]] = {"l": [], "r": []}

    for camera, pts in (("l", l_pts), ("r", r_pts)):
        for idx, gt in enumerate(pts):
            report, errors = evaluate_image(
                camera,
                idx,
                gt,
                base_cfg,
                save_plot=None,
                show_plot=False,
            )
            reports.append(report)
            agg[camera].append(errors)
            print(
                f"{camera}{idx+1:02d}: gt={report['ground_truth_points']} "
                f"det={report['detected_points']} acc={report['accuracy_px']}"
            )

    summary = {
        "left": aggregate_errors(agg["l"]),
        "right": aggregate_errors(agg["r"]),
    }
    summary["overall"] = aggregate_errors(agg["l"] + agg["r"])

    left_errors = np.concatenate([e for e in agg["l"] if e.size]) if agg["l"] else np.array([])
    right_errors = np.concatenate([e for e in agg["r"] if e.size]) if agg["r"] else np.array([])
    overall_errors = (
        np.concatenate([left_errors, right_errors])
        if left_errors.size or right_errors.size
        else np.array([])
    )

    plots = {
        "error_hist_left": str(outdir / "error_hist_left.png"),
        "error_hist_right": str(outdir / "error_hist_right.png"),
        "error_hist_overall": str(outdir / "error_hist_overall.png"),
        "error_scatter_left": str(outdir / "error_scatter_left.png"),
        "error_scatter_right": str(outdir / "error_scatter_right.png"),
        "error_scatter_overall": str(outdir / "error_scatter_overall.png"),
    }

    plot_error_hist(left_errors, "Left camera error histogram", outdir / "error_hist_left.png")
    plot_error_hist(right_errors, "Right camera error histogram", outdir / "error_hist_right.png")
    plot_error_hist(overall_errors, "Overall error histogram", outdir / "error_hist_overall.png")

    plot_error_scatter(left_errors, "Left camera error scatter", outdir / "error_scatter_left.png")
    plot_error_scatter(right_errors, "Right camera error scatter", outdir / "error_scatter_right.png")
    plot_error_scatter(overall_errors, "Overall error scatter", outdir / "error_scatter_overall.png")

    out = {"summary": summary, "images": reports, "plots": plots}
    report_path = outdir / "accuracy_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote report: {report_path}")


def main():
    """ChESS accuracy benchmark (ground truth vs. chess-corners output)."""
    if not BIN_PATH.exists():
        raise SystemExit(
            f"Missing binary at {BIN_PATH}; run `cargo build -p chess-corners --release` first."
        )

    parser = argparse.ArgumentParser(description="ChESS accuracy benchmark")
    parser.add_argument(
        "camera",
        choices=("l", "r"),
        nargs="?",
        default="l",
        help="Camera side: l or r (ignored in --batch mode).",
    )
    parser.add_argument(
        "-i", "--imgindex", type=int, default=0, help="Image index to load"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all left/right images, save plots and JSON report.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for batch plots/report.",
    )
    args = parser.parse_args()

    data = np.load(PARAMS_PATH)
    l_pts = data["L_Imgpoints"]
    r_pts = data["R_Imgpoints"]

    base_cfg = load_config(BASE_CFG)

    if args.batch:
        run_batch(l_pts, r_pts, base_cfg, args.outdir)
        return

    points = l_pts if args.camera == "l" else r_pts
    if args.imgindex < 0 or args.imgindex >= len(points):
        raise IndexError(
            f"imgindex {args.imgindex} out of range for camera '{args.camera}' with {len(points)} frames"
        )
    gt = points[args.imgindex]

    report, errors = evaluate_image(
        args.camera,
        args.imgindex,
        gt,
        base_cfg,
        save_plot=None,
        show_plot=True,
    )

    print(f"Image: {report['image']}")
    print(
        f"Ground truth points: {report['ground_truth_points']} | "
        f"Detected corners: {report['detected_points']}"
    )
    if report["trace_ms"]:
        print("Trace (ms):", report["trace_ms"])

    if errors.size:
        print("Accuracy (px):", report["accuracy_px"])
    else:
        print("Accuracy (px): no matches to compare.")


if __name__ == "__main__":
    main()
