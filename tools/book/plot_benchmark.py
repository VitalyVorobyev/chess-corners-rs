#!/usr/bin/env python3
"""Produce Part V plot figures from the bench_sweep + OpenCV JSON outputs.

Inputs
------
- ``book/src/img/bench/bench_sweep.json``       (from Rust ``bench_sweep`` example)
- ``book/src/img/bench/opencv_subpix_sweep.json`` (from ``opencv_subpix_sweep.py``)

Outputs
-------
- ``book/src/img/bench/accuracy_vs_blur.svg``
- ``book/src/img/bench/accuracy_vs_noise.svg``
- ``book/src/img/bench/accuracy_vs_cell.svg``
- ``book/src/img/bench/error_cdf_clean.svg``
- ``book/src/img/bench/throughput_vs_accuracy.svg``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Consistent colours per refiner — carried across all figures.
COLOURS: Dict[str, str] = {
    "CenterOfMass": "#1f77b4",
    "Forstner": "#ff7f0e",
    "SaddlePoint": "#2ca02c",
    "RadonPeak": "#d62728",
    "ML": "#9467bd",
    "OpenCV_subpix": "#17becf",
}

LABELS: Dict[str, str] = {
    "CenterOfMass": "CenterOfMass",
    "Forstner": "Förstner",
    "SaddlePoint": "SaddlePoint",
    "RadonPeak": "RadonPeak",
    "ML": "ML (ONNX v4)",
    "OpenCV_subpix": "cv2.cornerSubPix",
}

MARKERS: Dict[str, str] = {
    "CenterOfMass": "o",
    "Forstner": "s",
    "SaddlePoint": "^",
    "RadonPeak": "D",
    "ML": "P",
    "OpenCV_subpix": "X",
}


def load_rows(rust_json: Path, opencv_json: Path) -> List[dict]:
    with rust_json.open() as f:
        rust = json.load(f)
    with opencv_json.open() as f:
        cvsp = json.load(f)
    return rust["results"] + cvsp["results"]


def select(rows: List[dict], **kwargs) -> List[dict]:
    def match(row: dict) -> bool:
        for k, v in kwargs.items():
            if not math_eq(row[k], v):
                return False
        return True
    return [r for r in rows if match(r)]


def math_eq(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _setup_axes(ax, xlabel: str, ylabel: str = "Mean error (px)") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_line_sweep(
    rows: List[dict],
    sweep_key: str,
    fixed: Dict,
    title: str,
    xlabel: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    # Gather per-refiner (x, mean, p95) across the sweep.
    series: Dict[str, List[Tuple[float, float, float]]] = {}
    for r in rows:
        if any(not math_eq(r[k], v) for k, v in fixed.items()):
            continue
        series.setdefault(r["refiner"], []).append(
            (float(r[sweep_key]), r["mean"], r["p95"])
        )
    for name in series:
        series[name].sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, pts in series.items():
        xs = [p[0] for p in pts]
        means = [p[1] for p in pts]
        p95s = [p[2] for p in pts]
        colour = COLOURS.get(name, "gray")
        marker = MARKERS.get(name, ".")
        ax.plot(
            xs,
            means,
            marker=marker,
            color=colour,
            label=LABELS.get(name, name),
            linewidth=1.5,
            markersize=5.5,
        )
        ax.fill_between(xs, means, p95s, color=colour, alpha=0.10)

    ax.set_title(title)
    _setup_axes(ax, xlabel)
    if log_y:
        ax.set_yscale("log")
    ax.axhline(0.1, linestyle="--", linewidth=0.8, color="#888")
    ax.text(0.01, 0.105, "0.1 px shipping bar", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#666", ha="left", va="bottom")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_error_cdf(rows: List[dict], fixed: Dict, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    seen: set = set()
    for r in rows:
        if any(not math_eq(r[k], v) for k, v in fixed.items()):
            continue
        if r["refiner"] in seen:
            continue
        seen.add(r["refiner"])
        errs = np.asarray(r["errors"], dtype=np.float64)
        if errs.size == 0:
            continue
        errs_sorted = np.sort(errs)
        cdf = np.arange(1, errs_sorted.size + 1) / errs_sorted.size
        name = r["refiner"]
        ax.plot(
            errs_sorted,
            cdf,
            label=LABELS.get(name, name),
            color=COLOURS.get(name, "gray"),
            marker=MARKERS.get(name, "."),
            markersize=4.5,
            linewidth=1.5,
        )
    ax.set_title(title)
    _setup_axes(ax, "Euclidean error (px)", ylabel="Empirical CDF")
    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_throughput_vs_accuracy(rows: List[dict], fixed: Dict, out_path: Path) -> None:
    pts: Dict[str, Tuple[float, float]] = {}
    for r in rows:
        if any(not math_eq(r[k], v) for k, v in fixed.items()):
            continue
        # First match wins so the point is deterministic when a condition
        # shows up in multiple sweep columns.
        pts.setdefault(r["refiner"], (r["time_us_per_call"], r["mean"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, (x, y) in pts.items():
        ax.scatter(
            x,
            y,
            color=COLOURS.get(name, "gray"),
            marker=MARKERS.get(name, "o"),
            s=80,
            label=LABELS.get(name, name),
            edgecolors="black",
            linewidths=0.6,
        )
        ax.annotate(
            LABELS.get(name, name),
            xy=(x, y),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=8,
            color="#333",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Throughput-accuracy trade-off (clean cell=8)")
    _setup_axes(ax, "Time per refinement (µs, log scale)",
                ylabel="Mean error (px, log scale)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rust-json",
        type=Path,
        default=Path("book/src/img/bench/bench_sweep.json"),
    )
    parser.add_argument(
        "--opencv-json",
        type=Path,
        default=Path("book/src/img/bench/opencv_subpix_sweep.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("book/src/img/bench"),
    )
    args = parser.parse_args()

    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 10.5
    mpl.rcParams["axes.titlesize"] = 11.5
    mpl.rcParams["axes.labelsize"] = 10.5
    mpl.rcParams["savefig.bbox"] = "tight"

    rows = load_rows(args.rust_json, args.opencv_json)
    print(f"loaded {len(rows)} rows")

    plot_line_sweep(
        rows,
        sweep_key="blur_sigma",
        fixed={"cell": 8, "noise_sigma": 0.0},
        title="Accuracy vs Gaussian blur σ (cell=8 px, no noise)",
        xlabel="Blur σ (image pixels)",
        out_path=args.out_dir / "accuracy_vs_blur.svg",
    )
    plot_line_sweep(
        rows,
        sweep_key="noise_sigma",
        fixed={"cell": 8, "blur_sigma": 0.7},
        title="Accuracy vs additive noise σ (cell=8 px, blur σ=0.7)",
        xlabel="Noise σ (gray levels, on 0–255)",
        out_path=args.out_dir / "accuracy_vs_noise.svg",
    )
    plot_line_sweep(
        rows,
        sweep_key="cell",
        fixed={"blur_sigma": 0.0, "noise_sigma": 0.0},
        title="Accuracy vs cell size (clean data)",
        xlabel="Cell size (pixels)",
        out_path=args.out_dir / "accuracy_vs_cell.svg",
        log_y=True,
    )
    plot_error_cdf(
        rows,
        fixed={"cell": 8, "blur_sigma": 0.0, "noise_sigma": 0.0},
        out_path=args.out_dir / "error_cdf_clean.svg",
        title="Error CDF on clean cell=8 fixture",
    )
    plot_throughput_vs_accuracy(
        rows,
        fixed={"cell": 8, "blur_sigma": 0.0, "noise_sigma": 0.0},
        out_path=args.out_dir / "throughput_vs_accuracy.svg",
    )


if __name__ == "__main__":
    main()
