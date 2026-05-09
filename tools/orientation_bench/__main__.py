"""CLI entrypoint: ``python -m orientation_bench {sweep|report}``."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchConfig, SweepSpec, config_hash, load_config
from .gt import gt_axes_synthetic
from .match import match_to_gt, pair_axes_array
from .metrics import CellArrays, aggregate_cell, round_floats
from .plot import (
    have_matplotlib,
    plot_error_cdf,
    plot_error_vs_param,
    plot_sigma_calibration,
    plot_subpixel_aliasing,
)
from .report import render_report
from .sweep import (
    PATCH_CENTER,
    ChessSample,
    PatchSample,
    iter_synth_chess_cells,
    iter_synth_patch_cells,
)


logger = logging.getLogger("orientation_bench")


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()[:12]
    except Exception:
        return "unknown"


def _process_patch_sample(
    sample: PatchSample, method: str
) -> tuple[float, float, float, float, float, bool, bool, bool] | None:
    """Run detection, match the central corner, return per-corner data."""
    from .runner import detect_with_method

    corners = detect_with_method(sample.img, method)
    if corners.size == 0:
        return None
    pred_xy = corners[:, :2]
    gt_xy = np.asarray([sample.gt_xy], dtype=np.float64)
    matched_pred, matched_gt = match_to_gt(pred_xy, gt_xy, threshold_px=2.5)
    if matched_pred.size == 0:
        return None
    pi = int(matched_pred[0])
    p1 = float(corners[pi, 5])
    p2 = float(corners[pi, 7])
    s0 = float(corners[pi, 6])
    s1 = float(corners[pi, 8])
    fit_rms = float(corners[pi, 4])
    g1, g2 = sample.gt_angles
    d0, d1, swap = pair_axes_array(
        np.array([p1]), np.array([p2]), np.array([g1]), np.array([g2])
    )
    delta0 = float(d0[0])
    delta1 = float(d1[0])
    swap_flag = bool(swap[0])
    degenerate = (
        not math.isfinite(s0) or not math.isfinite(s1) or s0 <= 0.0 or s1 <= 0.0
    )
    return delta0, delta1, s0, s1, fit_rms, swap_flag, degenerate, True


def _process_chess_sample(
    sample: ChessSample, method: str
) -> dict[str, list[Any]]:
    from .runner import detect_with_method

    out: dict[str, list[Any]] = {
        "delta0": [],
        "delta1": [],
        "sigma0": [],
        "sigma1": [],
        "fit_rms": [],
        "axis_swap_flag": [],
        "degenerate_flag": [],
    }
    corners = detect_with_method(sample.img, method)
    if corners.size == 0:
        return out | {"matched": 0, "gt_total": int(sample.gt_xy.shape[0])}
    matched_pred, matched_gt = match_to_gt(
        corners[:, :2], sample.gt_xy, threshold_px=3.0
    )
    p1 = corners[matched_pred, 5] if matched_pred.size else np.empty((0,))
    p2 = corners[matched_pred, 7] if matched_pred.size else np.empty((0,))
    s0 = corners[matched_pred, 6] if matched_pred.size else np.empty((0,))
    s1 = corners[matched_pred, 8] if matched_pred.size else np.empty((0,))
    fit = corners[matched_pred, 4] if matched_pred.size else np.empty((0,))
    g1 = sample.gt_angles[matched_gt, 0] if matched_gt.size else np.empty((0,))
    g2 = sample.gt_angles[matched_gt, 1] if matched_gt.size else np.empty((0,))
    d0, d1, swap = pair_axes_array(p1, p2, g1, g2)
    # GT angles flagged NaN by gt_axes_from_homography mark a degenerate
    # local Jacobian. Treat those corners as degenerate and drop their
    # delta/swap contributions; they still count toward gt_total /
    # degenerate_frac via the flag.
    gt_nan = ~(np.isfinite(g1) & np.isfinite(g2))
    d0 = np.where(gt_nan, np.nan, d0)
    d1 = np.where(gt_nan, np.nan, d1)
    swap_arr = np.where(gt_nan, False, swap)
    fit_finite = np.where(gt_nan, np.nan, fit)
    sigma0_finite = np.where(gt_nan, np.nan, s0)
    sigma1_finite = np.where(gt_nan, np.nan, s1)
    out["delta0"] = list(map(float, d0))
    out["delta1"] = list(map(float, d1))
    out["sigma0"] = list(map(float, sigma0_finite))
    out["sigma1"] = list(map(float, sigma1_finite))
    out["fit_rms"] = list(map(float, fit_finite))
    out["axis_swap_flag"] = [bool(x) for x in swap_arr]
    degen_from_sigma = [
        not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0)
        for a, b in zip(s0, s1)
    ]
    out["degenerate_flag"] = [
        bool(nan) or bool(deg) for nan, deg in zip(gt_nan.tolist(), degen_from_sigma)
    ]
    return out | {
        "matched": int(matched_pred.size),
        "gt_total": int(sample.gt_xy.shape[0]),
    }


def _run_patch_sweep(
    cfg: BenchConfig,
    sweep: SweepSpec,
    method: str,
    method_dir: Path,
    plots_dir: Path,
    max_cells: int | None,
) -> dict[str, Any]:
    cells_payload: list[dict[str, Any]] = []
    cdf_buffer: list[float] = []
    sigma_buffer_d: list[float] = []
    sigma_buffer_s: list[float] = []
    grid_collector: dict[tuple[float, float], float] = {}

    for cell_idx, params, samples in iter_synth_patch_cells(cfg, sweep, max_cells):
        delta0: list[float] = []
        delta1: list[float] = []
        sigma0: list[float] = []
        sigma1: list[float] = []
        fit_rms: list[float] = []
        swap_flags: list[bool] = []
        degen_flags: list[bool] = []
        matched = 0
        for sample in samples:
            row = _process_patch_sample(sample, method)
            if row is None:
                continue
            d0, d1, s0, s1, fit, sw, deg, _ = row
            delta0.append(d0)
            delta1.append(d1)
            sigma0.append(s0)
            sigma1.append(s1)
            fit_rms.append(fit)
            swap_flags.append(sw)
            degen_flags.append(deg)
            matched += 1
        arrays = CellArrays(
            delta0=np.asarray(delta0),
            delta1=np.asarray(delta1),
            sigma0=np.asarray(sigma0),
            sigma1=np.asarray(sigma1),
            fit_rms=np.asarray(fit_rms),
            axis_swap_flag=np.asarray(swap_flags),
            degenerate_flag=np.asarray(degen_flags),
            matched=matched,
            gt_total=len(samples),
        )
        metrics = aggregate_cell(arrays)
        cdf_buffer.extend(map(float, delta0))
        cdf_buffer.extend(map(float, delta1))
        sigma_buffer_d.extend(map(float, delta0))
        sigma_buffer_d.extend(map(float, delta1))
        sigma_buffer_s.extend(map(float, sigma0))
        sigma_buffer_s.extend(map(float, sigma1))

        if sweep.mode == "patch_grid":
            key = (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
            grid_collector[key] = float(metrics.get("rmse_axis0_deg") or 0.0)

        param_value: Any
        if sweep.mode == "patch_grid":
            param_value = (params.get("dx"), params.get("dy"))
        elif sweep.param == "dx_dy":
            param_value = (params.get("dx"), params.get("dy"))
        else:
            param_value = params.get(sweep.param)

        cells_payload.append(
            {
                "cell_idx": int(cell_idx),
                "param_value": param_value,
                "metrics": round_floats(metrics, digits=6),
            }
        )

    out = {
        "sweep": sweep.name,
        "method": method,
        "param": sweep.param,
        "mode": sweep.mode,
        "n_per_cell": int(cfg.n_per_cell_patch),
        "cells": cells_payload,
    }
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = method_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True, indent=2)

    if cdf_buffer and have_matplotlib():
        plot_error_cdf(
            {method: np.asarray(cdf_buffer)},
            plots_dir / f"error_cdf_{sweep.name}.png",
        )
        plot_error_vs_param(
            cells_payload, sweep.name, "axis0", plots_dir / f"error_vs_{sweep.name}.png"
        )
        plot_sigma_calibration(
            np.asarray(sigma_buffer_d),
            np.asarray(sigma_buffer_s),
            plots_dir / f"sigma_calibration_{sweep.name}.png",
        )
        if sweep.mode == "patch_grid" and grid_collector:
            grid = sweep.extras.get("grid", {})
            sx = int(grid.get("size_x", 0))
            sy = int(grid.get("size_y", 0))
            if sx > 0 and sy > 0:
                arr = np.zeros((sy, sx), dtype=np.float64)
                xs = np.linspace(*grid.get("x_range", [-0.5, 0.5]), sx)
                ys = np.linspace(*grid.get("y_range", [-0.5, 0.5]), sy)
                for j, yy in enumerate(ys):
                    for i, xx in enumerate(xs):
                        arr[j, i] = grid_collector.get(
                            (float(xx), float(yy)), float("nan")
                        )
                plot_subpixel_aliasing(arr, plots_dir / "subpixel_aliasing.png")
    return out


def _run_chess_sweep(
    cfg: BenchConfig,
    sweep: SweepSpec,
    method: str,
    method_dir: Path,
    plots_dir: Path,
    max_cells: int | None,
) -> dict[str, Any]:
    cells_payload: list[dict[str, Any]] = []
    for cell_idx, params, samples in iter_synth_chess_cells(cfg, sweep, max_cells):
        delta0: list[float] = []
        delta1: list[float] = []
        sigma0: list[float] = []
        sigma1: list[float] = []
        fit_rms: list[float] = []
        swap_flags: list[bool] = []
        degen_flags: list[bool] = []
        matched = 0
        gt_total = 0
        for sample in samples:
            row = _process_chess_sample(sample, method)
            delta0.extend(row["delta0"])
            delta1.extend(row["delta1"])
            sigma0.extend(row["sigma0"])
            sigma1.extend(row["sigma1"])
            fit_rms.extend(row["fit_rms"])
            swap_flags.extend(row["axis_swap_flag"])
            degen_flags.extend(row["degenerate_flag"])
            matched += int(row["matched"])
            gt_total += int(row["gt_total"])
        arrays = CellArrays(
            delta0=np.asarray(delta0),
            delta1=np.asarray(delta1),
            sigma0=np.asarray(sigma0),
            sigma1=np.asarray(sigma1),
            fit_rms=np.asarray(fit_rms),
            axis_swap_flag=np.asarray(swap_flags),
            degenerate_flag=np.asarray(degen_flags),
            matched=matched,
            gt_total=gt_total,
        )
        metrics = aggregate_cell(arrays)
        param_value = params.get("pose_yaw"), params.get("pose_pitch")
        cells_payload.append(
            {
                "cell_idx": int(cell_idx),
                "param_value": list(param_value),
                "metrics": round_floats(metrics, digits=6),
            }
        )
    out = {
        "sweep": sweep.name,
        "method": method,
        "param": sweep.param,
        "mode": sweep.mode,
        "n_per_cell": int(cfg.n_per_cell_chess),
        "cells": cells_payload,
    }
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = method_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True, indent=2)
    return out


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_root = Path(args.out)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.timestamp_subdir:
        run_dir = out_root / timestamp
    else:
        run_dir = out_root
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"

    methods = list(args.methods)
    if not methods:
        methods = ["baseline"]

    metadata = {
        "config_name": cfg.name,
        "config_hash": config_hash(cfg),
        "config_path": str(Path(args.config).resolve()),
        "git_rev": _git_rev(),
        "methods": methods,
        "seed": int(cfg.seed),
        "n_per_cell_patch": int(cfg.n_per_cell_patch),
        "n_per_cell_chess": int(cfg.n_per_cell_chess),
        "timestamp": timestamp,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    for method in methods:
        for sweep in cfg.sweeps:
            sweep_dir = run_dir / sweep.name / method
            try:
                if sweep.mode == "chess":
                    _run_chess_sweep(
                        cfg, sweep, method, sweep_dir, plots_dir, args.max_cells
                    )
                else:
                    _run_patch_sweep(
                        cfg, sweep, method, sweep_dir, plots_dir, args.max_cells
                    )
            except Exception as exc:
                logger.exception("sweep %s/%s failed: %s", sweep.name, method, exc)
                # Still write an empty metrics.json so report.py can proceed
                sweep_dir.mkdir(parents=True, exist_ok=True)
                with (sweep_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "sweep": sweep.name,
                            "method": method,
                            "param": sweep.param,
                            "mode": sweep.mode,
                            "error": repr(exc),
                            "cells": [],
                        },
                        f,
                        sort_keys=True,
                        indent=2,
                    )

    # Final report after all sweeps complete.
    try:
        render_report(run_dir)
    except Exception as exc:
        logger.warning("could not render REPORT.md after sweep: %s", exc)
    print(f"Wrote run to {run_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.in_dir)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}", file=sys.stderr)
        return 2
    target_md = run_dir / "REPORT.md"
    if not target_md.exists() and not args.update:
        # bootstrap from in-tree template
        from .report import REPORT_TEMPLATE

        target_md.write_text(REPORT_TEMPLATE, encoding="utf-8")
    out_path = render_report(run_dir, target_md=target_md)
    print(f"Rendered {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orientation_bench",
        description="Two-axis orientation fit benchmark",
    )
    sub = parser.add_subparsers(dest="cmd")

    sweep = sub.add_parser("sweep", help="Run sweeps and write metrics.json files")
    sweep.add_argument("--config", required=True)
    sweep.add_argument(
        "--methods",
        nargs="+",
        default=["baseline"],
        help="Methods to run, e.g. baseline sigma_correction_lut disk_sector_rust disk_sector_py",
    )
    sweep.add_argument(
        "--max-cells", type=int, default=None, help="Cap cells per sweep for fast runs"
    )
    sweep.add_argument(
        "--out",
        default="out/orientation_bench",
        help="Output directory (a timestamped subdir is created within)",
    )
    sweep.add_argument(
        "--no-timestamp-subdir",
        dest="timestamp_subdir",
        action="store_false",
        default=True,
        help="Write directly into --out instead of a timestamped subdir",
    )
    sweep.set_defaults(func=cmd_sweep)

    report = sub.add_parser("report", help="Render REPORT.md")
    report.add_argument("--in", dest="in_dir", required=True)
    report.add_argument(
        "--update",
        action="store_true",
        help="Overwrite the AUTO-GENERATED block; preserve KEEP-MANUAL block",
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("ORIENTATION_BENCH_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
