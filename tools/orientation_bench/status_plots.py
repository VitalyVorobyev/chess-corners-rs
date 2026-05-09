"""Generate comparative status plots and orientation overlays.

Run from the repository root, for example:

    PYTHONPATH=tools python -m orientation_bench.status_plots \
      --run out/orientation_bench_corrected_gate \
      --config tools/orientation_bench/configs/bench_default.yaml \
      --out out/orientation_bench_status

The plots are diagnostic artefacts. They are intentionally written
under ``out/`` and are not required for normal library use.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import BenchConfig, SweepSpec, load_config
from .match import match_to_gt, pair_axes_array
from .runner import detect_with_method
from .sweep import ChessSample, PatchSample, iter_synth_chess_cells, iter_synth_patch_cells


METHOD_COLORS = {
    "baseline": "#2457a6",
    "sigma_correction_lut": "#1b8a5a",
    "disk_sector_py": "#7a4fb3",
    "disk_sector_rust": "#a05fbf",
    "adaptive_beta": "#bd3f32",
}
GT_AXIS_COLORS = ("#00d0ff", "#ffd54a")
PRED_AXIS_COLORS = ("#ff334e", "#ff9f1c")


@dataclass(frozen=True)
class PatchOverlayCase:
    slug: str
    title: str
    sweep: str
    value: Any
    selection: str = "median"


@dataclass(frozen=True)
class ChessOverlayCase:
    slug: str
    title: str
    value: tuple[float, float]


PATCH_CASES = [
    PatchOverlayCase("nominal_axis90", "Nominal 90 deg", "axis_skew", 90.0),
    PatchOverlayCase("moderate_axis60", "Projective 60 deg", "axis_skew", 60.0),
    PatchOverlayCase("extreme_axis30", "Extreme 30 deg", "axis_skew", 30.0),
    PatchOverlayCase("extreme_axis150", "Extreme 150 deg", "axis_skew", 150.0),
    PatchOverlayCase("sharp_blur0", "Sharp edge", "blur", 0.0),
    PatchOverlayCase("heavy_blur4", "Heavy blur", "blur", 4.0),
    PatchOverlayCase("noise20", "Noise sigma 20", "noise", 20.0),
    PatchOverlayCase("contrast03", "Contrast 0.3", "contrast", 0.3),
]

CHESS_CASES = [
    ChessOverlayCase("pose_0_0", "Chess pose yaw=0 pitch=0", (0.0, 0.0)),
    ChessOverlayCase("pose_45_35", "Chess pose yaw=45 pitch=35", (45.0, 35.0)),
]


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _axis_avg(metrics: dict[str, Any], prefix: str) -> float | None:
    a = _safe_float(metrics.get(f"{prefix}_axis0_deg"))
    b = _safe_float(metrics.get(f"{prefix}_axis1_deg"))
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def _z_avg(metrics: dict[str, Any]) -> float | None:
    a = _safe_float(metrics.get("z_std_axis0"))
    b = _safe_float(metrics.get("z_std_axis1"))
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def _param_value(cell: dict[str, Any]) -> Any:
    value = cell.get("param_value")
    if isinstance(value, list):
        return tuple(value)
    return value


def _x_values(cells: list[dict[str, Any]]) -> tuple[list[float], list[str], bool]:
    values = [_param_value(c) for c in cells]
    xs: list[float] = []
    labels: list[str] = []
    numeric = True
    for i, value in enumerate(values):
        if isinstance(value, tuple):
            numeric = False
            xs.append(float(i))
            labels.append(",".join(f"{float(v):.0f}" for v in value))
            continue
        f = _safe_float(value)
        if f is None:
            numeric = False
            xs.append(float(i))
            labels.append(str(value))
        else:
            xs.append(f)
            labels.append(f"{f:g}")
    return xs, labels, numeric


def load_metrics(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for metrics_path in sorted(run_dir.glob("*/*/metrics.json")):
        sweep = metrics_path.parents[1].name
        method = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8") as f:
            out.setdefault(sweep, {})[method] = json.load(f)
    return out


def plot_comparison_lines(
    metrics_by_sweep: dict[str, dict[str, dict[str, Any]]],
    methods: list[str],
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for sweep, by_method in sorted(metrics_by_sweep.items()):
        if "baseline" not in by_method:
            continue
        base_cells = by_method["baseline"].get("cells", [])
        if not base_cells:
            continue
        xs, labels, numeric = _x_values(base_cells)

        for kind, ylabel, filename_prefix in [
            ("rmse", "axis-average RMSE (deg)", "rmse"),
            ("z", "axis-average z std", "zstd"),
            ("failure", "failure fraction", "failure"),
        ]:
            fig, ax = plt.subplots(figsize=(7.8, 4.6))
            for method in methods:
                payload = by_method.get(method)
                if not payload:
                    continue
                ys: list[float] = []
                for cell in payload.get("cells", []):
                    m = cell.get("metrics", {})
                    if kind == "rmse":
                        y = _axis_avg(m, "rmse")
                    elif kind == "z":
                        y = _z_avg(m)
                    else:
                        y = _safe_float(m.get("failure_pct"))
                    ys.append(float("nan") if y is None else y)
                ax.plot(
                    xs[: len(ys)],
                    ys,
                    marker="o",
                    linewidth=1.8,
                    label=method,
                    color=METHOD_COLORS.get(method),
                )
            if kind == "z":
                ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
            if not numeric:
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_title(f"{sweep}: {ylabel}")
            ax.set_xlabel(by_method["baseline"].get("param", sweep))
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.28)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{filename_prefix}_{sweep}.png", dpi=160)
            plt.close(fig)


def plot_summary_heatmaps(
    metrics_by_sweep: dict[str, dict[str, dict[str, Any]]],
    methods: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    sweeps = sorted(metrics_by_sweep)
    rmse = np.full((len(methods), len(sweeps)), np.nan, dtype=np.float64)
    zstd = np.full_like(rmse, np.nan)
    fail = np.full_like(rmse, np.nan)
    summary: dict[str, Any] = {}

    for mi, method in enumerate(methods):
        rmse_values: list[float] = []
        z_values: list[float] = []
        failure_values: list[float] = []
        max_cell = {"sweep": None, "param": None, "rmse_deg": -1.0}
        for si, sweep in enumerate(sweeps):
            payload = metrics_by_sweep.get(sweep, {}).get(method)
            if not payload:
                continue
            sweep_rmse: list[float] = []
            sweep_z: list[float] = []
            sweep_fail: list[float] = []
            for cell in payload.get("cells", []):
                m = cell.get("metrics", {})
                r = _axis_avg(m, "rmse")
                z = _z_avg(m)
                f = _safe_float(m.get("failure_pct"))
                if r is not None:
                    sweep_rmse.append(r)
                    rmse_values.append(r)
                    if r > float(max_cell["rmse_deg"]):
                        max_cell = {
                            "sweep": sweep,
                            "param": _param_value(cell),
                            "rmse_deg": r,
                        }
                if z is not None:
                    sweep_z.append(z)
                    z_values.append(z)
                if f is not None:
                    sweep_fail.append(f)
                    failure_values.append(f)
            if sweep_rmse:
                rmse[mi, si] = float(np.mean(sweep_rmse))
            if sweep_z:
                zstd[mi, si] = float(np.mean(sweep_z))
            if sweep_fail:
                fail[mi, si] = float(np.max(sweep_fail))
        summary[method] = {
            "mean_rmse_deg": float(np.mean(rmse_values)) if rmse_values else None,
            "median_z_std": float(np.median(z_values)) if z_values else None,
            "mean_z_std": float(np.mean(z_values)) if z_values else None,
            "max_failure_pct": float(np.max(failure_values)) if failure_values else None,
            "worst_rmse_cell": max_cell,
            "n_metric_cells": len(rmse_values),
        }

    def draw_heatmap(arr: np.ndarray, title: str, cbar: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        im = ax.imshow(arr, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xticks(range(len(sweeps)))
        ax.set_xticklabels(sweeps, rotation=35, ha="right")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / filename, dpi=160)
        plt.close(fig)

    draw_heatmap(rmse, "Mean RMSE by sweep and method", "deg", "summary_rmse_heatmap.png")
    draw_heatmap(zstd, "Mean z std by sweep and method", "z std", "summary_zstd_heatmap.png")
    draw_heatmap(fail, "Max failure fraction by sweep and method", "fraction", "summary_failure_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    x = np.arange(len(methods))
    axes[0].bar(x, [summary[m]["mean_rmse_deg"] for m in methods], color=[METHOD_COLORS.get(m) for m in methods])
    axes[0].set_title("Mean RMSE")
    axes[0].set_ylabel("deg")
    axes[1].bar(x, [summary[m]["mean_z_std"] for m in methods], color=[METHOD_COLORS.get(m) for m in methods])
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title("Mean z std")
    axes[2].bar(x, [summary[m]["max_failure_pct"] for m in methods], color=[METHOD_COLORS.get(m) for m in methods])
    axes[2].set_title("Max failure fraction")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "summary_bars.png", dpi=160)
    plt.close(fig)

    return summary


def _wrap_to_half_pi(d: np.ndarray) -> np.ndarray:
    return np.mod(d + math.pi / 2.0, math.pi) - math.pi / 2.0


def _find_sweep(cfg: BenchConfig, name: str) -> SweepSpec:
    for sweep in cfg.sweeps:
        if sweep.name == name:
            return sweep
    raise KeyError(name)


def _cell_matches(case_value: Any, params: dict[str, Any], sweep: SweepSpec) -> bool:
    if sweep.param == "pose":
        value = (params.get("pose_yaw"), params.get("pose_pitch"))
        return tuple(map(float, value)) == tuple(map(float, case_value))
    value = params.get(sweep.param)
    try:
        return abs(float(value) - float(case_value)) < 1e-9
    except (TypeError, ValueError):
        return value == case_value


def _nearest_row(sample: PatchSample, method: str) -> np.ndarray | None:
    corners = detect_with_method(sample.img, method)
    if corners.size == 0:
        return None
    d = np.linalg.norm(corners[:, :2] - np.asarray(sample.gt_xy), axis=1)
    idx = int(np.argmin(d))
    if float(d[idx]) > 3.0:
        return None
    return corners[idx]


def _overlay_methods(methods: list[str]) -> list[str]:
    preferred = ["baseline", "disk_sector_rust", "disk_sector_py", "adaptive_beta"]
    out = [m for m in preferred if m in methods]
    if not out:
        out = [m for m in methods if m != "sigma_correction_lut"]
    return out or list(methods)


def _patch_error(sample: PatchSample, row: np.ndarray) -> float:
    d0, d1, _ = pair_axes_array(
        np.asarray([row[5]]),
        np.asarray([row[7]]),
        np.asarray([sample.gt_angles[0]]),
        np.asarray([sample.gt_angles[1]]),
    )
    return max(abs(float(d0[0])), abs(float(d1[0]))) * 180.0 / math.pi


def select_patch_sample(
    cfg: BenchConfig,
    case: PatchOverlayCase,
    limit: int,
    methods: list[str],
) -> tuple[PatchSample, dict[str, np.ndarray | None], dict[str, float]]:
    sweep = _find_sweep(cfg, case.sweep)
    chosen: list[tuple[float, PatchSample, np.ndarray]] = []
    for _, params, samples in iter_synth_patch_cells(cfg, sweep):
        if not _cell_matches(case.value, params, sweep):
            continue
        for sample in samples[:limit]:
            row = _nearest_row(sample, "baseline")
            if row is None:
                continue
            chosen.append((_patch_error(sample, row), sample, row))
        break
    if not chosen:
        raise RuntimeError(f"no patch sample selected for {case}")
    chosen.sort(key=lambda x: x[0])
    if case.selection == "worst":
        _, sample, base_row = chosen[-1]
    else:
        _, sample, base_row = chosen[len(chosen) // 2]
    rows = {"baseline": base_row}
    for method in dict.fromkeys([*methods, "sigma_correction_lut"]):
        if method == "baseline":
            continue
        rows[method] = _nearest_row(sample, method)
    errors = {
        method: _patch_error(sample, row)
        for method, row in rows.items()
        if row is not None
    }
    return sample, rows, errors


def _axis_segment(x: float, y: float, angle: float, length: float) -> tuple[list[float], list[float]]:
    dx = math.cos(angle) * length
    dy = math.sin(angle) * length
    return [x - dx, x + dx], [y - dy, y + dy]


def _draw_axes(
    ax: Any,
    xy: tuple[float, float],
    angles: tuple[float, float],
    colors: tuple[str, str],
    *,
    length: float,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    for angle, color in zip(angles, colors):
        xs, ys = _axis_segment(float(xy[0]), float(xy[1]), float(angle), length)
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def _patch_title(
    case: PatchOverlayCase,
    method: str,
    sample: PatchSample,
    row: np.ndarray | None,
    sigma_row: np.ndarray | None,
) -> str:
    if row is None:
        return f"{case.title}\n{method}: no detection"
    err = _patch_error(sample, row)
    sig = 0.5 * (float(row[6]) + float(row[8])) * 180.0 / math.pi
    text = f"{case.title}\n{method}: err={err:.1f} deg, sigma={sig:.1f} deg"
    if method == "baseline" and sigma_row is not None:
        sig_lut = 0.5 * (float(sigma_row[6]) + float(sigma_row[8])) * 180.0 / math.pi
        text += f"\nlut sigma={sig_lut:.1f} deg"
    return text


def draw_patch_overlay(
    out_path: Path,
    case: PatchOverlayCase,
    method: str,
    sample: PatchSample,
    row: np.ndarray | None,
    sigma_row: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(sample.img, cmap="gray", vmin=0, vmax=255)
    _draw_axes(ax, sample.gt_xy, sample.gt_angles, GT_AXIS_COLORS, length=10.5, linewidth=2.0, linestyle="--")
    ax.scatter([sample.gt_xy[0]], [sample.gt_xy[1]], s=28, color="white", edgecolor="black", linewidth=0.6)
    if row is not None:
        xy = (float(row[0]), float(row[1]))
        angles = (float(row[5]), float(row[7]))
        _draw_axes(ax, xy, angles, PRED_AXIS_COLORS, length=10.5, linewidth=2.2)
        ax.scatter([xy[0]], [xy[1]], s=34, color="#ff334e", edgecolor="white", linewidth=0.6)
    ax.set_title(_patch_title(case, method, sample, row, sigma_row), fontsize=9)
    ax.set_xlim(5, 35)
    ax.set_ylim(35, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def draw_patch_grid(
    out_path: Path,
    selections: list[tuple[PatchOverlayCase, PatchSample, dict[str, np.ndarray | None]]],
    methods: list[str],
) -> None:
    fig, axes = plt.subplots(len(methods), len(selections), figsize=(3.0 * len(selections), 6.2))
    if len(methods) == 1:
        axes = np.asarray([axes])
    for row_idx, method in enumerate(methods):
        for col_idx, (case, sample, rows) in enumerate(selections):
            ax = axes[row_idx, col_idx]
            ax.imshow(sample.img, cmap="gray", vmin=0, vmax=255)
            _draw_axes(ax, sample.gt_xy, sample.gt_angles, GT_AXIS_COLORS, length=9.5, linewidth=1.2, linestyle="--")
            row = rows.get(method)
            if row is not None:
                _draw_axes(
                    ax,
                    (float(row[0]), float(row[1])),
                    (float(row[5]), float(row[7])),
                    PRED_AXIS_COLORS,
                    length=9.5,
                    linewidth=1.5,
                )
                err = _patch_error(sample, row)
                title = f"{case.title}\n{method} err={err:.1f} deg"
            else:
                title = f"{case.title}\n{method}: no detection"
            ax.set_title(title, fontsize=8)
            ax.set_xlim(5, 35)
            ax.set_ylim(35, 5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Patch overlays: GT dashed cyan/yellow, detected solid red/orange", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def draw_chess_overlay(out_path: Path, case: ChessOverlayCase, sample: ChessSample, method: str) -> None:
    corners = detect_with_method(sample.img, method)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.imshow(sample.img, cmap="gray", vmin=0, vmax=255)
    if corners.size:
        pred_idx, gt_idx = match_to_gt(corners[:, :2], sample.gt_xy, threshold_px=3.0)
        if pred_idx.size:
            p1 = corners[pred_idx, 5]
            p2 = corners[pred_idx, 7]
            g1 = sample.gt_angles[gt_idx, 0]
            g2 = sample.gt_angles[gt_idx, 1]
            d0, d1, _ = pair_axes_array(p1, p2, g1, g2)
            rmse = math.sqrt(float(np.nanmean(np.concatenate([d0, d1]) ** 2))) * 180.0 / math.pi
        else:
            rmse = float("nan")
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            gt_xy = tuple(map(float, sample.gt_xy[gi]))
            pred_xy = (float(corners[pi, 0]), float(corners[pi, 1]))
            gt_angles = tuple(map(float, sample.gt_angles[gi]))
            pred_angles = (float(corners[pi, 5]), float(corners[pi, 7]))
            if not all(math.isfinite(v) for v in gt_angles):
                continue
            _draw_axes(ax, gt_xy, gt_angles, GT_AXIS_COLORS, length=8.0, linewidth=0.9, linestyle="--", alpha=0.9)
            _draw_axes(ax, pred_xy, pred_angles, PRED_AXIS_COLORS, length=8.0, linewidth=1.0, alpha=0.9)
        ax.set_title(f"{case.title} / {method}: matched={len(pred_idx)}/{len(sample.gt_xy)}, RMSE={rmse:.2f} deg")
    else:
        ax.set_title(f"{case.title} / {method}: no detections")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_overlays(
    cfg: BenchConfig,
    out_dir: Path,
    selection_limit: int,
    methods: list[str],
) -> dict[str, Any]:
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    selections: list[tuple[PatchOverlayCase, PatchSample, dict[str, np.ndarray | None]]] = []
    overlay_index: dict[str, Any] = {"patch": [], "chess": []}
    overlay_methods = _overlay_methods(methods)

    for case in PATCH_CASES:
        sample, rows, errors = select_patch_sample(cfg, case, selection_limit, overlay_methods)
        selections.append((case, sample, rows))
        for method in overlay_methods:
            out_path = overlays_dir / f"patch_{case.slug}_{method}.png"
            draw_patch_overlay(
                out_path,
                case,
                method,
                sample,
                rows.get(method),
                rows.get("sigma_correction_lut"),
            )
            overlay_index["patch"].append(
                {
                    "case": case.slug,
                    "method": method,
                    "path": str(out_path),
                    "error_deg": errors.get(method),
                }
            )
    grid_path = overlays_dir / "patch_overlays_grid.png"
    draw_patch_grid(grid_path, selections, overlay_methods)
    overlay_index["patch_grid"] = str(grid_path)

    chess_sweep = _find_sweep(cfg, "chess_pose")
    for chess_case in CHESS_CASES:
        selected_sample = None
        for _, params, samples in iter_synth_chess_cells(cfg, chess_sweep):
            if _cell_matches(chess_case.value, params, chess_sweep):
                selected_sample = samples[0]
                break
        if selected_sample is None:
            continue
        for method in overlay_methods:
            out_path = overlays_dir / f"chess_{chess_case.slug}_{method}.png"
            draw_chess_overlay(out_path, chess_case, selected_sample, method)
            overlay_index["chess"].append(
                {"case": chess_case.slug, "method": method, "path": str(out_path)}
            )

    return overlay_index


def write_markdown_report(
    out_dir: Path,
    run_dir: Path,
    methods: list[str],
    summary: dict[str, Any],
    overlays: dict[str, Any],
) -> None:
    def rel(path: str | Path) -> Path:
        path = Path(path)
        try:
            return path.relative_to(out_dir)
        except ValueError:
            return path.resolve().relative_to(out_dir.resolve())

    lines: list[str] = []
    lines.append("# Orientation Benchmark Status Artefacts")
    lines.append("")
    lines.append(f"Source run: `{run_dir}`")
    lines.append(f"Methods: {', '.join(methods)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | Mean RMSE deg | Mean z std | Median z std | Max failure | Worst RMSE cell |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for method in methods:
        s = summary.get(method, {})
        worst = s.get("worst_rmse_cell") or {}
        worst_text = (
            f"{worst.get('sweep')} / {worst.get('param')} "
            f"({worst.get('rmse_deg'):.2f} deg)"
            if worst.get("sweep")
            else "n/a"
        )
        lines.append(
            "| {m} | {rmse:.3f} | {mean_z:.3f} | {med_z:.3f} | {fail:.3f} | {worst} |".format(
                m=method,
                rmse=s.get("mean_rmse_deg") or float("nan"),
                mean_z=s.get("mean_z_std") or float("nan"),
                med_z=s.get("median_z_std") or float("nan"),
                fail=s.get("max_failure_pct") or 0.0,
                worst=worst_text,
            )
        )
    lines.append("")
    lines.append("## Practical Readout")
    lines.append("")
    lines.append(
        "- `baseline` is already practical for the normal synthetic envelope, including moderate projective "
        "axis skew. It is not a solved estimator for extreme apparent axis angles."
    )
    lines.append(
        "- `sigma_correction_lut` is the practical opt-in for precision reporting: it preserves the baseline "
        "axes and improves z calibration, so it is useful for weighting and filtering downstream."
    )
    if "disk_sector_py" in methods:
        lines.append(
            "- `disk_sector_py` is a benchmark-only full-disk prototype. It may replace the baseline angles "
            "only when its disk/edge confidence gate accepts the local evidence."
        )
    if "disk_sector_rust" in methods:
        lines.append(
            "- `disk_sector_rust` is the opt-in Rust port of the full-disk estimator. It should match the "
            "Python prototype on accuracy while removing Python post-processing cost."
        )
    if "adaptive_beta" in methods:
        lines.append(
            "- `adaptive_beta` is diagnostic only in this run. It improves a few cells but regresses blur, low "
            "contrast, noise, projective skew, and chess-pose cases badly enough that it should not be used."
        )
    lines.append("")
    lines.append("## Current Fail Conditions")
    lines.append("")
    disk = summary.get("disk_sector_rust") or summary.get("disk_sector_py")
    if disk:
        disk_name = "disk_sector_rust" if "disk_sector_rust" in summary else "disk_sector_py"
        worst = disk.get("worst_rmse_cell") or {}
        lines.append(
            f"- The previous baseline extreme-skew failure is resolved in this gate by `{disk_name}`: "
            f"worst cell is {worst.get('sweep')} / {worst.get('param')} "
            f"at {worst.get('rmse_deg'):.2f} deg RMSE."
        )
        if disk_name == "disk_sector_py":
            lines.append(
                "- Remaining risk is operational, not the synthetic accuracy gate: this is a Python prototype "
                "and is too slow to ship as-is. Full-frame runs refine only top-response rows."
            )
        lines.append(
            "- Weak/low-SNR local evidence still falls back to sigma-LUT baseline; real-image validation remains "
            "required before making this the default orientation method."
        )
    else:
        lines.append(
            "- Extreme local projective skew: the 30 deg and 150 deg axis-skew cells dominate worst-case RMSE. "
            "These should be treated as low-confidence unless topology or neighboring geometry validates them."
        )
        lines.append(
            "- Very sharp aliased corners: blur=0 remains materially worse than mild blur because the fixed ring "
            "samples see discretization bias at the edge transition."
        )
        lines.append(
            "- Low-SNR cells: low contrast, high noise, and heavy blur are where unstable extra parameters overfit. "
            "Use corrected sigma, contrast, fit residuals, and border proximity as rejection or downweighting signals."
        )
    lines.append("")
    lines.append("## Core Plots")
    lines.append("")
    for name in [
        "summary_bars.png",
        "summary_rmse_heatmap.png",
        "summary_zstd_heatmap.png",
        "summary_failure_heatmap.png",
    ]:
        lines.append(f"- [plots/{name}](plots/{name})")
    lines.append("")
    lines.append("Per-sweep comparison plots are in `plots/rmse_*.png`, `plots/zstd_*.png`, and `plots/failure_*.png`.")
    lines.append("")
    lines.append("## Overlay Conventions")
    lines.append("")
    lines.append("- GT axes: dashed cyan/yellow.")
    lines.append("- Detected axes: solid red/orange.")
    lines.append("- Coordinates are image pixels. Synthetic patch GT axes use the corrected source-to-image homography Jacobian.")
    lines.append("")
    lines.append("## Overlay Panels")
    lines.append("")
    lines.append(f"- [patch_overlays_grid.png]({rel(overlays['patch_grid'])})")
    for item in overlays.get("patch", []):
        path = rel(item["path"])
        err = item.get("error_deg")
        err_text = f", err={err:.2f} deg" if err is not None else ""
        lines.append(f"- [{item['case']} / {item['method']}]({path}){err_text}")
    for item in overlays.get("chess", []):
        path = rel(item["path"])
        lines.append(f"- [{item['case']} / {item['method']}]({path})")
    lines.append("")
    (out_dir / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Benchmark run directory with metrics.json files")
    parser.add_argument("--config", required=True, help="Benchmark YAML used to reproduce overlays")
    parser.add_argument("--out", required=True, help="Output directory for status artefacts")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "sigma_correction_lut", "adaptive_beta"],
    )
    parser.add_argument("--overlay-selection-limit", type=int, default=120)
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_dir)
    plot_comparison_lines(metrics, args.methods, out_dir)
    summary = plot_summary_heatmaps(metrics, args.methods, out_dir)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    cfg = load_config(args.config)
    overlays = generate_overlays(cfg, out_dir, args.overlay_selection_limit, args.methods)
    with (out_dir / "overlays.json").open("w", encoding="utf-8") as f:
        json.dump(overlays, f, indent=2, sort_keys=True)
    write_markdown_report(out_dir, run_dir, args.methods, summary, overlays)

    print(f"Wrote status artefacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
