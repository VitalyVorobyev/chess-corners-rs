"""REPORT.md renderer.

Reads the metrics dumped by ``orientation_bench.sweep`` and renders a
Markdown report. The renderer overwrites only the content between the
``<!-- AUTO-GENERATED BELOW -->`` and ``<!-- AUTO-GENERATED ABOVE -->``
markers, so the manual ``KEEP-MANUAL`` section survives re-runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "REPORT_TEMPLATE",
    "render_report",
    "load_run",
]


AUTO_BEGIN = "<!-- AUTO-GENERATED BELOW: do not edit by hand -->"
AUTO_END = "<!-- AUTO-GENERATED ABOVE -->"
MANUAL_BEGIN = "<!-- KEEP-MANUAL-START -->"
MANUAL_END = "<!-- KEEP-MANUAL-END -->"


REPORT_TEMPLATE = f"""# Two-Axis Orientation Fit — Benchmark Report

{AUTO_BEGIN}
Generated: <ts>  Commit: <git-rev>  Config: <name>

## Setup
This report is auto-rendered. Replace the placeholders by re-running
`python -m orientation_bench report --in <out_dir>`.

## Sweeps
| Sweep | Param | n/cell |
| --- | --- | --- |

## Baseline results (current `fit_two_axes`)
### Per-sweep summary
| Sweep | Param | Bias_a0 deg | RMSE_a0 deg | Bias_a1 deg | RMSE_a1 deg | z_std_a0 | z_std_a1 | failure% | detection% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Sigma calibration summary
- mean(z) across all cells: n/a
- std(z) across all cells: n/a
- |z|>2 fraction: n/a

### Plots
- CDF: plots/error_cdf_blur.png
- Error vs blur: plots/error_vs_blur.png

{AUTO_END}

{MANUAL_BEGIN}
## Recommended changes
_(filled in after baseline run)_

## Open questions
_(append as they arise)_
{MANUAL_END}
"""


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json missing in {run_dir}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    sweeps: dict[str, dict[str, dict[str, Any]]] = {}
    for sweep_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name != "plots"):
        sweep_name = sweep_dir.name
        sweeps[sweep_name] = {}
        for method_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
            metrics_path = method_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                sweeps[sweep_name][method_dir.name] = json.load(f)

    return {"metadata": metadata, "sweeps": sweeps}


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ",".join(_format_value(v) for v in value)
    return str(value)


def _setup_section(metadata: dict[str, Any]) -> str:
    seed = metadata.get("seed", "?")
    config = metadata.get("config_name", "?")
    methods = ", ".join(metadata.get("methods", []))
    return (
        "## Setup\n"
        f"- Seed: {seed}\n"
        f"- Config: {config}\n"
        f"- Methods: {methods}\n"
        f"- Config hash: {metadata.get('config_hash', '?')}\n"
        f"- Git rev: {metadata.get('git_rev', '?')}\n"
    )


def _sweeps_table(metadata: dict[str, Any], sweeps: dict[str, Any]) -> str:
    rows = ["## Sweeps", "| Sweep | Param | n/cell | Cells |", "| --- | --- | --- | --- |"]
    for name, methods in sweeps.items():
        first_method = next(iter(methods.values()), {})
        cells = first_method.get("cells", [])
        param = first_method.get("param", "?")
        n_per_cell = first_method.get("n_per_cell", "?")
        rows.append(f"| {name} | {param} | {n_per_cell} | {len(cells)} |")
    return "\n".join(rows)


def _per_sweep_summary(sweeps: dict[str, Any]) -> str:
    rows = [
        "### Per-sweep summary",
        (
            "| Sweep | Param | Bias_a0 | RMSE_a0 | Bias_a1 | RMSE_a1 | "
            "z_std_a0 | z_std_a1 | failure% | detection% |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, methods in sweeps.items():
        for method_name, method in methods.items():
            cells = method.get("cells", [])
            for cell in cells:
                metrics = cell.get("metrics", {})
                rows.append(
                    "| {sweep}/{method} | {param} | {b0} | {r0} | {b1} | {r1} | "
                    "{z0} | {z1} | {f} | {det} |".format(
                        sweep=name,
                        method=method_name,
                        param=_format_value(cell.get("param_value")),
                        b0=_format_value(metrics.get("bias_axis0_deg")),
                        r0=_format_value(metrics.get("rmse_axis0_deg")),
                        b1=_format_value(metrics.get("bias_axis1_deg")),
                        r1=_format_value(metrics.get("rmse_axis1_deg")),
                        z0=_format_value(metrics.get("z_std_axis0")),
                        z1=_format_value(metrics.get("z_std_axis1")),
                        f=_format_value(metrics.get("failure_pct")),
                        det=_format_value(metrics.get("detection_pct")),
                    )
                )
    return "\n".join(rows)


def _sigma_summary(sweeps: dict[str, Any]) -> str:
    z0_means: list[float] = []
    z1_means: list[float] = []
    z0_stds: list[float] = []
    z1_stds: list[float] = []
    z0_outs: list[float] = []
    z1_outs: list[float] = []
    for methods in sweeps.values():
        for method in methods.values():
            for cell in method.get("cells", []):
                m = cell.get("metrics", {})
                for src, dst in (
                    ("z_mean_axis0", z0_means),
                    ("z_mean_axis1", z1_means),
                    ("z_std_axis0", z0_stds),
                    ("z_std_axis1", z1_stds),
                    ("z_outlier_frac_axis0", z0_outs),
                    ("z_outlier_frac_axis1", z1_outs),
                ):
                    v = m.get(src)
                    if isinstance(v, (int, float)) and v == v:
                        dst.append(float(v))

    def avg(a: list[float]) -> str:
        if not a:
            return "n/a"
        return f"{sum(a) / len(a):.3f}"

    return (
        "### Sigma calibration summary\n"
        f"- mean(z_a0): {avg(z0_means)}, mean(z_a1): {avg(z1_means)}\n"
        f"- std(z_a0): {avg(z0_stds)}, std(z_a1): {avg(z1_stds)}\n"
        f"- frac(|z_a0|>2): {avg(z0_outs)}, frac(|z_a1|>2): {avg(z1_outs)}\n"
    )


def _plots_section(run_dir: Path) -> str:
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        return "### Plots\n_(no plots generated)_"
    items = sorted(p.name for p in plots_dir.iterdir() if p.suffix == ".png")
    if not items:
        return "### Plots\n_(no plots generated)_"
    rows = ["### Plots"] + [f"- plots/{name}" for name in items]
    return "\n".join(rows)


def _build_auto_block(run_dir: Path, payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    sweeps = payload["sweeps"]
    sections = [
        AUTO_BEGIN,
        f"Generated: {metadata.get('timestamp', '?')}  "
        f"Commit: {metadata.get('git_rev', '?')}  "
        f"Config: {metadata.get('config_name', '?')}",
        "",
        _setup_section(metadata),
        "",
        _sweeps_table(metadata, sweeps),
        "",
        "## Baseline results (current `fit_two_axes`)",
        _per_sweep_summary(sweeps),
        "",
        _sigma_summary(sweeps),
        "",
        _plots_section(run_dir),
        "",
        AUTO_END,
    ]
    return "\n".join(sections)


def _extract_manual(existing_text: str) -> str:
    pattern = re.compile(
        rf"{re.escape(MANUAL_BEGIN)}(.*?){re.escape(MANUAL_END)}", re.DOTALL
    )
    match = pattern.search(existing_text)
    if match:
        return f"{MANUAL_BEGIN}{match.group(1)}{MANUAL_END}"
    return (
        f"{MANUAL_BEGIN}\n"
        "## Recommended changes\n"
        "_(filled in after baseline run)_\n\n"
        "## Open questions\n"
        "_(append as they arise)_\n"
        f"{MANUAL_END}"
    )


def render_report(run_dir: str | Path, target_md: str | Path | None = None) -> Path:
    """Render REPORT.md based on the run at ``run_dir``.

    If ``target_md`` is None, writes to ``run_dir / REPORT.md``.
    Preserves any manual section in the existing file at ``target_md``.
    """
    run_dir = Path(run_dir)
    payload = load_run(run_dir)
    if target_md is None:
        target = run_dir / "REPORT.md"
    else:
        target = Path(target_md)

    auto_block = _build_auto_block(run_dir, payload)
    manual_block = REPORT_TEMPLATE  # default
    if target.exists():
        manual_block = _extract_manual(target.read_text(encoding="utf-8"))
    elif (Path(__file__).resolve().parent / "REPORT.md").exists():
        # fall back to the in-tree template
        template_path = Path(__file__).resolve().parent / "REPORT.md"
        manual_block = _extract_manual(template_path.read_text(encoding="utf-8"))
    else:
        manual_block = _extract_manual(REPORT_TEMPLATE)

    text = (
        "# Two-Axis Orientation Fit — Benchmark Report\n\n"
        f"{auto_block}\n\n{manual_block}\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
