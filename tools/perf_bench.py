#!/usr/bin/env python3
"""
Automate performance runs across feature combinations and test images.

For each combination of `simd`, `rayon`, and `par_pyramid`, this script:
- builds the `chess-corners` CLI with tracing enabled,
- runs traced detection on every test image (multi-scale + single-scale),
- repeats each experiment N times and averages the trace metrics per image,
- parses INFO-level JSON trace lines, and
- prints per-image timings (no feature-level aggregation).

Usage:
    python3 tools/perf_bench.py

Optional flags:
    --config PATH           Base config to clone per multiscale run (default: config/config.json)
    --single-config PATH    Config to use for single-scale runs (default: config/config_single.json)
    --images PATH ...       Specific images to test (default: all non-overlay PNGs in testdata/)
    --levels N              Override pyramid levels in the cloned config
    --min-size N            Override min_size in the cloned config
    --runs N                Repeat count per experiment (averaged; default: 3)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "chess-corners"


# Duration helpers ---------------------------------------------------------

_DURATION_RE = re.compile(r"(?P<val>[\d.]+)(?P<unit>[mun]?s)")


def parse_duration_ms(raw: str) -> float:
    """Convert strings like '1.2ms', '42.1µs', '750ns' to milliseconds."""
    clean = raw.replace("µ", "u")
    m = _DURATION_RE.fullmatch(clean)
    if not m:
        raise ValueError(f"Unrecognized duration format: {raw}")
    val = float(m.group("val"))
    unit = m.group("unit")
    if unit == "s":
        return val * 1000.0
    if unit == "ms":
        return val
    if unit == "us":
        return val / 1000.0
    if unit == "ns":
        return val / 1_000_000.0
    raise ValueError(f"Unhandled duration unit in {raw}")


# Trace parsing ------------------------------------------------------------

def parse_trace(stdout: str) -> Dict[str, float]:
    """Extract key timings from the INFO-level JSON trace output."""
    metrics: Dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        busy = evt.get("time.busy")
        if not busy:
            continue
        span = evt.get("span") or {}
        name = span.get("name")
        target = evt.get("target")
        try:
            ms = parse_duration_ms(busy)
        except ValueError:
            continue

        if target == "chess_corners::multiscale":
            if name == "find_chess_corners":
                metrics["levels"] = span.get("levels")
                metrics["min_size"] = span.get("min_size")
                metrics["total_ms"] = ms
            elif name in {"coarse", "refine", "merge"}:
                metrics[f"{name}_ms"] = ms
        elif target == "chess_corners::pyramid" and name == "build_pyramid":
            metrics["pyramid_ms"] = ms
        elif (
            target == "chess_corners_core::descriptor"
            and name == "corners_to_descriptors"
        ):
            metrics["descriptor_ms"] = ms

    return metrics


# Data structures ----------------------------------------------------------

FeatureCombo = Tuple[str, ...]


def combo_label(combo: FeatureCombo) -> str:
    return "none" if not combo else "+".join(combo)


def all_feature_combos() -> List[FeatureCombo]:
    combos: List[FeatureCombo] = []
    flags = ["simd", "rayon", "par_pyramid"]
    for i in range(1 << len(flags)):
        active = tuple(f for bit, f in enumerate(flags) if i & (1 << bit))
        combos.append(active)
    return combos


def discover_images(base: Path) -> List[Path]:
    """Pick only first-level PNGs under `base` (skip nested dirs like testdata/images)."""
    images: List[Path] = []
    for path in base.glob("*.png"):
        name = path.name.lower()
        if "corners" in name or "overlay" in name:
            continue
        images.append(path)
    return sorted(images)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_config(cfg: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


@dataclass
class RunResult:
    combo: FeatureCombo
    image: Path
    config_label: str
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float | None:
        return self.metrics.get("total_ms")


# Runner -------------------------------------------------------------------

def build_binary(features: Sequence[str]) -> None:
    feature_str = ",".join(features)
    cmd = [
        "cargo",
        "build",
        "-p",
        "chess-corners",
        "--release",
        "--no-default-features",
        "--features",
        feature_str,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_once(bin_path: Path, cfg_path: Path, tmpdir: Path) -> str:
    cmd = [
        str(bin_path),
        "run",
        str(cfg_path),
        "--output-json",
        str(tmpdir / "out.json"),
        "--output-png",
        str(tmpdir / "out.png"),
        "--json-trace",
    ]
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stderr:
        print(proc.stderr.strip())
    return proc.stdout


def average_metrics(runs: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across repeated runs, ignoring missing entries per key."""
    keys = {k for r in runs for k in r.keys()}
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def run_combo(
    combo: FeatureCombo,
    images: Sequence[Path],
    base_cfg: Dict,
    overrides: Dict[str, int | float | None],
    runs: int,
    config_label: str,
) -> List[RunResult]:
    features = ["image", "tracing"] + list(combo)
    build_binary(features)

    results: List[RunResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for img in images:
            cfg = dict(base_cfg)
            cfg["image"] = str(img)
            cfg["output_json"] = None
            cfg["output_png"] = None
            if overrides.get("levels") is not None:
                cfg["pyramid_levels"] = overrides["levels"]
            if overrides.get("min_size") is not None:
                cfg["min_size"] = overrides["min_size"]
            cfg_path = tmpdir / f"config_{img.stem}.json"
            write_config(cfg, cfg_path)

            run_metrics: List[Dict[str, float]] = []
            for _ in range(runs):
                stdout = run_once(BINARY, cfg_path, tmpdir)
                run_metrics.append(parse_trace(stdout))
            avg = average_metrics(run_metrics)
            results.append(
                RunResult(combo=combo, image=img, config_label=config_label, metrics=avg)
            )
    return results


# Reporting ----------------------------------------------------------------

def print_results(results: Iterable[RunResult], runs: int) -> None:
    rows = []
    for r in results:
        m = r.metrics
        rows.append(
            (
                r.config_label,
                combo_label(r.combo),
                r.image,
                m.get("levels"),
                m.get("total_ms"),
                m.get("pyramid_ms"),
                m.get("coarse_ms"),
                m.get("refine_ms"),
                m.get("merge_ms"),
                m.get("descriptor_ms"),
            )
        )

    print(f"\nPer-image timings averaged over {runs} run(s) (ms):")
    header = (
        "config",
        "features",
        "image",
        "levels",
        "total",
        "pyramid",
        "coarse",
        "refine",
        "merge",
        "descriptor",
    )
    print(
        "{:<10} {:<18} {:<30} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
            *header
        )
    )
    for row in rows:
        config, combo, img, lvl, tot, pyr, coarse, refine, merge, desc = row
        print(
            "{:<10} {:<18} {:<30} {:>6} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.3f}".format(
                config,
                combo,
                img.name,
                lvl if lvl is not None else "-",
                tot or 0.0,
                pyr or 0.0,
                coarse or 0.0,
                refine or 0.0,
                merge or 0.0,
                desc or 0.0,
            )
        )


# CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run traced perf sweeps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "config.json",
        help="Base config JSON to clone per run",
    )
    parser.add_argument(
        "--single-config",
        type=Path,
        default=ROOT / "config" / "config_single.json",
        help="Config JSON to use for single-scale runs",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        help="Explicit list of images to run. Defaults to all non-overlay PNGs in testdata/",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Override pyramid_levels in the cloned config",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        dest="min_size",
        help="Override min_size in the cloned config",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of times to repeat each experiment (averaged)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    single_cfg = load_config(args.single_config)
    images = args.images or discover_images(ROOT / "testdata")
    if not images:
        raise SystemExit("No images found under testdata/")

    overrides_multi = {"levels": args.levels, "min_size": args.min_size}
    overrides_single: Dict[str, int | float | None] = {}

    combos = all_feature_combos()
    all_results: List[RunResult] = []
    configs = [
        (base_cfg, "multi", overrides_multi),
        (single_cfg, "single", overrides_single),
    ]

    for cfg, label, ov in configs:
        for combo in combos:
            print(f"=== Config: {label} | Features: {combo_label(combo)} ===")
            results = run_combo(combo, images, cfg, ov, args.runs, label)
            all_results.extend(results)

    print_results(all_results, args.runs)


if __name__ == "__main__":
    main()
