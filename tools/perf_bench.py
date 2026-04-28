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
"""

from __future__ import annotations

from trace.io import print_results, combo_label
from trace.runner import FeatureCombo, RunResult, run_combo

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "testdata" / "out" / "perf_report.json"

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

def to_json(results: List[RunResult], runs: int) -> Dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "results": [
            {
                "config": r.config_label,
                "features": combo_label(r.combo),
                "image": str(r.image),
                "metrics": r.metrics,
            }
            for r in results
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run chess-corners perf benchmarks.")
    parser.add_argument("--runs", type=int, default=10, help="Runs per image per combo.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--radon",
        action="store_true",
        help=(
            "Also run a Radon-detector config alongside the ChESS multi/single "
            "configs. Reads config/config_radon.json if present, otherwise "
            "patches config_single.json with detector_mode='radon'."
        ),
    )
    args = parser.parse_args()

    config = ROOT / "config" / "config.json"
    single_config = ROOT / "config" / "config_single.json"
    radon_config_path = ROOT / "config" / "config_radon.json"
    runs = args.runs

    base_cfg = load_config(config)
    single_cfg = load_config(single_config)
    images = discover_images(ROOT / "testdata")
    if not images:
        raise SystemExit("No images found under testdata/")

    combos = all_feature_combos()
    all_results: List[RunResult] = []
    configs = [
        (base_cfg, "multi"),
        (single_cfg, "single"),
    ]
    if args.radon:
        if radon_config_path.exists():
            radon_cfg = load_config(radon_config_path)
        else:
            radon_cfg = dict(single_cfg)
            radon_cfg["detector_mode"] = "radon"
            radon_cfg["pyramid_levels"] = 1
        configs.append((radon_cfg, "radon"))

    for cfg, label in configs:
        for combo in combos:
            print(f"=== Config: {label} | Features: {combo_label(combo)} ===")
            results = run_combo(ROOT, combo, images, cfg, runs, label)
            all_results.extend(results)

    print_results(all_results, runs)
    report = to_json(all_results, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote JSON report: {args.out}")

if __name__ == "__main__":
    main()
