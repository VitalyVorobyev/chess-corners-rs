#!/usr/bin/env python3
"""Regenerate the ``book/src/img/{small,mid,large}_chess.png`` overlays
from ``testimages/{small,mid,large}.png`` with a user-selected absolute
ChESS threshold.

Run from the workspace root:

    python tools/render_book_overlays.py --threshold 5.0

The script:

1. Builds the ``chess-corners`` CLI in release mode if it is missing.
2. For each of the three test images, writes a temporary
   ``DetectionConfig`` JSON and invokes the CLI with
   ``threshold_mode=absolute`` and the chosen ``threshold_value``.
3. Renders each detection JSON with ``tools/plot_output.py`` and saves
   the overlay under ``book/src/img/``.

No pyramid: ``pyramid_levels=1`` (single-scale) — matches the paper's
contract and avoids the coarse-to-fine seed loss observed on
low-contrast scenes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
TESTIMAGES = WORKSPACE / "testimages"
BOOK_IMG = WORKSPACE / "book" / "src" / "img"
CLI = WORKSPACE / "target" / "release" / "chess-corners"
PLOT = WORKSPACE / "tools" / "plot_output.py"

IMAGES = [
    ("small", "small_chess.png"),
    ("mid", "mid_chess.png"),
    ("large", "large_chess.png"),
]


def build_cli() -> None:
    if CLI.exists():
        return
    print("[build] chess-corners CLI not found; building release binary...")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "chess-corners",
            "--features",
            "image cli",
        ],
        cwd=WORKSPACE,
        check=True,
    )


def detection_config(image_path: Path, threshold: float, output_json: Path) -> dict:
    return {
        "image": str(image_path),
        "detector_mode": "canonical",
        "descriptor_mode": "follow_detector",
        "threshold_mode": "absolute",
        "threshold_value": float(threshold),
        "nms_radius": 2,
        "min_cluster_size": 2,
        "refiner": {"kind": "center_of_mass"},
        "pyramid_levels": 1,
        "pyramid_min_size": 128,
        "refinement_radius": 3,
        "merge_radius": 3.0,
        "upscale": {"mode": "disabled"},
        "output_json": str(output_json),
        "output_png": None,
        "log_level": "warn",
    }


def run_one(stem: str, overlay_name: str, threshold: float, tmp_dir: Path) -> int:
    image_path = TESTIMAGES / f"{stem}.png"
    if not image_path.exists():
        raise SystemExit(f"missing test image: {image_path}")

    cfg_path = tmp_dir / f"{stem}.json"
    det_path = tmp_dir / f"{stem}.corners.json"
    overlay_path = BOOK_IMG / overlay_name

    cfg_path.write_text(
        json.dumps(detection_config(image_path, threshold, det_path), indent=2)
    )

    subprocess.run([str(CLI), "run", str(cfg_path)], cwd=WORKSPACE, check=True)

    count = len(json.loads(det_path.read_text()).get("corners", []))

    python = os.environ.get("PYTHON", sys.executable)
    subprocess.run(
        [
            python,
            str(PLOT),
            str(det_path),
            "--image",
            str(image_path),
            "--out",
            str(overlay_path),
            "--no-show",
        ],
        cwd=WORKSPACE,
        check=True,
    )

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Absolute ChESS threshold. 0.0 = paper's contract (any R > 0). "
        "Larger values suppress sensor-noise detections in flat regions.",
    )
    parser.add_argument(
        "--tmp",
        type=Path,
        default=Path("/tmp/chess-corners-overlays"),
        help="Scratch directory for CLI configs + detection JSONs.",
    )
    args = parser.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    BOOK_IMG.mkdir(parents=True, exist_ok=True)
    build_cli()

    print(f"[render] absolute threshold = {args.threshold}")
    for stem, overlay_name in IMAGES:
        count = run_one(stem, overlay_name, args.threshold, args.tmp)
        print(f"  {stem}.png -> {count} corners -> {BOOK_IMG / overlay_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
