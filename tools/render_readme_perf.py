#!/usr/bin/env python3
"""Time the four canonical detector configs on testimages/mid.png.

Runs each of:

- `DetectorConfig.chess()`
- `DetectorConfig.chess_multiscale()`
- `DetectorConfig.radon()`
- `DetectorConfig.radon_multiscale()`

against testimages/mid.png (1024×576), reusing the same `Detector`
instance across iterations so the pyramid / scratch buffers are
amortised. Prints a markdown table to stdout — the README pastes
the result verbatim.

Wall times reflect the Rust-side `Detector.detect()` call only; numpy
view conversion is excluded from the timer. Runs in release mode if
the wheel was built with `maturin develop --release`.

Usage::

    (cd crates/chess-corners-py && maturin develop --release)
    .venv/bin/python tools/render_readme_perf.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import chess_corners as cc

WORKSPACE = Path(__file__).resolve().parents[1]
IMAGE = WORKSPACE / "testimages" / "mid.png"

WARMUP = 5
RUNS = 50


@dataclass
class Case:
    label: str
    factory: Callable[[], cc.DetectorConfig]


# Threshold choices have detector-dependent semantics, so each cell uses
# the operating point that emits the same 77 true X-junctions on
# `testimages/mid.png` with zero false positives. This makes the wall-time
# comparison apples-to-apples (matched precision = recall = 1).
#
# - ChESS responses are dimensionful gray-level sums on the 16-sample ring;
#   they spike sharply at true X-junctions (~600+ here) and stay near
#   single-digit at background noise. A single Threshold::Absolute(100.0)
#   splits cleanly on both single-scale and multiscale.
# - Radon's heatmap is broader because rays integrate through the corner.
#   The multiscale pipeline's cross-level merge shifts the operating point
#   relative to single-scale, so each variant needs its own relative value
#   to land at precision = recall = 1.
CHESS_ABS_THRESHOLD = 100.0
RADON_SINGLE_RELATIVE = 0.28
RADON_MULTI_RELATIVE = 0.34


def _abs(factory: Callable[[], cc.DetectorConfig], v: float) -> Callable[[], cc.DetectorConfig]:
    def make() -> cc.DetectorConfig:
        cfg = factory()
        cfg.threshold = cc.Threshold.absolute(v)
        return cfg
    return make


def _rel(factory: Callable[[], cc.DetectorConfig], v: float) -> Callable[[], cc.DetectorConfig]:
    def make() -> cc.DetectorConfig:
        cfg = factory()
        cfg.threshold = cc.Threshold.relative(v)
        return cfg
    return make


CASES = [
    Case("ChESS — single-scale",     _abs(cc.DetectorConfig.chess,            CHESS_ABS_THRESHOLD)),
    Case("ChESS — 3-level pyramid",  _abs(cc.DetectorConfig.chess_multiscale, CHESS_ABS_THRESHOLD)),
    Case("Radon — single-scale",     _rel(cc.DetectorConfig.radon,            RADON_SINGLE_RELATIVE)),
    Case("Radon — 3-level pyramid",  _rel(cc.DetectorConfig.radon_multiscale, RADON_MULTI_RELATIVE)),
]


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def time_one(detector: cc.Detector, img: np.ndarray) -> float:
    """Return one wall-clock measurement of `detect()` in milliseconds."""
    t0 = time.perf_counter()
    detector.detect(img)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3


def measure(case: Case, img: np.ndarray) -> tuple[int, float, float]:
    detector = cc.Detector(case.factory())
    for _ in range(WARMUP):
        detector.detect(img)
    samples = np.array([time_one(detector, img) for _ in range(RUNS)])
    corners = detector.detect(img)
    return int(corners.shape[0]), float(np.median(samples)), float(np.percentile(samples, 95))


def main() -> None:
    img = load_gray(IMAGE)
    h, w = img.shape
    print(f"# {IMAGE.name} ({w}×{h}), wheel = {cc.__file__}")
    print(f"# warmup={WARMUP}, runs={RUNS}")
    print()
    print("| Config                       | Corners | Median (ms) | p95 (ms) |")
    print("|------------------------------|--------:|------------:|---------:|")
    for case in CASES:
        n, med, p95 = measure(case, img)
        print(f"| {case.label:<28} | {n:>7d} | {med:>11.2f} | {p95:>8.2f} |")


if __name__ == "__main__":
    main()
