"""Orientation-fit benchmark for chess-corners.

This package builds, runs, and reports a parametric benchmark for the
two-axis orientation estimator in `chess_corners`. Phase 1 only consumes
the existing baseline (no Rust changes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["bootstrap_paths", "PACKAGE_ROOT", "REPO_ROOT"]


PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parents[1]


def bootstrap_paths() -> None:
    """Add `tools/ml_refiner/synth` to sys.path so its modules are importable.

    The ml_refiner directory is not an installed package, so we extend
    sys.path once at first import. The function is idempotent.
    """
    synth_dir = REPO_ROOT / "tools" / "ml_refiner" / "synth"
    candidate = str(synth_dir)
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path and os.path.isdir(tools_dir):
        sys.path.insert(0, str(tools_dir))


bootstrap_paths()
