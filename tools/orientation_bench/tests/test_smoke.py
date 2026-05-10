"""End-to-end smoke test for the bench CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CONFIG = REPO_ROOT / "tools" / "orientation_bench" / "configs" / "bench_smoke.yaml"


def _chess_corners_available() -> bool:
    try:
        import chess_corners  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _chess_corners_available(),
    reason=(
        "chess_corners is not importable; build it with `maturin develop -m "
        "crates/chess-corners-py/pyproject.toml --release` to enable the smoke test."
    ),
)
def test_smoke_sweep_then_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench_smoke"
    cmd = [
        sys.executable,
        "-m",
        "orientation_bench",
        "sweep",
        "--config",
        str(SMOKE_CONFIG),
        "--methods",
        "ring_fit",
        "--max-cells",
        "2",
        "--out",
        str(out_dir),
        "--no-timestamp-subdir",
    ]
    env = {**__import__("os").environ}
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "tools"))
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"sweep failed: {result.stderr}"

    metadata_path = out_dir / "metadata.json"
    assert metadata_path.exists(), "metadata.json missing"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["seed"] == 1
    assert "ring_fit" in metadata["methods"]

    # At least one sweep should have produced metrics.json
    metrics_files = list(out_dir.rglob("metrics.json"))
    assert metrics_files, "no metrics.json files produced"
    found_nonempty = False
    for mf in metrics_files:
        with mf.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cells = payload.get("cells", [])
        for cell in cells:
            metrics = cell.get("metrics", {})
            bias = metrics.get("bias_axis0_deg")
            if bias is not None and bias == bias:  # not NaN
                found_nonempty = True
                break
        if found_nonempty:
            break
    assert found_nonempty, "no cell produced a finite bias"

    # Run report
    report_cmd = [
        sys.executable,
        "-m",
        "orientation_bench",
        "report",
        "--in",
        str(out_dir),
    ]
    rep = subprocess.run(
        report_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert rep.returncode == 0, f"report failed: {rep.stderr}"
    assert (out_dir / "REPORT.md").exists(), "REPORT.md not generated"
