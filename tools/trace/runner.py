import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import json

from typing import Dict, List, Sequence, Tuple
from trace.parser import parse_trace

FeatureCombo = Tuple[str, ...]

@dataclass
class RunResult:
    combo: FeatureCombo
    image: Path
    config_label: str
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float | None:
        return self.metrics.get("total_ms")

def build_binary(features: Sequence[str], root: Path) -> None:
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
    subprocess.run(cmd, cwd=root, check=True)

def run_once(root: Path, bin_path: Path, cfg_path: Path, tmpdir: Path) -> str:
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
        cwd=root,
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

def write_config(cfg: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def run_combo(
    root: Path,
    combo: FeatureCombo,
    images: Sequence[Path],
    base_cfg: Dict,
    runs: int,
    config_label: str,
) -> List[RunResult]:
    features = ["image", "tracing"] + list(combo)
    build_binary(features, root)
    BINARY = root / "target" / "release" / "chess-corners"

    results: List[RunResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for img in images:
            cfg = dict(base_cfg)
            cfg["image"] = str(img)
            cfg["output_json"] = None
            cfg["output_png"] = None
            cfg_path = tmpdir / f"config_{img.stem}.json"
            write_config(cfg, cfg_path)

            run_metrics: List[Dict[str, float]] = []
            for _ in range(runs):
                stdout = run_once(root, BINARY, cfg_path, tmpdir)
                run_metrics.append(parse_trace(stdout))
            avg = average_metrics(run_metrics)
            results.append(
                RunResult(combo=combo, image=img, config_label=config_label, metrics=avg)
            )
    return results
