"""YAML config loader for the orientation benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BenchConfig", "SweepSpec", "load_config", "config_hash"]


@dataclass
class SweepSpec:
    """One sweep block.

    A sweep produces a list of cells. ``param`` names the swept axis
    (informational; used in REPORT.md), ``values`` is the list of
    parameter values, and ``mode`` selects the cell iterator
    (``"patch"`` or ``"chess"``). Optional ``grid`` defines a 2D grid of
    (param_x, param_y) values for the ``"patch_grid"`` mode used by
    subpixel-aliasing sweeps.
    """

    name: str
    mode: str
    param: str
    values: list[Any] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def cell_count(self) -> int:
        if self.mode == "patch_grid":
            grid = self.extras.get("grid", {})
            return int(grid.get("size_x", 0)) * int(grid.get("size_y", 0))
        return len(self.values)


@dataclass
class BenchConfig:
    name: str
    seed: int
    n_per_cell_patch: int
    n_per_cell_chess: int
    defaults: dict[str, Any]
    homography: dict[str, Any]
    chess: dict[str, Any]
    sweeps: list[SweepSpec]
    raw: dict[str, Any]

    def total_cells(self) -> int:
        return sum(s.cell_count() for s in self.sweeps)


def load_config(path: str | Path) -> BenchConfig:
    """Load a bench YAML from `path`."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config at {p} did not parse as a mapping")
    name = str(raw.get("name", p.stem))
    seed = int(raw.get("seed", 0))
    n_patch = int(raw.get("n_per_cell_patch", 100))
    n_chess = int(raw.get("n_per_cell_chess", 5))
    defaults = dict(raw.get("defaults", {}))
    homography = dict(raw.get("homography", {}))
    chess = dict(raw.get("chess", {}))
    sweeps_raw = raw.get("sweeps", [])
    if not isinstance(sweeps_raw, list):
        raise ValueError("`sweeps` must be a list")
    sweeps: list[SweepSpec] = []
    for entry in sweeps_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"sweep entry must be a mapping: {entry!r}")
        sweeps.append(
            SweepSpec(
                name=str(entry["name"]),
                mode=str(entry.get("mode", "patch")),
                param=str(entry.get("param", entry["name"])),
                values=list(entry.get("values", [])),
                extras={
                    k: v
                    for k, v in entry.items()
                    if k not in {"name", "mode", "param", "values"}
                },
            )
        )
    return BenchConfig(
        name=name,
        seed=seed,
        n_per_cell_patch=n_patch,
        n_per_cell_chess=n_chess,
        defaults=defaults,
        homography=homography,
        chess=chess,
        sweeps=sweeps,
        raw=raw,
    )


def config_hash(cfg: BenchConfig) -> str:
    """Stable hash of the raw config for reproducibility metadata."""
    payload = yaml.safe_dump(cfg.raw, sort_keys=True).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=8).hexdigest()
