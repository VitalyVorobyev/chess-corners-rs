"""Deterministic tests for the benchmark-only disk-sector estimator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from orientation_bench.config import load_config
from orientation_bench.disk_sector import (
    DiskSectorParams,
    _sobel_gradients,
    disk_sector_refine_row,
    refine_detection_rows,
)
from orientation_bench.match import pair_axes
from orientation_bench.sweep import iter_synth_patch_cells


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "tools" / "orientation_bench" / "configs" / "bench_default.yaml"


def _row(
    xy: tuple[float, float],
    axis0: float,
    axis1: float,
    *,
    contrast: float = 120.0,
    fit_rms: float = 32.0,
    sigma_deg: float = 3.0,
) -> np.ndarray:
    return np.array(
        [
            xy[0],
            xy[1],
            1.0,
            contrast,
            fit_rms,
            axis0,
            math.radians(sigma_deg),
            axis1,
            math.radians(sigma_deg),
        ],
        dtype=np.float32,
    )


def _axis_error_deg(row: np.ndarray, gt: tuple[float, float]) -> float:
    d0, d1, _ = pair_axes(float(row[5]), float(row[7]), float(gt[0]), float(gt[1]))
    return max(abs(d0), abs(d1)) * 180.0 / math.pi


def _patch_sample(sweep_name: str, value: float, sample_idx: int = 0):
    cfg = load_config(CONFIG)
    sweep = next(s for s in cfg.sweeps if s.name == sweep_name)
    for _, params, samples in iter_synth_patch_cells(cfg, sweep):
        if abs(float(params.get(sweep.param)) - float(value)) < 1e-9:
            return samples[sample_idx]
    raise AssertionError(f"missing sample for {sweep_name}={value}")


def test_disk_sector_recovers_extreme_projective_axes_from_bad_seed() -> None:
    sample = _patch_sample("axis_skew", 30.0, sample_idx=0)
    # Deliberately bad orthogonal-ish seed; the disk estimator must use
    # image evidence, not the seed's axis separation.
    bad = _row(
        sample.gt_xy,
        sample.gt_angles[0] + math.radians(100.0),
        sample.gt_angles[1] + math.radians(130.0),
        fit_rms=25.0,
    )
    gx, gy = _sobel_gradients(sample.img)
    refined, debug = disk_sector_refine_row(sample.img, gx, gy, bad)

    assert debug.accepted, debug
    assert _axis_error_deg(refined, sample.gt_angles) < 6.0


def test_disk_sector_keeps_good_blurry_orthogonal_seed() -> None:
    sample = _patch_sample("blur", 4.0, sample_idx=3)
    good = _row(
        sample.gt_xy,
        sample.gt_angles[0],
        sample.gt_angles[1],
        fit_rms=10.0,
        sigma_deg=2.0,
    )
    gx, gy = _sobel_gradients(sample.img)
    refined, debug = disk_sector_refine_row(sample.img, gx, gy, good)

    assert not debug.accepted
    assert np.allclose(refined[[5, 7]], good[[5, 7]])


def test_disk_sector_rejects_near_border_support() -> None:
    img = np.zeros((41, 41), dtype=np.uint8)
    row = _row((1.0, 1.0), 0.0, math.pi / 2.0)
    gx, gy = _sobel_gradients(img)
    refined, debug = disk_sector_refine_row(img, gx, gy, row)

    assert not debug.accepted
    assert debug.reason == "insufficient_support"
    assert np.allclose(refined, row)


def test_refine_detection_rows_preserves_shape_and_frame() -> None:
    sample = _patch_sample("axis_skew", 90.0, sample_idx=0)
    rows = np.stack(
        [
            _row(
                sample.gt_xy,
                sample.gt_angles[0],
                sample.gt_angles[1],
                fit_rms=8.0,
            )
        ]
    )
    refined = refine_detection_rows(sample.img, rows, params=DiskSectorParams())

    assert refined.shape == rows.shape
    assert np.allclose(refined[:, :2], rows[:, :2])
    assert np.all(np.isfinite(refined[:, 5:9]))
