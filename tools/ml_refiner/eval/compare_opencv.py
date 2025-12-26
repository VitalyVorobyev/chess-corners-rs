#!/usr/bin/env python3
"""Compare ML refiner vs OpenCV cornerSubPix on synthetic patches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from model import CornerRefinerNet  # noqa: E402
from metrics import summarize_binned, summarize_conf, summarize_errors  # noqa: E402
from visualize import plot_binned_metric, plot_cdf, plot_conf_hist  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def select_device(device_cfg: str) -> torch.device:
    device_cfg = device_cfg.lower()
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_cfg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("mps requested but not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_cfg)


def list_shards(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("shard_*.npz"))


def shard_sizes(shards: List[Path]) -> List[int]:
    sizes = []
    for shard in shards:
        with np.load(shard) as data:
            sizes.append(int(data["patches"].shape[0]))
    return sizes


def sample_indices(total: int, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if total <= num_samples:
        return np.arange(total, dtype=np.int64)
    return rng.choice(total, size=num_samples, replace=False).astype(np.int64)


def group_indices(indices: np.ndarray, offsets: np.ndarray) -> Dict[int, np.ndarray]:
    shard_idx = np.searchsorted(offsets, indices, side="right") - 1
    groups: Dict[int, List[int]] = {}
    for idx, sidx in zip(indices, shard_idx):
        local = int(idx - offsets[sidx])
        groups.setdefault(int(sidx), []).append(local)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def load_samples(
    data_dir: Path,
    patch_size: int,
    num_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    shards = list_shards(data_dir)
    if not shards:
        raise FileNotFoundError(f"no shards found in {data_dir}")

    sizes = shard_sizes(shards)
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total = int(offsets[-1])

    rng = np.random.default_rng(seed)
    indices = sample_indices(total, num_samples, rng)
    groups = group_indices(indices, offsets)

    patches_list = []
    dx_list = []
    dy_list = []
    conf_list = []
    is_pos_list = []
    blur_list = []
    noise_list = []
    H_list = []
    has_blur = False
    has_noise = False
    has_H = False

    for shard_idx, shard in enumerate(shards):
        if shard_idx not in groups:
            continue
        local_idx = groups[shard_idx]
        with np.load(shard) as data:
            patches = data["patches"]
            if patches.shape[1:] != (patch_size, patch_size):
                raise ValueError(
                    f"patch size mismatch in {shard}: {patches.shape[1:]}"
                )
            patches_list.append(patches[local_idx])
            dx_list.append(data["dx"][local_idx])
            dy_list.append(data["dy"][local_idx])
            conf_list.append(data["conf"][local_idx])
            if "is_pos" in data:
                is_pos_list.append(data["is_pos"][local_idx])
            else:
                is_pos_list.append((data["conf"][local_idx] > 0.0).astype(np.uint8))
            if "blur_sigma" in data:
                blur_list.append(data["blur_sigma"][local_idx])
                has_blur = True
            if "noise_sigma" in data:
                noise_list.append(data["noise_sigma"][local_idx])
                has_noise = True
            if "H" in data:
                H_list.append(data["H"][local_idx])
                has_H = True

    def _cat(lst, dtype=None):
        if not lst:
            return None
        arr = np.concatenate(lst, axis=0)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    out = {
        "patches": _cat(patches_list),
        "dx": _cat(dx_list, np.float32),
        "dy": _cat(dy_list, np.float32),
        "conf": _cat(conf_list, np.float32),
        "is_pos": _cat(is_pos_list, np.uint8),
    }
    if has_blur:
        out["blur_sigma"] = _cat(blur_list, np.float32)
    if has_noise:
        out["noise_sigma"] = _cat(noise_list, np.float32)
    if has_H:
        out["H"] = _cat(H_list, np.float32)
    return out


def patches_to_uint8(patches: np.ndarray) -> np.ndarray:
    if patches.dtype == np.uint8:
        return patches
    patches = patches.astype(np.float32)
    patches = np.clip(patches, 0.0, 1.0)
    return (patches * 255.0 + 0.5).astype(np.uint8)


def run_opencv(
    patches: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required: pip install opencv-python")

    win = int(cfg.get("win_size", 5))
    zero_zone = int(cfg.get("zero_zone", -1))
    max_iter = int(cfg.get("max_iter", 50))
    eps = float(cfg.get("eps", 1e-4))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    num = patches.shape[0]
    errors = np.full(num, np.nan, dtype=np.float32)
    valid = np.zeros(num, dtype=bool)
    center = (patches.shape[1] - 1) / 2.0

    for i in range(num):
        img = np.ascontiguousarray(patches[i])
        pts = np.array([[[center, center]]], dtype=np.float32)
        cv2.cornerSubPix(
            img,
            pts,
            winSize=(win, win),
            zeroZone=(zero_zone, zero_zone),
            criteria=criteria,
        )
        pt = pts[0, 0]
        if not np.all(np.isfinite(pt)):
            continue
        if pt[0] < 0.0 or pt[0] > img.shape[1] - 1:
            continue
        if pt[1] < 0.0 or pt[1] > img.shape[0] - 1:
            continue
        dx = float(pt[0] - gt_x[i])
        dy = float(pt[1] - gt_y[i])
        errors[i] = math.sqrt(dx * dx + dy * dy)
        valid[i] = True

    return errors, valid


def run_ml(
    patches: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    patches_f = patches.astype(np.float32)
    if patches_f.dtype != np.float32:
        patches_f = patches_f.astype(np.float32)
    if patches.dtype == np.uint8:
        patches_f = patches_f / 255.0
    patches_f = np.clip(patches_f, 0.0, 1.0)

    num = patches_f.shape[0]
    dx_hat = np.empty(num, dtype=np.float32)
    dy_hat = np.empty(num, dtype=np.float32)
    conf_logit = np.empty(num, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, num, batch_size):
            end = min(num, start + batch_size)
            batch = torch.from_numpy(patches_f[start:end]).unsqueeze(1)
            batch = batch.to(device)
            out = model(batch).detach().cpu().numpy()
            dx_hat[start:end] = out[:, 0]
            dy_hat[start:end] = out[:, 1]
            conf_logit[start:end] = out[:, 2]

    conf_pred = 1.0 / (1.0 + np.exp(-conf_logit))
    return dx_hat, dy_hat, conf_pred


def ensure_dir(base: Path, run_id: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    suffix = 1
    while True:
        candidate = base / f"{run_id}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def resolve_run_id(checkpoint: str, override: str | None) -> str:
    if override:
        return override
    try:
        return Path(checkpoint).resolve().parent.name or "run"
    except Exception:
        return "run"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_dir = Path(cfg.get("dataset_dir", "tools/ml_refiner/data/synth_v1"))
    patch_size = int(cfg.get("patch_size", 21))
    num_samples = int(cfg.get("num_samples_eval", 20000))
    seed = int(cfg.get("seed", 0))

    samples = load_samples(data_dir, patch_size, num_samples, seed)
    patches = samples["patches"]
    dx = samples["dx"]
    dy = samples["dy"]
    is_pos = samples["is_pos"].astype(np.float32)

    center = (patch_size - 1) / 2.0
    gt_x = center + dx
    gt_y = center + dy

    pos_mask = is_pos > 0.5
    pos_idx = np.where(pos_mask)[0]

    patches_pos = patches[pos_idx]
    gt_x_pos = gt_x[pos_idx]
    gt_y_pos = gt_y[pos_idx]

    patches_u8 = patches_to_uint8(patches_pos)

    opencv_cfg = cfg.get("opencv", {}) if isinstance(cfg.get("opencv"), dict) else {}
    errors_cv, valid_cv = run_opencv(patches_u8, gt_x_pos, gt_y_pos, opencv_cfg)
    summary_cv = summarize_errors(errors_cv, valid_cv)

    device = select_device(str(cfg.get("device", "auto")))
    ml_cfg = cfg.get("ml", {}) if isinstance(cfg.get("ml"), dict) else {}
    checkpoint = ml_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("ml.checkpoint is required")
    model = CornerRefinerNet()
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    batch_size = int(cfg.get("batch_size", 512))
    dx_hat, dy_hat, conf_pred = run_ml(patches, model, device, batch_size)

    dx_hat_pos = dx_hat[pos_idx]
    dy_hat_pos = dy_hat[pos_idx]
    conf_pred_pos = conf_pred[pos_idx]
    conf_pred_neg = conf_pred[~pos_mask] if np.any(~pos_mask) else np.array([])

    pt_x = center + dx_hat_pos
    pt_y = center + dy_hat_pos
    in_bounds = (pt_x >= 0.0) & (pt_x <= patch_size - 1)
    in_bounds = in_bounds & (pt_y >= 0.0) & (pt_y <= patch_size - 1)
    errors_ml = np.sqrt((pt_x - gt_x_pos) ** 2 + (pt_y - gt_y_pos) ** 2)
    valid_ml = np.isfinite(errors_ml) & in_bounds
    summary_ml = summarize_errors(errors_ml, valid_ml)

    summary = {
        "dataset": {
            "total": int(patches.shape[0]),
            "pos": int(np.sum(pos_mask)),
            "neg": int(np.sum(~pos_mask)),
        },
        "opencv": summary_cv,
        "ml": summary_ml,
        "conf": summarize_conf(conf_pred, is_pos),
    }

    if "blur_sigma" in samples:
        blur = samples["blur_sigma"][pos_idx]
        bins = cfg.get("report", {}).get("blur_bins")
        if bins:
            summary["opencv_blur"] = summarize_binned(blur, errors_cv, valid_cv, bins)
            summary["ml_blur"] = summarize_binned(blur, errors_ml, valid_ml, bins)

    if "noise_sigma" in samples:
        noise = samples["noise_sigma"][pos_idx]
        bins = cfg.get("report", {}).get("noise_bins")
        if bins:
            summary["opencv_noise"] = summarize_binned(noise, errors_cv, valid_cv, bins)
            summary["ml_noise"] = summarize_binned(noise, errors_ml, valid_ml, bins)

    if "H" in samples:
        H = samples["H"][pos_idx]
        severity = np.abs(H[:, 2, 0]) + np.abs(H[:, 2, 1])
        bins = cfg.get("report", {}).get("severity_bins")
        if bins:
            summary["opencv_severity"] = summarize_binned(
                severity, errors_cv, valid_cv, bins
            )
            summary["ml_severity"] = summarize_binned(
                severity, errors_ml, valid_ml, bins
            )

    report_cfg = cfg.get("report", {}) if isinstance(cfg.get("report"), dict) else {}
    out_root = Path(report_cfg.get("out_dir", "tools/ml_refiner/runs/compare_opencv"))
    run_id = resolve_run_id(checkpoint, report_cfg.get("run_name"))
    out_dir = ensure_dir(out_root, run_id)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if report_cfg.get("make_plots", True):
        plot_cdf(
            {
                "opencv": errors_cv,
                "ml": errors_ml,
            },
            str(out_dir / "cdf.png"),
        )
        if "blur_sigma" in samples and report_cfg.get("blur_bins"):
            plot_binned_metric(
                report_cfg["blur_bins"],
                {
                    "opencv": summary.get("opencv_blur", []),
                    "ml": summary.get("ml_blur", []),
                },
                "p50",
                str(out_dir / "error_vs_blur.png"),
                "blur_sigma",
                "p50 error vs blur",
            )
        if "noise_sigma" in samples and report_cfg.get("noise_bins"):
            plot_binned_metric(
                report_cfg["noise_bins"],
                {
                    "opencv": summary.get("opencv_noise", []),
                    "ml": summary.get("ml_noise", []),
                },
                "p50",
                str(out_dir / "error_vs_noise.png"),
                "noise_sigma",
                "p50 error vs noise",
            )
        if "H" in samples and report_cfg.get("severity_bins"):
            plot_binned_metric(
                report_cfg["severity_bins"],
                {
                    "opencv": summary.get("opencv_severity", []),
                    "ml": summary.get("ml_severity", []),
                },
                "p50",
                str(out_dir / "error_vs_severity.png"),
                "projective severity",
                "p50 error vs severity",
            )
        if conf_pred_neg.size > 0:
            plot_conf_hist(
                conf_pred_pos,
                conf_pred_neg,
                str(out_dir / "conf_hist.png"),
            )


if __name__ == "__main__":
    main()
