#!/usr/bin/env python3
"""Evaluate a trained ML refiner on the hard-cell held-out validation set.

Reports overall mean/p50/p90/p95/p99 radial error in **pixels**, plus
a conditional breakdown by cell size, blur sigma, and noise sigma.
This is the promotion gate for v5 model selection — the shipping
model must meet the thresholds defined in
`docs/proposal-ml-refiner-v3.md` (tightened here to the user's
<0.1 px success bar for the clean regime).

Usage:

    python tools/ml_refiner/eval_hardcell.py \\
        --data-dir tools/ml_refiner/data/val_hardcell_v5 \\
        --checkpoint tools/ml_refiner/runs/<run>/model_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from model import CornerRefinerNet  # noqa: E402


def load_shards(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all shards in `data_dir` and concatenate.

    Returns
    -------
    patches: uint8 (N, P, P)
    dx, dy: float32 (N,)
    blur, noise, cell: float32 (N,)
    """
    shards = sorted(data_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {data_dir}")
    patches = []
    dx_all = []
    dy_all = []
    blur_all = []
    noise_all = []
    cell_all = []
    for path in shards:
        with np.load(path) as data:
            patches.append(data["patches"])
            dx_all.append(np.asarray(data["dx"], dtype=np.float32))
            dy_all.append(np.asarray(data["dy"], dtype=np.float32))
            blur_all.append(np.asarray(data["blur_sigma"], dtype=np.float32))
            noise_all.append(np.asarray(data["noise_sigma"], dtype=np.float32))
            cell = data.get("cell_size_px")
            if cell is None:
                # Legacy tanh shards — no cell_size field. Fill with NaN
                # so the per-cell breakdown is skipped gracefully.
                cell = np.full(dx_all[-1].shape, np.nan, dtype=np.float32)
            else:
                cell = np.asarray(cell, dtype=np.float32)
            cell_all.append(cell)
    return (
        np.concatenate(patches, axis=0),
        np.concatenate(dx_all, axis=0),
        np.concatenate(dy_all, axis=0),
        np.concatenate(blur_all, axis=0),
        np.concatenate(noise_all, axis=0),
        np.concatenate(cell_all, axis=0),
    )


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


def infer_all(model: torch.nn.Module, patches: np.ndarray, device: torch.device, batch: int = 1024) -> np.ndarray:
    """Run the model on every patch. Returns (N, 2) predicted (dx, dy)."""
    model.eval()
    n = patches.shape[0]
    out = np.empty((n, 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(start + batch, n)
            x = patches[start:end].astype(np.float32) / 255.0
            x = torch.from_numpy(x).unsqueeze(1).to(device)  # (B, 1, P, P)
            pred = model(x).detach().cpu().numpy()  # (B, 3)
            out[start:end, 0] = pred[:, 0]
            out[start:end, 1] = pred[:, 1]
    return out


def radial_stats(err: np.ndarray) -> Dict[str, Any]:
    if err.size == 0:
        out: Dict[str, Any] = {"n": 0}
        out.update({k: float("nan") for k in ("mean", "p50", "p90", "p95", "p99", "max")})
        return out
    return {
        "n": int(err.size),
        "mean": float(np.mean(err)),
        "p50": float(np.percentile(err, 50)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(np.max(err)),
    }


def bin_by(
    err: np.ndarray,
    key: np.ndarray,
    edges: List[float],
) -> List[Tuple[str, Dict[str, float]]]:
    """Bucket errors by `key`, report stats per bucket."""
    out: List[Tuple[str, Dict[str, float]]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (key >= lo) & (key <= hi)
            label = f"[{lo:g}, {hi:g}]"
        else:
            mask = (key >= lo) & (key < hi)
            label = f"[{lo:g}, {hi:g})"
        out.append((label, radial_stats(err[mask])))
    return out


def format_table(rows: List[Tuple[str, Dict[str, float]]], key_header: str) -> str:
    header = f"  {key_header:<18}  n       mean    p50     p90     p95     p99     max"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for label, s in rows:
        if s["n"] == 0:
            lines.append(f"  {label:<18}  (empty)")
            continue
        lines.append(
            f"  {label:<18}  {s['n']:<6d}  {s['mean']:6.4f}  {s['p50']:6.4f}  "
            f"{s['p90']:6.4f}  {s['p95']:6.4f}  {s['p99']:6.4f}  {s['max']:6.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Held-out dataset directory")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Enforce the v5 promotion gate (<0.1 px clean mean).",
    )
    parser.add_argument("--json", help="Optional path to dump metrics as JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)

    patches, dx_true, dy_true, blur, noise, cell = load_shards(data_dir)
    print(f"loaded {patches.shape[0]} patches from {data_dir}")

    device = select_device(args.device)
    print(f"device={device}")

    model = CornerRefinerNet()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device)

    pred = infer_all(model, patches, device, batch=args.batch)
    dx_err = pred[:, 0] - dx_true
    dy_err = pred[:, 1] - dy_true
    err = np.sqrt(dx_err * dx_err + dy_err * dy_err)

    overall = radial_stats(err)
    print("\n=== Overall (pixels, Euclidean) ===")
    for k, v in overall.items():
        print(f"  {k:<6}: {v}")

    # Clean subset: no noise, minimal blur. Useful for the <0.1 px gate.
    clean_mask = (noise <= 0.5) & (blur <= 0.7)
    print("\n=== Clean subset (noise≤0.5, blur≤0.7) ===")
    clean = radial_stats(err[clean_mask])
    for k, v in clean.items():
        print(f"  {k:<6}: {v}")

    metrics: Dict[str, Any] = {
        "overall": overall,
        "clean": clean,
    }

    # Per-cell-size breakdown (skipped for legacy tanh shards).
    if not np.all(np.isnan(cell)):
        cell_rows = bin_by(err, cell, [4.0, 5.5, 7.0, 8.5, 10.0, 12.0])
        print("\n=== By cell_size_px ===")
        print(format_table(cell_rows, "cell_size_px"))
        metrics["by_cell_size"] = {label: s for label, s in cell_rows}

    blur_rows = bin_by(err, blur, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    print("\n=== By blur_sigma ===")
    print(format_table(blur_rows, "blur_sigma"))
    metrics["by_blur"] = {label: s for label, s in blur_rows}

    noise_rows = bin_by(err, noise, [0.0, 2.0, 5.0, 8.0, 10.0])
    print("\n=== By noise_sigma ===")
    print(format_table(noise_rows, "noise_sigma"))
    metrics["by_noise"] = {label: s for label, s in noise_rows}

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote metrics to {args.json}")

    if args.gate:
        print("\n=== Promotion gate ===")
        # User's success bar: <0.1 px clean mean. Keep the legacy 0.15
        # fallback as a secondary "progress" bar below 0.1.
        clean_mean = clean["mean"]
        if clean_mean < 0.10:
            print(f"  ✓ clean mean {clean_mean:.4f} < 0.10 px — GATE PASSED")
            sys.exit(0)
        if clean_mean < 0.15:
            print(
                f"  △ clean mean {clean_mean:.4f} in [0.10, 0.15) — PROGRESS, not yet shippable"
            )
            sys.exit(2)
        print(f"  ✗ clean mean {clean_mean:.4f} >= 0.15 px — GATE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
