#!/usr/bin/env python3
"""Export the ML refiner model to ONNX and generate parity fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import onnx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    onnx = None

ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from model import CornerRefinerNet  # noqa: E402
from synth import render_corner  # noqa: E402
from synth import generate_dataset as synth_gen  # noqa: E402


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return commit.decode("utf-8").strip()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unexpected checkpoint format")


def export_onnx(
    model: torch.nn.Module,
    out_path: Path,
    patch_size: int,
    opset: int,
) -> None:
    if onnx is None:  # pragma: no cover
        raise RuntimeError("onnx is required: pip install onnx")

    dummy = torch.zeros(1, 1, patch_size, patch_size, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["patches"],
        output_names=["pred"],
        dynamic_axes={"patches": {0: "batch"}, "pred": {0: "batch"}},
        do_constant_folding=True,
        opset_version=opset,
    )

    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    actual_opset = None
    if model_onnx.opset_import:
        actual_opset = model_onnx.opset_import[0].version
    if actual_opset is not None and actual_opset != opset:
        print(f"warning: exported opset {actual_opset} differs from requested {opset}")

    size_kb = out_path.stat().st_size / 1024.0
    print(f"wrote {out_path} ({size_kb:.1f} KB)")


def _normalize_patches(patches: np.ndarray) -> np.ndarray:
    if patches.dtype == np.uint8:
        patches = patches.astype(np.float32) / 255.0
    else:
        patches = patches.astype(np.float32)
    return np.clip(patches, 0.0, 1.0)


def load_dataset_patches(
    data_dir: Path, patch_size: int, num: int, seed: int
) -> np.ndarray:
    shards = sorted(data_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards found in {data_dir}")

    rng = np.random.default_rng(seed)
    patches_out: List[np.ndarray] = []

    for shard in shards:
        with np.load(shard) as data:
            patches = data["patches"]
            if patches.shape[1:] != (patch_size, patch_size):
                raise ValueError(
                    f"patch size mismatch in {shard}: {patches.shape[1:]}"
                )
            patches_out.append(patches)

    patches_all = np.concatenate(patches_out, axis=0)
    if patches_all.shape[0] > num:
        idx = rng.choice(patches_all.shape[0], size=num, replace=False)
        patches_all = patches_all[idx]
    return patches_all


def synth_patches(patch_size: int, num: int, seed: int) -> np.ndarray:
    cfg = {
        "seed": seed,
        "patch_size": patch_size,
        "num_samples": num,
        "shard_size": num,
        "dx_range": [-0.5, 0.5],
        "dy_range": [-0.5, 0.5],
        "rotation": [0.0, 6.2831853],
        "scale": [0.9, 1.1],
        "noise_sigma": [0.0, 5.0],
        "blur_sigma": [0.0, 1.0],
        "contrast": [0.9, 1.1],
        "brightness": [-5.0, 5.0],
        "gamma": [0.9, 1.1],
        "conf_params": {"a": 0.6, "b": 0.02},
        "homography": {"enabled": False},
        "neg": {"enabled": False},
    }

    rng = np.random.default_rng(seed)
    extent = synth_gen._compute_extent(cfg)
    super_res = int(cfg.get("super_res", 4))
    render_x, render_y = render_corner.make_render_grid(super_res, extent)
    patch_x, patch_y = render_corner.make_patch_grid(patch_size)
    data = synth_gen.generate_samples(
        cfg, rng, num, render_x, render_y, patch_x, patch_y, extent, super_res
    )
    return data["patches"]


def make_fixtures(
    model: torch.nn.Module,
    patch_size: int,
    num: int,
    seed: int,
    out_dir: Path,
    dataset_dir: Path | None = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset_dir is not None:
        patches = load_dataset_patches(dataset_dir, patch_size, num, seed)
        source = str(dataset_dir)
    else:
        patches = synth_patches(patch_size, num, seed)
        source = "synth"

    patches_f = _normalize_patches(patches)
    patches_f = patches_f[:, None, :, :].astype(np.float32)

    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(patches_f)
        outputs = model(inputs).cpu().numpy().astype(np.float32)

    np.save(out_dir / "patches.npy", patches_f)
    np.save(out_dir / "torch_out.npy", outputs)

    meta = {
        "patch_size": patch_size,
        "num_fixtures": int(patches_f.shape[0]),
        "input": {
            "dtype": "float32",
            "shape": [int(patches_f.shape[0]), 1, patch_size, patch_size],
            "normalization": "uint8->float32/255, clamp to [0,1]",
        },
        "output": {
            "dtype": "float32",
            "shape": [int(patches_f.shape[0]), 3],
            "semantics": ["dx", "dy", "conf_logit"],
        },
        "seed": seed,
        "source": source,
        "git_commit": _get_git_commit(ML_ROOT.parents[1]),
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-fixtures", type=int, default=64)
    parser.add_argument("--fixtures-out", required=True, help="Fixtures output dir")
    parser.add_argument("--dataset-dir", help="Optional dataset dir")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out)
    fixtures_out = Path(args.fixtures_out)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    model = CornerRefinerNet()
    state = load_checkpoint(checkpoint)
    model.load_state_dict(state)
    model.eval()
    model.cpu()

    export_onnx(model, out_path, args.patch_size, args.opset)

    meta = make_fixtures(
        model,
        args.patch_size,
        args.num_fixtures,
        args.seed,
        fixtures_out,
        dataset_dir,
    )

    print("wrote fixtures:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
