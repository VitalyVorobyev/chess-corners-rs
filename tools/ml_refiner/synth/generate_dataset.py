#!/usr/bin/env python3
"""Generate synthetic subpixel corner refinement datasets."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from synth import augment, homography, io as synth_io, negatives, render_corner  # noqa: E402


REQUIRED_KEYS = (
    "seed",
    "patch_size",
    "num_samples",
    "shard_size",
    "dx_range",
    "dy_range",
    "rotation",
    "noise_sigma",
    "blur_sigma",
    "contrast",
    "brightness",
    "gamma",
    "conf_params",
)


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")
    return cfg


def _as_range(cfg: Dict[str, Any], key: str) -> Tuple[float, float]:
    value = cfg[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a 2-element list")
    return float(value[0]), float(value[1])


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return commit.decode("utf-8").strip()


def _compute_extent(cfg: Dict[str, Any]) -> float:
    patch_size = int(cfg["patch_size"])
    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")
    max_offset = max(
        abs(dx_range[0]),
        abs(dx_range[1]),
        abs(dy_range[0]),
        abs(dy_range[1]),
    )
    margin = float(cfg.get("render_margin", 2.0))
    extent = (patch_size - 1) / 2.0 + max_offset + margin

    homography_cfg = cfg.get("homography")
    if isinstance(homography_cfg, dict) and homography_cfg.get("enabled", False):
        if "scale_x" in homography_cfg:
            sx_range = _as_range(homography_cfg, "scale_x")
        else:
            sx_range = (1.0, 1.0)
        if "scale_y" in homography_cfg:
            sy_range = _as_range(homography_cfg, "scale_y")
        else:
            sy_range = (1.0, 1.0)
        min_scale = min(sx_range[0], sx_range[1], sy_range[0], sy_range[1])
        if min_scale > 0.0:
            extent *= 1.0 / min_scale
        if "shear_range" in homography_cfg:
            shear_range = _as_range(homography_cfg, "shear_range")
            max_shear = max(abs(shear_range[0]), abs(shear_range[1]))
            extent *= 1.0 + max_shear

    return extent


def _compute_confidence(
    blur_sigma: np.ndarray,
    noise_sigma: np.ndarray,
    conf_params: Dict[str, Any],
    severity: np.ndarray | None = None,
) -> np.ndarray:
    a = float(conf_params.get("a", 0.0))
    b = float(conf_params.get("b", 0.0))
    conf = np.exp(-a * (blur_sigma**2) - b * (noise_sigma**2))
    if severity is not None:
        c = float(conf_params.get("c", 0.0))
        if c != 0.0:
            conf = conf * np.exp(-c * (severity**2))
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _sample_params(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
    include_theta: bool,
) -> Dict[str, np.ndarray]:
    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")
    noise_range = _as_range(cfg, "noise_sigma")
    blur_range = _as_range(cfg, "blur_sigma")
    scale_range = _as_range(cfg, "scale") if "scale" in cfg else (1.0, 1.0)

    params = {
        "dx": rng.uniform(*dx_range, size=count).astype(np.float32),
        "dy": rng.uniform(*dy_range, size=count).astype(np.float32),
        "scale": rng.uniform(*scale_range, size=count).astype(np.float32),
        "noise_sigma": rng.uniform(*noise_range, size=count).astype(np.float32),
        "blur_sigma": rng.uniform(*blur_range, size=count).astype(np.float32),
    }
    if include_theta:
        rot_range = _as_range(cfg, "rotation")
        params["theta"] = rng.uniform(*rot_range, size=count).astype(np.float32)
    return params


def _homography_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("homography", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("homography config must be a mapping")
    merged = dict(raw)
    if "rotation" not in merged and "rotation" in cfg:
        merged["rotation"] = cfg["rotation"]
    return merged


def _neg_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("neg", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("neg config must be a mapping")
    return dict(raw)


def generate_samples(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
    render_x: np.ndarray,
    render_y: np.ndarray,
    patch_x: np.ndarray,
    patch_y: np.ndarray,
    extent: float,
    super_res: int,
) -> Dict[str, np.ndarray]:
    patch_size = int(cfg["patch_size"])
    edge_softness = float(cfg.get("edge_softness", 0.15))

    contrast_range = _as_range(cfg, "contrast")
    brightness_range = _as_range(cfg, "brightness")
    gamma_range = _as_range(cfg, "gamma")

    homography_cfg = _homography_cfg(cfg)
    homography_enabled = bool(homography_cfg.get("enabled", False))
    params = _sample_params(cfg, rng, count, include_theta=not homography_enabled)
    neg_cfg = _neg_cfg(cfg)
    neg_enabled = bool(neg_cfg.get("enabled", False))
    neg_fraction = float(neg_cfg.get("fraction", 0.0)) if neg_enabled else 0.0
    neg_fraction = float(np.clip(neg_fraction, 0.0, 1.0))
    conf_negative = float(neg_cfg.get("conf_negative", 0.0))
    max_outside_frac = float(homography_cfg.get("max_outside_frac", 0.05))
    max_outside_attempts = int(homography_cfg.get("max_outside_attempts", 10))

    is_pos = np.ones(count, dtype=bool)
    if neg_enabled and neg_fraction > 0.0:
        is_pos = rng.uniform(0.0, 1.0, size=count) >= neg_fraction

    patches = np.empty((count, patch_size, patch_size), dtype=np.uint8)
    Hs = np.empty((count, 3, 3), dtype=np.float32) if homography_enabled else None
    severity = np.zeros(count, dtype=np.float32) if homography_enabled else None
    identity = np.eye(3, dtype=np.float32) if homography_enabled else None

    for idx in range(count):
        if is_pos[idx]:
            theta = 0.0 if homography_enabled else float(params["theta"][idx])
            ideal = render_corner.render_ideal_corner_from_grid(
                render_x,
                render_y,
                theta,
                float(params["scale"][idx]),
                edge_softness,
            )

            dx = params["dx"][idx]
            dy = params["dy"][idx]

            if homography_enabled:
                u_shift = patch_x - dx
                v_shift = patch_y - dy
                uv_shift = np.stack((u_shift, v_shift), axis=-1)

                H = None
                xs = ys = None
                outside_frac = 1.0
                for _ in range(max_outside_attempts):
                    H = homography.sample_homography(rng, homography_cfg, patch_size)
                    Hinv = homography.invert_homography(H)
                    coords = homography.apply_homography(Hinv, uv_shift)
                    xs = coords[..., 0]
                    ys = coords[..., 1]
                    outside = (
                        (xs < -extent)
                        | (xs > extent)
                        | (ys < -extent)
                        | (ys > extent)
                    )
                    outside_frac = float(np.mean(outside))
                    if outside_frac <= max_outside_frac:
                        break
                if H is None or xs is None or ys is None:
                    H = np.eye(3, dtype=np.float32)
                    xs = u_shift
                    ys = v_shift
                elif outside_frac > max_outside_frac:
                    H = np.eye(3, dtype=np.float32)
                    xs = u_shift
                    ys = v_shift

                Hs[idx] = H
                if severity is not None:
                    severity[idx] = abs(float(H[2, 0])) + abs(float(H[2, 1]))
                patch = render_corner.sample_from_image(
                    ideal, extent, super_res, xs, ys
                )
            else:
                patch = render_corner.sample_patch_from_image(
                    ideal,
                    extent,
                    super_res,
                    patch_x,
                    patch_y,
                    dx,
                    dy,
                )
        else:
            patch = negatives.generate_negative_patch(rng, patch_size, neg_cfg)
            if Hs is not None and identity is not None:
                Hs[idx] = identity
        patch = augment.apply_blur(patch, float(params["blur_sigma"][idx]))
        patch = augment.apply_photometric(
            patch,
            rng,
            contrast_range,
            brightness_range,
            gamma_range,
        )
        patch = augment.apply_noise(patch, rng, float(params["noise_sigma"][idx]))
        patches[idx] = np.clip(patch * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    conf = _compute_confidence(
        params["blur_sigma"], params["noise_sigma"], cfg["conf_params"], severity
    )
    if neg_enabled:
        conf = conf.copy()
        conf[~is_pos] = conf_negative
        params["dx"][~is_pos] = 0.0
        params["dy"][~is_pos] = 0.0

    return {
        "patches": patches,
        "dx": params["dx"],
        "dy": params["dy"],
        "conf": conf,
        "is_pos": is_pos.astype(np.uint8),
        "noise_sigma": params["noise_sigma"],
        "blur_sigma": params["blur_sigma"],
        **({"H": Hs} if Hs is not None else {}),
    }


def run_self_test(cfg: Dict[str, Any]) -> None:
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required: pip install numpy")
    cfg = dict(cfg)
    cfg["seed"] = 0
    count = 128
    rng = np.random.default_rng(cfg["seed"])

    patch_size = int(cfg["patch_size"])
    extent = _compute_extent(cfg)
    super_res = int(cfg.get("super_res", 4))
    render_x, render_y = render_corner.make_render_grid(super_res, extent)
    patch_x, patch_y = render_corner.make_patch_grid(patch_size)

    data = generate_samples(
        cfg, rng, count, render_x, render_y, patch_x, patch_y, extent, super_res
    )

    patches = data["patches"]
    dx = data["dx"]
    dy = data["dy"]
    conf = data["conf"]

    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")

    assert patches.dtype == np.uint8
    assert patches.shape == (count, patch_size, patch_size)
    assert np.all(patches >= 0) and np.all(patches <= 255)
    assert np.all(dx >= dx_range[0]) and np.all(dx <= dx_range[1])
    assert np.all(dy >= dy_range[0]) and np.all(dy <= dy_range[1])
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)
    if "is_pos" in data:
        is_pos = data["is_pos"].astype(np.uint8)
        assert is_pos.shape == (count,)
        if np.any(is_pos == 0):
            assert np.all(conf[is_pos == 0] == 0.0)
            assert np.all(dx[is_pos == 0] == 0.0)
            assert np.all(dy[is_pos == 0] == 0.0)
    if "H" in data:
        H = data["H"]
        assert H.dtype == np.float32
        assert H.shape == (count, 3, 3)

    print("self-test OK")


def save_preview(
    out_dir: Path,
    patches: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    is_pos: np.ndarray | None = None,
    H: np.ndarray | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping preview")
        return

    count = patches.shape[0]
    grid = int(math.ceil(math.sqrt(count)))
    patch_size = patches.shape[1]
    center = (patch_size - 1) / 2.0

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2.0, grid * 2.0))
    axes = np.array(axes).reshape(-1)

    if H is not None:
        for idx in range(min(3, H.shape[0])):
            print(
                f"preview H[{idx}] p=({H[idx, 2, 0]:.4g}, {H[idx, 2, 1]:.4g})"
            )

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= count:
            continue
        patch = patches[idx]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255, origin="upper")
        sample_is_pos = True
        if is_pos is not None:
            sample_is_pos = bool(is_pos[idx])
        label = "POS" if sample_is_pos else "NEG"
        ax.text(1.0, 2.5, label, color="yellow", fontsize=6, weight="bold")
        if sample_is_pos:
            corner_x = center + float(dx[idx])
            corner_y = center + float(dy[idx])
            ax.plot(
                [corner_x - 1.5, corner_x + 1.5],
                [corner_y, corner_y],
                color="lime",
                linewidth=1.0,
            )
            ax.plot(
                [corner_x, corner_x],
                [corner_y - 1.5, corner_y + 1.5],
                color="lime",
                linewidth=1.0,
            )
            ax.arrow(
                center,
                center,
                float(dx[idx]),
                float(dy[idx]),
                color="red",
                width=0.3,
                head_width=2.0,
                length_includes_head=True,
            )

    out_path = out_dir / "preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", help="Output directory for dataset shards")
    parser.add_argument("--preview", type=int, default=0, help="Save preview grid")
    parser.add_argument("--self-test", action="store_true", help="Run a quick self-test")
    args = parser.parse_args()

    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required: pip install numpy")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.self_test:
        run_self_test(cfg)
        return

    if not args.out:
        parser.error("--out is required unless --self-test is set")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(cfg["patch_size"])
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    extent = _compute_extent(cfg)
    super_res = int(cfg.get("super_res", 4))
    if patch_size <= 0 or super_res <= 0:
        raise ValueError("patch_size and super_res must be positive")

    render_x, render_y = render_corner.make_render_grid(super_res, extent)
    patch_x, patch_y = render_corner.make_patch_grid(patch_size)

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    num_samples = int(cfg["num_samples"])
    shard_size = int(cfg["shard_size"])
    if num_samples <= 0 or shard_size <= 0:
        raise ValueError("num_samples and shard_size must be positive")
    num_shards = int(math.ceil(num_samples / shard_size))

    preview_count = max(0, int(args.preview))
    preview_data = None
    if preview_count > 0:
        preview_rng = np.random.default_rng(seed + 1)
        preview_data = generate_samples(
            cfg,
            preview_rng,
            preview_count,
            render_x,
            render_y,
            patch_x,
            patch_y,
            extent,
            super_res,
        )

    meta = {
        "generator": "ml_refiner_synth_v1",
        "seed": seed,
        "patch_dtype": "uint8",
        "config": cfg,
        "git_commit": _get_git_commit(ML_ROOT.parents[1]),
    }
    synth_io.write_json(out_dir / "meta.json", meta)

    neg_cfg = _neg_cfg(cfg)
    neg_fraction = float(neg_cfg.get("fraction", 0.0)) if neg_cfg.get("enabled") else 0.0
    total_pos = 0
    total_count = 0

    written = 0
    for shard_idx in range(num_shards):
        count = min(shard_size, num_samples - written)
        data = generate_samples(
            cfg,
            rng,
            count,
            render_x,
            render_y,
            patch_x,
            patch_y,
            extent,
            super_res,
        )

        shard_path = out_dir / f"shard_{shard_idx:05d}.npz"
        extra = {
            "noise_sigma": data["noise_sigma"],
            "blur_sigma": data["blur_sigma"],
            "is_pos": data["is_pos"],
        }
        if "H" in data:
            extra["H"] = data["H"]
        synth_io.save_shard(
            shard_path,
            data["patches"],
            data["dx"],
            data["dy"],
            data["conf"],
            extra=extra,
        )
        written += count
        shard_pos = int(np.sum(data["is_pos"]))
        shard_neg = count - shard_pos
        total_pos += shard_pos
        total_count += count
        print(f"wrote {shard_path} ({count} samples)")
        if neg_cfg.get("enabled"):
            neg_frac = shard_neg / max(1, count)
            print(
                f"  pos={shard_pos} neg={shard_neg} neg_frac={neg_frac:.3f}"
            )

    if preview_data is not None:
        save_preview(
            out_dir,
            preview_data["patches"],
            preview_data["dx"],
            preview_data["dy"],
            preview_data.get("is_pos"),
            preview_data.get("H"),
        )

    if neg_cfg.get("enabled"):
        total_neg = total_count - total_pos
        neg_frac = total_neg / max(1, total_count)
        diff = abs(neg_frac - neg_fraction)
        print(
            f"overall neg_frac={neg_frac:.3f} "
            f"(target={neg_fraction:.3f}, diff={diff:.3f})"
        )


if __name__ == "__main__":
    main()
