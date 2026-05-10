"""Per-cell iterators for the orientation benchmark sweeps.

Two cell modes:

* ``patch``: render a 41x41 zero-padded grayscale patch with one
  central corner. Each cell yields ``n_per_cell_patch`` patches with
  GT center pixel and GT axis angles.
* ``chess``: invoke ``synthimages.generate_one`` to produce full
  chessboard images; per-image meta provides ``H_tex2img`` and
  ``gt_corners_uv_tex`` so we can compute GT axes via
  ``gt.gt_axes_from_homography``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from . import bootstrap_paths
from .config import BenchConfig, SweepSpec
from .gt import gt_axes_from_homography, gt_axes_synthetic

bootstrap_paths()

# Imports from sibling tool packages — sys.path was extended in __init__.py.
from render_corner import (  # type: ignore  # noqa: E402
    make_patch_grid,
    make_render_grid,
    render_corner_pattern_from_grid,
    sample_patch_from_image,
)
from homography import sample_homography  # type: ignore  # noqa: E402
from augment import apply_blur  # type: ignore  # noqa: E402

__all__ = [
    "PatchSample",
    "ChessSample",
    "iter_synth_patch_cells",
    "iter_synth_chess_cells",
    "cell_rng",
]


PATCH_SIZE = 41
PATCH_CENTER = (PATCH_SIZE - 1) / 2.0  # 20.0
SUPER_RES = 8
EXTENT = 11.0  # >= 20 px patch radius / super_res, with margin
DEFAULT_SCALE = 1.0
DEFAULT_EDGE_SOFTNESS = 0.15
DEFAULT_CELL_SIZE = 6.0


def cell_rng(seed_str: str, sweep_name: str, cell_idx: int) -> np.random.Generator:
    """Per-cell deterministic RNG."""
    digest = hashlib.blake2s(
        f"{seed_str}|{sweep_name}|{cell_idx}".encode("utf-8"), digest_size=8
    ).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed)


@dataclass
class PatchSample:
    img: np.ndarray  # (41, 41) uint8
    gt_xy: tuple[float, float]  # corner pixel
    gt_angles: tuple[float, float]  # (theta1, theta2) mod pi
    H: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChessSample:
    img: np.ndarray  # (h, w) uint8
    gt_xy: np.ndarray  # (n, 2) float
    gt_angles: np.ndarray  # (n, 2) float, mod pi
    H_tex2img: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _resolve_param(defaults: dict[str, Any], spec: SweepSpec, value: Any) -> dict[str, Any]:
    """Apply the swept value to a fresh copy of `defaults`."""
    out = dict(defaults)
    out[spec.param] = value
    return out


def _render_corner_image(
    rng: np.random.Generator,
    theta_render: float,
    render_mode: str,
    scale: float,
    edge_softness: float,
    cell_size_px: float,
) -> np.ndarray:
    x_grid, y_grid = make_render_grid(SUPER_RES, EXTENT)
    return render_corner_pattern_from_grid(
        x_grid,
        y_grid,
        theta=float(theta_render),
        scale=float(scale),
        edge_softness=float(edge_softness),
        cell_size_px=float(cell_size_px),
        render_mode=render_mode,
    )


def _sample_homography(
    rng: np.random.Generator,
    spec_block: dict[str, Any],
) -> np.ndarray:
    if not spec_block or not spec_block.get("enabled", False):
        return np.eye(3, dtype=np.float32)
    return sample_homography(rng, spec_block, patch_size=PATCH_SIZE, center=(0.0, 0.0))


def _apply_photometric_uint8(
    patch01: np.ndarray,
    rng: np.random.Generator,
    contrast: float,
    brightness: float,
    gamma: float,
) -> np.ndarray:
    """Apply contrast/brightness/gamma in [0, 1] then return uint8."""
    out = np.clip(patch01, 0.0, 1.0)
    if gamma != 1.0:
        out = np.clip(out, 1e-6, 1.0) ** float(gamma)
    if contrast != 1.0:
        out = (out - 0.5) * float(contrast) + 0.5
    if brightness != 0.0:
        out = out + (float(brightness) / 255.0)
    out = np.clip(out * 255.0, 0.0, 255.0)
    return out.astype(np.uint8)


def _apply_vignetting(
    patch_u8: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return patch_u8
    h, w = patch_u8.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / max(w, 1)
    dy = (yy - cy) / max(h, 1)
    r2 = dx * dx + dy * dy
    mask = np.clip(1.0 - float(strength) * r2, 0.0, 1.0)
    out = patch_u8.astype(np.float32) * mask
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_noise_uint8(
    patch_u8: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
    sp_prob: float,
) -> np.ndarray:
    out = patch_u8.astype(np.float32)
    if noise_sigma > 0.0:
        out = out + rng.normal(0.0, float(noise_sigma), size=out.shape).astype(np.float32)
    if sp_prob > 0.0:
        m = rng.random(size=out.shape)
        out[m < sp_prob * 0.5] = 0.0
        out[m > 1.0 - sp_prob * 0.5] = 255.0
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _make_patch(
    rng: np.random.Generator,
    params: dict[str, Any],
) -> PatchSample:
    """Render one synthetic patch with the given parameter dict."""
    render_mode = str(params.get("render_mode", "tanh"))
    # GT contract: returned axes are at theta and theta+pi/2. The renderer's
    # internal rotation is the inverse so the rendered grid axes line up with
    # the contract.
    theta_gt = float(params.get("theta", rng.uniform(0.0, math.pi)))
    skew_deg = float(params.get("axis_skew_deg", 90.0))
    skew_rad = math.radians(skew_deg)

    scale = float(params.get("scale", DEFAULT_SCALE))
    edge_softness = float(params.get("edge_softness", DEFAULT_EDGE_SOFTNESS))
    cell_size_px = float(params.get("cell_size_px", DEFAULT_CELL_SIZE))

    # The two-axis fit fundamentally assumes orthogonal axes. We render the
    # base orthogonal corner with theta_gt, then if the user requested a
    # non-90-degree skew, encode the skew via a shear homography post-render.
    theta_render = -theta_gt  # see notes in gt.py header
    img_high = _render_corner_image(
        rng, theta_render, render_mode, scale, edge_softness, cell_size_px
    )

    # Build optional homography (bench config block). Skew sweep stacks a
    # shear on top.
    H = _sample_homography(rng, params.get("homography", {}))
    if abs(skew_deg - 90.0) > 1e-6:
        # One-sided shear so that e_1 = (1, 0) maps to (1, 0) and
        # e_2 = (0, 1) maps to (k, 1), with k = cot(skew_deg). The angle
        # between the projected axes is then arctan(1/k) = skew_deg.
        # The earlier symmetric shear (s on both off-diagonals) realised
        # 90 - 2*arctan(s) instead, which differs from the requested
        # skew_deg outside the small-angle regime.
        k = math.tan(math.radians(90.0 - skew_deg))  # cot(skew_deg)
        H_shear = np.array(
            [[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        H = (H_shear @ H).astype(np.float32)

    # Sample 41x41 patch grid. The corner sits at the geometric center.
    x_grid, y_grid = make_patch_grid(PATCH_SIZE)
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))

    if not np.allclose(H, np.eye(3, dtype=np.float32)):
        # H is the source/patch-coordinate -> image-coordinate map used by
        # the GT Jacobian. Rendering samples the source texture for each image
        # pixel, so the actual sampler must use H^{-1}.
        try:
            H_sample = np.linalg.inv(H).astype(np.float32)
        except np.linalg.LinAlgError:
            H_sample = np.eye(3, dtype=np.float32)
        xs = x_grid - dx
        ys = y_grid - dy
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3)
        warped = (pts @ H_sample.T).reshape(PATCH_SIZE, PATCH_SIZE, 3)
        w = warped[..., 2]
        w_safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
        xs_w = warped[..., 0] / w_safe
        ys_w = warped[..., 1] / w_safe
        from render_corner import sample_from_image  # type: ignore

        patch01 = sample_from_image(img_high, EXTENT, SUPER_RES, xs_w, ys_w)
    else:
        patch01 = sample_patch_from_image(
            img_high, EXTENT, SUPER_RES, x_grid, y_grid, dx, dy
        )

    blur_sigma = float(params.get("blur_sigma", 0.0))
    if blur_sigma > 0.0:
        patch01 = apply_blur(patch01, blur_sigma)

    contrast = float(params.get("contrast", 1.0))
    brightness = float(params.get("brightness", 0.0))
    gamma = float(params.get("gamma", 1.0))
    patch_u8 = _apply_photometric_uint8(patch01, rng, contrast, brightness, gamma)

    vignetting = float(params.get("vignetting", 0.0))
    patch_u8 = _apply_vignetting(patch_u8, rng, vignetting)

    noise_sigma = float(params.get("noise_sigma", 0.0))
    sp_prob = float(params.get("sp_prob", 0.0))
    patch_u8 = _apply_noise_uint8(patch_u8, rng, noise_sigma, sp_prob)

    H_for_gt = None if np.allclose(H, np.eye(3, dtype=np.float32)) else H
    gt_angles = gt_axes_synthetic(theta_gt, render_mode, H_for_gt)

    gt_xy = (PATCH_CENTER + dx, PATCH_CENTER + dy)
    return PatchSample(
        img=patch_u8,
        gt_xy=gt_xy,
        gt_angles=gt_angles,
        H=H_for_gt,
        meta={
            "theta": theta_gt,
            "axis_skew_deg": skew_deg,
            "blur_sigma": blur_sigma,
            "noise_sigma": noise_sigma,
            "render_mode": render_mode,
            "dx": dx,
            "dy": dy,
        },
    )


def iter_synth_patch_cells(
    cfg: BenchConfig, sweep: SweepSpec, max_cells: int | None = None
) -> Iterator[tuple[int, dict[str, Any], list[PatchSample]]]:
    """Yield (cell_idx, params, samples) for each cell of a patch sweep."""
    n = int(cfg.n_per_cell_patch)
    seed_str = str(cfg.seed)
    cells = _enumerate_cells(sweep)
    if max_cells is not None:
        cells = cells[: int(max_cells)]
    for cell_idx, value in cells:
        params = _resolve_params_for_cell(cfg, sweep, value)
        rng = cell_rng(seed_str, sweep.name, cell_idx)
        samples = [_make_patch(rng, params) for _ in range(n)]
        yield cell_idx, params, samples


def _enumerate_cells(sweep: SweepSpec) -> list[tuple[int, Any]]:
    if sweep.mode == "patch_grid":
        grid = sweep.extras.get("grid", {})
        size_x = int(grid.get("size_x", 0))
        size_y = int(grid.get("size_y", 0))
        x_lo, x_hi = grid.get("x_range", [-0.5, 0.5])
        y_lo, y_hi = grid.get("y_range", [-0.5, 0.5])
        xs = np.linspace(float(x_lo), float(x_hi), size_x) if size_x > 1 else np.array([0.0])
        ys = np.linspace(float(y_lo), float(y_hi), size_y) if size_y > 1 else np.array([0.0])
        out: list[tuple[int, Any]] = []
        idx = 0
        for yy in ys:
            for xx in xs:
                out.append((idx, (float(xx), float(yy))))
                idx += 1
        return out
    return [(i, v) for i, v in enumerate(sweep.values)]


def _resolve_params_for_cell(
    cfg: BenchConfig, sweep: SweepSpec, value: Any
) -> dict[str, Any]:
    base = dict(cfg.defaults)
    base.setdefault("homography", cfg.homography)
    if sweep.mode == "patch_grid":
        x, y = value
        base["dx"] = float(x)
        base["dy"] = float(y)
        return base
    if sweep.param in ("dx_dy",):
        x, y = value
        base["dx"] = float(x)
        base["dy"] = float(y)
        return base
    base[sweep.param] = value
    return base


# ----------------------------- Chess board sweeps ----------------------------


def _import_synthimages():
    bootstrap_paths()
    import synthimages  # type: ignore

    return synthimages


def _build_camera_intrinsics(chess_cfg: dict[str, Any]) -> np.ndarray:
    img_w = int(chess_cfg.get("img_w", 640))
    img_h = int(chess_cfg.get("img_h", 480))
    fx = float(chess_cfg.get("fx", 800.0))
    fy = float(chess_cfg.get("fy", 800.0))
    cx = float(chess_cfg.get("cx", img_w * 0.5))
    cy = float(chess_cfg.get("cy", img_h * 0.5))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_chess_sample(
    rng: np.random.Generator,
    chess_cfg: dict[str, Any],
    pose_yaw: float,
    pose_pitch: float,
) -> ChessSample:
    synth = _import_synthimages()
    inner_x = int(chess_cfg.get("inner_x", 7))
    inner_y = int(chess_cfg.get("inner_y", 5))
    pps = int(chess_cfg.get("pps", 24))
    square_size = float(chess_cfg.get("square_size", 0.03))
    img_w = int(chess_cfg.get("img_w", 640))
    img_h = int(chess_cfg.get("img_h", 480))
    K = _build_camera_intrinsics(chess_cfg)
    tex = synth.make_chess_texture(inner_x, inner_y, pps, invert=False)
    img, meta = synth.generate_one(
        idx=0,
        tex=tex,
        inner_x=inner_x,
        inner_y=inner_y,
        square_size=square_size,
        pps=pps,
        img_w=img_w,
        img_h=img_h,
        K=K,
        yaw_max=max(abs(pose_yaw), 1e-3),
        pitch_max=max(abs(pose_pitch), 1e-3),
        roll_max=float(chess_cfg.get("roll_max", 5.0)),
        z_range=tuple(chess_cfg.get("z_range", [0.4, 0.7])),
        margin=int(chess_cfg.get("margin", 24)),
        noise_sigma=float(chess_cfg.get("noise_sigma", 1.0)),
        sp_prob=float(chess_cfg.get("sp_prob", 0.0)),
        blur_sigma=float(chess_cfg.get("blur_sigma", 0.7)),
        gamma_range=tuple(chess_cfg.get("gamma_range", [0.95, 1.05])),
        contrast_range=tuple(chess_cfg.get("contrast_range", [0.95, 1.05])),
        brightness_range=tuple(chess_cfg.get("brightness_range", [-5.0, 5.0])),
        vignetting=float(chess_cfg.get("vignetting", 0.0)),
        rng=rng,
    )
    H_tex2img = np.asarray(meta["H_tex2img"], dtype=np.float64)
    gt_xy = np.asarray(meta["gt_corners_uv"], dtype=np.float64)
    gt_uv_tex = np.asarray(meta["gt_corners_uv_tex"], dtype=np.float64)
    gt_angles = gt_axes_from_homography(H_tex2img, gt_uv_tex)
    return ChessSample(
        img=img,
        gt_xy=gt_xy,
        gt_angles=gt_angles,
        H_tex2img=H_tex2img,
        meta={
            "yaw_max": pose_yaw,
            "pitch_max": pose_pitch,
            "inner_x": inner_x,
            "inner_y": inner_y,
        },
    )


def iter_synth_chess_cells(
    cfg: BenchConfig, sweep: SweepSpec, max_cells: int | None = None
) -> Iterator[tuple[int, dict[str, Any], list[ChessSample]]]:
    """Yield (cell_idx, params, samples) for each cell of a chess sweep."""
    n = int(cfg.n_per_cell_chess)
    seed_str = str(cfg.seed)
    chess_cfg = dict(cfg.chess)
    cells = _enumerate_cells(sweep)
    if max_cells is not None:
        cells = cells[: int(max_cells)]
    for cell_idx, value in cells:
        # Pose values are pairs (yaw, pitch) in degrees by convention.
        if isinstance(value, (list, tuple)) and len(value) == 2:
            yaw, pitch = float(value[0]), float(value[1])
        else:
            yaw, pitch = float(value), float(value)
        params = {"pose_yaw": yaw, "pose_pitch": pitch, **chess_cfg}
        rng = cell_rng(seed_str, sweep.name, cell_idx)
        samples = [_make_chess_sample(rng, chess_cfg, yaw, pitch) for _ in range(n)]
        yield cell_idx, params, samples


def iter_cells(
    cfg: BenchConfig, sweep: SweepSpec, max_cells: int | None = None
) -> Iterable[tuple[int, dict[str, Any], list[Any]]]:
    """Dispatch to the right iterator based on `sweep.mode`."""
    if sweep.mode in ("patch", "patch_grid"):
        yield from iter_synth_patch_cells(cfg, sweep, max_cells)
    elif sweep.mode == "chess":
        yield from iter_synth_chess_cells(cfg, sweep, max_cells)
    else:
        raise ValueError(f"unknown sweep mode {sweep.mode!r}")
