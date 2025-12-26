"""Parity tests between PyTorch fixtures and ONNX runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "ml"
ONNX_PATH = ASSETS_DIR / "chess_refiner_v1.onnx"
FIXTURES_DIR = ASSETS_DIR / "fixtures"
PATCHES_PATH = FIXTURES_DIR / "patches.npy"
TORCH_OUT_PATH = FIXTURES_DIR / "torch_out.npy"


def _load_fixtures() -> tuple[np.ndarray, np.ndarray]:
    if not ONNX_PATH.exists():
        raise AssertionError(
            f"ONNX model missing at {ONNX_PATH}. Run export_onnx.py to generate it."
        )
    if not PATCHES_PATH.exists() or not TORCH_OUT_PATH.exists():
        raise AssertionError(
            "Fixtures missing. Run tools/ml_refiner/export_onnx.py to generate them."
        )

    patches = np.load(PATCHES_PATH).astype(np.float32)
    torch_out = np.load(TORCH_OUT_PATH).astype(np.float32)
    return patches, torch_out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_onnx_parity() -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise AssertionError(
            "onnxruntime is required for parity tests. Install it in your venv."
        ) from exc

    patches, torch_out = _load_fixtures()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(["pred"], {"patches": patches})[0]

    diff = out - torch_out
    abs_max = float(np.max(np.abs(diff)))
    rel_max = float(np.max(np.abs(diff) / (np.abs(torch_out) + 1e-6)))

    assert abs_max < 1e-4, f"abs_max {abs_max} exceeds tolerance"
    assert rel_max < 1e-4, f"rel_max {rel_max} exceeds tolerance"

    conf_torch = _sigmoid(torch_out[:, 2])
    conf_onnx = _sigmoid(out[:, 2])
    conf_diff = conf_onnx - conf_torch
    conf_abs = float(np.max(np.abs(conf_diff)))
    conf_rel = float(np.max(np.abs(conf_diff) / (np.abs(conf_torch) + 1e-6)))

    assert conf_abs < 1e-4, f"conf abs_max {conf_abs} exceeds tolerance"
    assert conf_rel < 1e-4, f"conf rel_max {conf_rel} exceeds tolerance"
