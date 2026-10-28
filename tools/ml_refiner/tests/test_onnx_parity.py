"""Parity tests between PyTorch fixtures and ONNX runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[3] / "crates" / "chess-corners-ml" / "assets" / "ml"
ONNX_PATH = ASSETS_DIR / "chess_refiner_v4.onnx"
FIXTURES_DIR = ASSETS_DIR / "fixtures" / "v4"
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


ATOL = 1e-4
RTOL = 1e-4


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

    # |a - b| <= atol + rtol·|b| (the np.allclose form): a pure relative
    # bound is ill-posed for outputs near zero.
    excess = np.abs(out - torch_out) - (ATOL + RTOL * np.abs(torch_out))
    assert float(np.max(excess)) <= 0.0, f"pred exceeds tolerance by {np.max(excess)}"

    conf_torch = _sigmoid(torch_out[:, 2])
    conf_onnx = _sigmoid(out[:, 2])
    excess = np.abs(conf_onnx - conf_torch) - (ATOL + RTOL * np.abs(conf_torch))
    assert float(np.max(excess)) <= 0.0, f"conf exceeds tolerance by {np.max(excess)}"
