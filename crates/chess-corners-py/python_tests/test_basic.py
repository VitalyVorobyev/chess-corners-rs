import numpy as np
import pytest

import chess_corners


def _checkerboard(square_size: int = 16, squares: int = 8) -> np.ndarray:
    grid = (np.indices((squares, squares)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(grid, np.ones((square_size, square_size), dtype=np.uint8)) * 255
    return board


def test_find_chess_corners_basic():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig()
    cfg.threshold_value = 0.1
    cfg.min_cluster_size = 1

    corners = chess_corners.find_chess_corners(img, cfg)
    assert corners.dtype == np.float32
    assert corners.ndim == 2
    assert corners.shape[1] == 9
    assert corners.shape[0] > 0


def test_find_chess_corners_rejects_wrong_dtype():
    img = _checkerboard(square_size=16, squares=8).astype(np.float32)
    with pytest.raises(TypeError):
        chess_corners.find_chess_corners(img)


def test_nested_config_objects():
    cfg = chess_corners.ChessConfig()
    cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER
    cfg.refiner.forstner.max_offset = 2.0
    assert cfg.refiner.kind is chess_corners.RefinementMethod.FORSTNER
    assert cfg.refiner.forstner.max_offset == pytest.approx(2.0)


def test_config_roundtrip_and_print_helpers():
    cfg = chess_corners.ChessConfig.multiscale()
    cfg.detector_mode = chess_corners.DetectorMode.BROAD
    cfg.descriptor_mode = chess_corners.DescriptorMode.CANONICAL
    cfg.threshold_mode = chess_corners.ThresholdMode.ABSOLUTE
    cfg.threshold_value = 4.5
    cfg.refiner.kind = chess_corners.RefinementMethod.SADDLE_POINT
    cfg.refiner.saddle_point.max_offset = 2.0

    encoded = cfg.to_json()
    decoded = chess_corners.ChessConfig.from_json(encoded)

    assert decoded.to_dict() == cfg.to_dict()
    assert "threshold_mode" in str(cfg)
    assert "_native" not in chess_corners.__all__


def test_public_signature_and_defaults():
    import inspect

    sig = inspect.signature(chess_corners.find_chess_corners)
    assert str(sig) == "(image: 'Any', cfg: 'ChessConfig | None' = None) -> 'Any'"

    cfg = chess_corners.ChessConfig()
    assert isinstance(cfg.refiner, chess_corners.RefinerConfig)
    assert isinstance(cfg.refiner.forstner, chess_corners.ForstnerConfig)
    assert isinstance(cfg.refiner.saddle_point, chess_corners.SaddlePointConfig)


def test_ml_refiner_api():
    if not hasattr(chess_corners, "find_chess_corners_with_ml"):
        pytest.skip("ml-refiner bindings not enabled")
    img = _checkerboard(square_size=8, squares=4)
    cfg = chess_corners.ChessConfig()
    corners = chess_corners.find_chess_corners_with_ml(img, cfg)
    assert corners.dtype == np.float32
    assert corners.ndim == 2
    assert corners.shape[1] == 9


def test_radon_heatmap_shape_and_dtype():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig.radon()

    heatmap = chess_corners.radon_heatmap(img, cfg)

    upsample = max(1, min(2, cfg.radon_detector.image_upsample))
    assert heatmap.dtype == np.float32
    assert heatmap.shape == (img.shape[0] * upsample, img.shape[1] * upsample)
    assert float(heatmap.max()) > 0.0
    assert float(heatmap.min()) >= 0.0


def test_radon_heatmap_default_config():
    img = _checkerboard(square_size=16, squares=8)

    heatmap = chess_corners.radon_heatmap(img)

    assert heatmap.dtype == np.float32
    assert heatmap.ndim == 2
    assert heatmap.shape[0] >= img.shape[0]
    assert heatmap.shape[1] >= img.shape[1]
