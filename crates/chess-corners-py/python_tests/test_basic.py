import numpy as np
import pytest

import chess_corners


def _checkerboard(square_size: int = 16, squares: int = 8) -> np.ndarray:
    grid = (np.indices((squares, squares)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(grid, np.ones((square_size, square_size), dtype=np.uint8)) * 255
    return board


def test_detector_basic():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig()
    cfg.threshold = chess_corners.Threshold.relative(0.1)
    cfg.strategy.chess.min_cluster_size = 1

    detector = chess_corners.Detector(cfg)
    corners = detector.detect(img)
    assert corners.dtype == np.float32
    assert corners.ndim == 2
    assert corners.shape[1] == 9
    assert corners.shape[0] > 0


def test_detector_rejects_wrong_dtype():
    img = _checkerboard(square_size=16, squares=8).astype(np.float32)
    detector = chess_corners.Detector()
    with pytest.raises(TypeError):
        detector.detect(img)


def test_nested_config_objects():
    cfg = chess_corners.ChessConfig()
    cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER
    cfg.refiner.forstner.max_offset = 2.0
    assert cfg.refiner.kind is chess_corners.RefinementMethod.FORSTNER
    assert cfg.refiner.forstner.max_offset == pytest.approx(2.0)


def test_config_roundtrip_and_print_helpers():
    cfg = chess_corners.ChessConfig.multiscale()
    cfg.strategy.chess.ring = chess_corners.ChessRing.BROAD
    cfg.descriptor_mode = chess_corners.DescriptorMode.CANONICAL
    cfg.threshold = chess_corners.Threshold.absolute(4.5)
    cfg.refiner.kind = chess_corners.RefinementMethod.SADDLE_POINT
    cfg.refiner.saddle_point.max_offset = 2.0

    encoded = cfg.to_json()
    decoded = chess_corners.ChessConfig.from_json(encoded)

    assert decoded.to_dict() == cfg.to_dict()
    assert "threshold" in str(cfg)
    assert "_native" not in chess_corners.__all__


def test_detector_constructor_signature():
    import inspect

    sig = inspect.signature(chess_corners.Detector)
    assert "cfg" in sig.parameters

    cfg = chess_corners.ChessConfig()
    assert isinstance(cfg.refiner, chess_corners.RefinerConfig)
    assert isinstance(cfg.refiner.forstner, chess_corners.ForstnerConfig)
    assert isinstance(cfg.refiner.saddle_point, chess_corners.SaddlePointConfig)


def test_radon_heatmap_shape_and_dtype():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig.radon()

    detector = chess_corners.Detector(cfg)
    heatmap = detector.radon_heatmap(img)

    upsample = max(1, min(2, cfg.strategy.radon.image_upsample))
    assert heatmap.dtype == np.float32
    assert heatmap.shape == (img.shape[0] * upsample, img.shape[1] * upsample)
    assert float(heatmap.max()) > 0.0
    assert float(heatmap.min()) >= 0.0


def test_radon_heatmap_default_config():
    img = _checkerboard(square_size=16, squares=8)

    detector = chess_corners.Detector()
    heatmap = detector.radon_heatmap(img)

    assert heatmap.dtype == np.float32
    assert heatmap.ndim == 2
    assert heatmap.shape[0] >= img.shape[0]
    assert heatmap.shape[1] >= img.shape[1]


def test_typed_config_passes_through_ffi_directly():
    """Native typed `ChessConfig` reaches the detector without JSON serialization."""

    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig()
    cfg.threshold = chess_corners.Threshold.relative(0.1)
    cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER
    cfg.refiner.forstner.max_offset = 1.75

    corners = chess_corners.Detector(cfg).detect(img)
    assert corners.dtype == np.float32
    assert corners.ndim == 2
    assert corners.shape[1] == 9


def test_invalid_cfg_type_raises_type_error():
    with pytest.raises(TypeError):
        # Plain dicts aren't accepted at the FFI boundary; they must
        # be converted to a ChessConfig first via ChessConfig.from_dict().
        chess_corners.Detector({"merge_radius": 2.0})


def test_unknown_top_level_keys_rejected():
    with pytest.raises(chess_corners.ConfigError, match="unknown keys"):
        chess_corners.ChessConfig.from_dict({"unexpected": 1})


def test_detection_strategy_factory_and_accessor():
    """`from_radon` factory builds a radon-variant strategy with the right tag."""

    cfg = chess_corners.ChessConfig()
    cfg.strategy = chess_corners.DetectionStrategy.from_radon(
        chess_corners.RadonStrategy()
    )

    assert cfg.strategy.kind == "radon"
    assert cfg.strategy.chess is None
    assert cfg.strategy.radon is not None
    assert isinstance(cfg.strategy.radon, chess_corners.RadonStrategy)


def test_multiscale_params_attached_to_chess_strategy():
    """Multiscale settings live on `cfg.strategy.chess.multiscale`."""

    cfg = chess_corners.ChessConfig.multiscale()
    ms = cfg.strategy.chess.multiscale
    assert ms is not None
    assert ms.pyramid_levels == 3
    assert ms.pyramid_min_size == 128

    # `single_scale` should leave multiscale=None.
    single = chess_corners.ChessConfig.single_scale()
    assert single.strategy.chess.multiscale is None
