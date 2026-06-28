import numpy as np
import pytest

import chess_corners


def _checkerboard(square_size: int = 16, squares: int = 8) -> np.ndarray:
    grid = (np.indices((squares, squares)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(grid, np.ones((square_size, square_size), dtype=np.uint8)) * 255
    return board


def test_detector_basic():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.DetectorConfig()
    cfg.threshold = 0.1
    cfg.detection.min_cluster_size = 1

    detector = chess_corners.Detector(cfg)
    result = detector.detect(img)
    assert isinstance(result, chess_corners.Detections)
    assert result.xy.dtype == np.float32
    assert result.xy.ndim == 2
    assert result.xy.shape[1] == 2
    assert result.response.dtype == np.float32
    assert result.response.ndim == 1
    assert len(result) > 0
    assert result.xy.shape[0] == len(result)


def test_detect_without_orientation_yields_nan_axes():
    img = _checkerboard(square_size=16, squares=8)
    base = chess_corners.DetectorConfig()
    base.threshold = 0.1
    base.detection.min_cluster_size = 1

    # Orientation on: angles and sigmas are finite (N, 2) arrays.
    on = chess_corners.Detector(base).detect(img)
    assert len(on) > 0
    assert on.angles is not None and on.sigmas is not None
    assert on.angles.shape == (len(on), 2)
    assert on.sigmas.shape == (len(on), 2)
    assert np.isfinite(on.angles).all()
    assert np.isfinite(on.sigmas).all()
    assert np.isfinite(on.xy).all()
    assert np.isfinite(on.response).all()

    # Orientation off: same corner count; angles and sigmas are None.
    off_cfg = base.without_orientation()
    off = chess_corners.Detector(off_cfg).detect(img)
    assert len(off) == len(on)
    assert off.angles is None
    assert off.sigmas is None
    assert np.isfinite(off.xy).all()
    assert np.isfinite(off.response).all()


def test_detector_rejects_wrong_dtype():
    img = _checkerboard(square_size=16, squares=8).astype(np.float32)
    detector = chess_corners.Detector()
    with pytest.raises(TypeError):
        detector.detect(img)


def test_chess_refiner_variant_and_payload():
    """ChESS refiner is a tagged class; `payload` returns the active tuning."""

    com = chess_corners.CenterOfMassConfig()
    com.radius = 3
    refiner = chess_corners.ChessRefiner.center_of_mass(com)
    assert refiner.kind == "center_of_mass"
    assert isinstance(refiner.payload, chess_corners.CenterOfMassConfig)
    assert refiner.payload.radius == 3

    fcfg = chess_corners.ForstnerConfig()
    fcfg.max_offset = 2.0
    refiner = chess_corners.ChessRefiner.forstner(fcfg)
    assert refiner.kind == "forstner"
    assert isinstance(refiner.payload, chess_corners.ForstnerConfig)
    assert refiner.payload.max_offset == pytest.approx(2.0)


def test_chess_refiner_attached_to_chess_strategy():
    cfg = chess_corners.DetectorConfig().with_chess(refiner=chess_corners.ChessRefiner.forstner())
    assert cfg.strategy.chess.refiner.kind == "forstner"


def test_config_roundtrip_and_print_helpers():
    cfg = chess_corners.DetectorConfig.chess_multiscale()
    cfg.strategy.chess.ring = chess_corners.ChessRing.BROAD
    cfg.threshold = 4.5
    saddle = chess_corners.SaddlePointConfig()
    saddle.max_offset = 2.0
    chess = cfg.strategy.chess
    chess.refiner = chess_corners.ChessRefiner.saddle_point(saddle)
    cfg.strategy = chess_corners.DetectionStrategy.from_chess(chess)

    encoded = cfg.to_json()
    decoded = chess_corners.DetectorConfig.from_json(encoded)

    assert decoded.to_dict() == cfg.to_dict()
    assert "threshold" in str(cfg)
    assert "_native" not in chess_corners.__all__


def test_detector_constructor_signature():
    import inspect

    sig = inspect.signature(chess_corners.Detector)
    assert "cfg" in sig.parameters

    cfg = chess_corners.DetectorConfig()
    assert isinstance(cfg.strategy.chess, chess_corners.ChessConfig)
    assert isinstance(cfg.strategy.chess.refiner, chess_corners.ChessRefiner)


def test_radon_heatmap_shape_and_dtype():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.DetectorConfig.radon()

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
    """Native typed `DetectorConfig` reaches the detector without JSON serialization."""

    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.DetectorConfig()
    cfg.threshold = 0.1
    fcfg = chess_corners.ForstnerConfig()
    fcfg.max_offset = 1.75
    chess = cfg.strategy.chess
    chess.refiner = chess_corners.ChessRefiner.forstner(fcfg)
    cfg.strategy = chess_corners.DetectionStrategy.from_chess(chess)

    result = chess_corners.Detector(cfg).detect(img)
    assert isinstance(result, chess_corners.Detections)
    assert result.xy.dtype == np.float32


def test_invalid_cfg_type_raises_type_error():
    with pytest.raises(TypeError):
        # Plain dicts aren't accepted at the FFI boundary; they must
        # be converted to a DetectorConfig first via DetectorConfig.from_dict().
        chess_corners.Detector({"merge_radius": 2.0})


def test_unknown_top_level_keys_rejected():
    with pytest.raises(chess_corners.ConfigError, match="unknown keys"):
        chess_corners.DetectorConfig.from_dict({"unexpected": 1})


def test_descriptor_mode_on_top_level_rejected():
    with pytest.raises(chess_corners.ConfigError, match="descriptor_mode"):
        chess_corners.DetectorConfig.from_dict(
            {"descriptor_mode": "canonical"}
        )


def test_refiner_on_top_level_rejected():
    with pytest.raises(chess_corners.ConfigError, match="refiner"):
        chess_corners.DetectorConfig.from_dict(
            {"refiner": {"center_of_mass": {}}}
        )


def test_detection_strategy_factory_and_accessor():
    """`from_radon` factory builds a radon-variant strategy with the right tag."""

    cfg = chess_corners.DetectorConfig()
    cfg.strategy = chess_corners.DetectionStrategy.from_radon(
        chess_corners.RadonConfig()
    )

    assert cfg.strategy.kind == "radon"
    assert cfg.strategy.chess is None
    assert cfg.strategy.radon is not None
    assert isinstance(cfg.strategy.radon, chess_corners.RadonConfig)


def test_multiscale_tagged_class_single_scale():
    cfg = chess_corners.DetectorConfig.chess()
    ms = cfg.multiscale
    assert ms.kind == "single_scale"
    with pytest.raises(AttributeError):
        _ = ms.levels
    with pytest.raises(AttributeError):
        _ = ms.min_size


def test_multiscale_tagged_class_pyramid():
    """Multiscale pyramid variant exposes its tuning fields."""

    cfg = chess_corners.DetectorConfig.chess_multiscale()
    ms = cfg.multiscale
    assert ms.kind == "pyramid"
    assert ms.levels == 3
    assert ms.min_size == 128
    assert ms.refinement_radius == 3


def test_multiscale_factories():
    single = chess_corners.MultiscaleConfig.single_scale()
    assert single.kind == "single_scale"

    pyramid = chess_corners.MultiscaleConfig.pyramid(
        levels=4, min_size=96, refinement_radius=2
    )
    assert pyramid.kind == "pyramid"
    assert pyramid.levels == 4
    assert pyramid.min_size == 96
    assert pyramid.refinement_radius == 2


def test_detector_config_roundtrip():
    """config() / apply_config() round-trip and buffer reuse."""
    img = _checkerboard(square_size=16, squares=8)

    cfg = chess_corners.DetectorConfig.chess()
    cfg.threshold = 0.1
    detector = chess_corners.Detector(cfg)

    # Snapshot the live config and verify it reflects the applied threshold.
    snapshot = detector.config()
    assert abs(snapshot.threshold - 0.1) < 1e-6

    # Mutate the snapshot and apply it back.
    snapshot.threshold = 0.0
    detector.apply_config(snapshot)

    # The updated config should be reflected in a new snapshot.
    updated = detector.config()
    assert abs(updated.threshold - 0.0) < 1e-6

    # Detector must still produce a valid result after apply_config.
    result = detector.detect(img)
    assert isinstance(result, chess_corners.Detections)
    assert result.xy.dtype == np.float32


def test_radon_multiscale_classmethod():
    """DetectorConfig.radon_multiscale() builds a valid radon+multiscale config."""

    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.DetectorConfig.radon_multiscale()

    # Config shape checks.
    assert cfg.strategy.kind == "radon"
    assert cfg.multiscale.kind == "pyramid"

    # End-to-end: detector must produce at least some corners.
    cfg.threshold = 0.05
    result = chess_corners.Detector(cfg).detect(img)
    assert isinstance(result, chess_corners.Detections)
    assert result.xy.dtype == np.float32
    assert len(result) > 0, "radon_multiscale detector returned no corners"
