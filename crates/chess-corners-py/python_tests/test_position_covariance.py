import json

import numpy as np
import pytest

import chess_corners


def _noisy_board(size: int = 128, cell: int = 8, sigma: float = 2.0) -> np.ndarray:
    grid = (np.indices((size, size)) // cell).sum(axis=0) % 2
    board = np.where(grid == 0, 200.0, 40.0)
    rng = np.random.default_rng(7)
    noisy = board + rng.normal(0.0, sigma, board.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def _saddle_config() -> chess_corners.DetectorConfig:
    return chess_corners.DetectorConfig.chess().with_chess(
        refiner=chess_corners.ChessRefiner.saddle_point()
    )


def test_saddle_point_reports_position_covariance():
    img = _noisy_board()
    det = chess_corners.Detector(_saddle_config()).detect(img)
    assert len(det) > 0
    cov = det.position_covariance
    assert cov.dtype == np.float32
    assert cov.shape == (len(det), 2, 2)
    present = ~np.isnan(cov[:, 0, 0])
    assert present.sum() > len(det) // 2
    c = cov[present]
    assert (c[:, 0, 0] > 0).all() and (c[:, 1, 1] > 0).all()
    np.testing.assert_array_equal(c[:, 0, 1], c[:, 1, 0])
    # Rows without a covariance are NaN throughout.
    assert np.isnan(cov[~present]).all()


def test_default_refiner_reports_no_covariance():
    det = chess_corners.Detector(chess_corners.DetectorConfig.chess()).detect(_noisy_board())
    assert len(det) > 0
    assert det.position_covariance.shape == (len(det), 2, 2)
    assert np.isnan(det.position_covariance).all()


def test_supplied_noise_sigma_scales_covariance():
    img = _noisy_board()
    lo = chess_corners.Detector(_saddle_config().with_noise_sigma(1.0)).detect(img)
    hi = chess_corners.Detector(_saddle_config().with_noise_sigma(4.0)).detect(img)
    a, b = lo.position_covariance, hi.position_covariance
    present = ~np.isnan(a[:, 0, 0])
    assert present.any()
    # Off-diagonal: no floor, scales with σ².
    np.testing.assert_allclose(b[present, 0, 1], 16.0 * a[present, 0, 1], rtol=1e-4, atol=1e-9)
    assert (b[present, 0, 0] > a[present, 0, 0]).all()


def test_noise_sigma_config_round_trip():
    cfg = chess_corners.DetectorConfig.chess()
    assert cfg.noise_sigma is None
    assert cfg.to_dict()["noise_sigma"] is None
    cfg2 = cfg.with_noise_sigma(2.5)
    assert cfg2.noise_sigma == pytest.approx(2.5)
    assert cfg.noise_sigma is None
    back = chess_corners.DetectorConfig.from_json(cfg2.to_json())
    assert back.noise_sigma == pytest.approx(2.5)
    assert json.loads(cfg2.to_json())["noise_sigma"] == pytest.approx(2.5)
    assert chess_corners.DetectorConfig.from_dict({"noise_sigma": None}).noise_sigma is None
    cfg2.noise_sigma = None
    assert cfg2.noise_sigma is None
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict({"noise_sigma": "high"})


def test_invalid_noise_sigma_is_rejected():
    with pytest.raises(ValueError, match="noise_sigma"):
        chess_corners.Detector(_saddle_config().with_noise_sigma(-1.0))


def test_radon_saddle_point_reports_position_covariance():
    img = _noisy_board()
    cfg = chess_corners.DetectorConfig.radon()
    assert cfg.strategy.radon.refiner.kind == "peak_fit"
    saddle = cfg.with_radon(refiner=chess_corners.RadonRefiner.saddle_point())
    det = chess_corners.Detector(saddle).detect(img)
    assert len(det) > 0
    present = ~np.isnan(det.position_covariance[:, 0, 0])
    assert present.sum() > len(det) // 2

    plain = chess_corners.Detector(cfg).detect(img)
    assert len(plain) == len(det)
    assert np.isnan(plain.position_covariance).all()
    # Corners without a covariance keep their peak-fit position.
    np.testing.assert_array_equal(det.xy[~present], plain.xy[~present])


def test_estimate_noise_sigma_matches_the_detector():
    img = _noisy_board(sigma=3.0)
    sigma = chess_corners.estimate_noise_sigma(img)
    assert sigma is not None and 2.5 < sigma < 3.5
    # The detector's estimate: same covariances as with σ supplied.
    est = chess_corners.Detector(_saddle_config()).detect(img)
    known = chess_corners.Detector(_saddle_config().with_noise_sigma(sigma)).detect(img)
    np.testing.assert_array_equal(est.position_covariance, known.position_covariance)
    assert chess_corners.estimate_noise_sigma(np.full((3, 3), 9, np.uint8)) is None
    with pytest.raises(TypeError):
        chess_corners.estimate_noise_sigma(img.astype(np.float32))
