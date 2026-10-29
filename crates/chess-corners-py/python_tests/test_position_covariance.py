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


def _propagated(cfg: chess_corners.DetectorConfig) -> chess_corners.DetectorConfig:
    return cfg.with_uncertainty(chess_corners.Uncertainty.propagated())


def _known(cfg: chess_corners.DetectorConfig, sigma: float) -> chess_corners.DetectorConfig:
    return cfg.with_uncertainty(chess_corners.Uncertainty.propagated(noise_sigma=sigma))


def _chess() -> chess_corners.DetectorConfig:
    return _propagated(chess_corners.DetectorConfig.chess())


def _radon() -> chess_corners.DetectorConfig:
    return _propagated(chess_corners.DetectorConfig.radon())


def _saddle_config() -> chess_corners.DetectorConfig:
    return _chess().with_refiner(chess_corners.ImageRefiner.saddle_point())


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


@pytest.mark.parametrize(
    "cfg",
    [
        chess_corners.DetectorConfig(),
        chess_corners.DetectorConfig.chess(),
        chess_corners.DetectorConfig.chess_multiscale(),
        chess_corners.DetectorConfig.radon(),
        chess_corners.DetectorConfig.radon_multiscale(),
    ],
)
def test_covariance_is_off_by_default(cfg):
    assert cfg.uncertainty == chess_corners.Uncertainty.off()
    det = chess_corners.Detector(cfg).detect(_noisy_board())
    assert len(det) > 0
    assert np.isnan(det.position_covariance).all()


def test_default_refiner_reports_covariance():
    det = chess_corners.Detector(_chess()).detect(_noisy_board())
    assert len(det) > 0
    cov = det.position_covariance
    assert cov.shape == (len(det), 2, 2)
    assert np.isfinite(cov).all()


def test_radon_peak_fit_reports_covariance():
    cfg = _radon()
    assert cfg.refiner is None
    det = chess_corners.Detector(cfg).detect(_noisy_board())
    assert len(det) > 0
    assert np.isfinite(det.position_covariance).all()


def test_uncertainty_value_object():
    off, est, known = (
        chess_corners.Uncertainty.off(),
        chess_corners.Uncertainty.propagated(),
        chess_corners.Uncertainty.propagated(noise_sigma=1.5),
    )
    assert (off.kind, off.noise_sigma) == ("off", None)
    assert (est.kind, est.noise_sigma) == ("propagated", None)
    assert (known.kind, known.noise_sigma) == ("propagated", 1.5)
    assert est == chess_corners.Uncertainty.propagated(None)
    assert known == chess_corners.Uncertainty.propagated(1.5)
    assert len({repr(off), repr(est), repr(known)}) == 3 and off != est != known
    assert repr(known) == "Uncertainty.propagated(noise_sigma=1.5)"
    assert off.to_dict() == {"off": None}
    assert est.to_dict() == {"propagated": {"noise": "estimate"}}
    assert known.to_dict() == {"propagated": {"noise": {"known": 1.5}}}
    for u in (off, est, known):
        assert chess_corners.Uncertainty.from_dict(u.to_dict()) == u
    # The serde JSON shapes the Rust config and the CLI read.
    assert chess_corners.Uncertainty.from_dict("off") == off
    assert chess_corners.Uncertainty.from_dict(
        {"propagated": {"noise": {"estimate": None}}}
    ) == est
    with pytest.raises(AttributeError):
        known.noise_sigma = 2.0


@pytest.mark.parametrize(
    "bad",
    [
        "propagated",
        "bogus",
        1,
        {},
        {"off": None, "propagated": {"noise": "estimate"}},
        {"off": 1},
        {"propagated": {}},
        {"propagated": None},
        {"propagated": {"noise": "known"}},
        {"propagated": {"noise": {"known": "high"}}},
        {"propagated": {"noise": {"known": 1.0, "estimate": None}}},
        {"propagated": {"noise": "estimate", "sigma": 1.0}},
    ],
)
def test_unknown_uncertainty_shapes_raise(bad):
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.Uncertainty.from_dict(bad)
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict({"uncertainty": bad})


def test_uncertainty_default_and_round_trip():
    cfg = chess_corners.DetectorConfig.chess()
    assert cfg.uncertainty == chess_corners.Uncertainty.off()
    assert cfg.to_dict()["uncertainty"] == {"off": None}
    for u in (
        chess_corners.Uncertainty.propagated(),
        chess_corners.Uncertainty.propagated(noise_sigma=2.5),
    ):
        on = cfg.with_uncertainty(u)
        assert on.uncertainty == u
        assert cfg.uncertainty == chess_corners.Uncertainty.off()
        assert json.loads(on.to_json())["uncertainty"] == u.to_dict()
        back = chess_corners.DetectorConfig.from_json(on.to_json())
        assert back.uncertainty == u
    assert chess_corners.DetectorConfig.from_dict({}).uncertainty == (
        chess_corners.Uncertainty.off()
    )
    on.uncertainty = chess_corners.Uncertainty.off()
    assert on.uncertainty == chess_corners.Uncertainty.off()
    # The removed top-level key points to its replacement.
    with pytest.raises(chess_corners.ConfigError, match="uncertainty"):
        chess_corners.DetectorConfig.from_dict({"noise_sigma": 1.0})
    assert not hasattr(cfg, "noise_sigma")
    assert not hasattr(cfg, "with_noise_sigma")


@pytest.mark.parametrize(
    "cfg",
    [_chess(), _radon(), _saddle_config()],
)
def test_uncertainty_off_drops_covariance_not_positions(cfg):
    img = _noisy_board()
    on = chess_corners.Detector(cfg).detect(img)
    off = chess_corners.Detector(cfg.with_uncertainty(chess_corners.Uncertainty.off())).detect(img)
    assert len(on) > 0
    assert np.isfinite(on.position_covariance).any()
    assert np.isnan(off.position_covariance).all()
    np.testing.assert_array_equal(on.xy, off.xy)


def test_supplied_noise_sigma_scales_covariance():
    img = _noisy_board()
    lo = chess_corners.Detector(_known(_saddle_config(), 1.0)).detect(img)
    hi = chess_corners.Detector(_known(_saddle_config(), 4.0)).detect(img)
    a, b = lo.position_covariance, hi.position_covariance
    present = ~np.isnan(a[:, 0, 0])
    assert present.any()
    # Off-diagonal: no floor, scales with σ².
    np.testing.assert_allclose(b[present, 0, 1], 16.0 * a[present, 0, 1], rtol=1e-4, atol=1e-9)
    assert (b[present, 0, 0] > a[present, 0, 0]).all()


def test_invalid_noise_sigma_is_rejected():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="noise sigma"):
            chess_corners.Detector(_known(_saddle_config(), bad))
    chess_corners.Detector(_known(_saddle_config(), 0.0))


def test_radon_saddle_point_reports_position_covariance():
    img = _noisy_board()
    cfg = _radon()
    assert cfg.refiner is None
    saddle = cfg.with_refiner(chess_corners.ImageRefiner.saddle_point())
    det = chess_corners.Detector(saddle).detect(img)
    assert len(det) > 0
    present = ~np.isnan(det.position_covariance[:, 0, 0])
    assert present.sum() > len(det) // 2

    plain = chess_corners.Detector(cfg).detect(img)
    assert len(plain) == len(det)
    # Corners without a saddle-point covariance keep their peak-fit position.
    np.testing.assert_array_equal(det.xy[~present], plain.xy[~present])


def test_estimate_noise_sigma_matches_the_detector():
    img = _noisy_board(sigma=3.0)
    sigma = chess_corners.estimate_noise_sigma(img)
    assert sigma is not None and 2.5 < sigma < 3.5
    # The detector's estimate: same covariances as with σ supplied.
    est = chess_corners.Detector(_saddle_config()).detect(img)
    known = chess_corners.Detector(_known(_saddle_config(), sigma)).detect(img)
    np.testing.assert_array_equal(est.position_covariance, known.position_covariance)
    assert chess_corners.estimate_noise_sigma(np.full((3, 3), 9, np.uint8)) is None
    with pytest.raises(TypeError):
        chess_corners.estimate_noise_sigma(img.astype(np.float32))


@pytest.mark.parametrize("make", [_chess, _radon, _saddle_config])
def test_detections_report_the_frame_noise_sigma(make):
    img = _noisy_board(sigma=3.0)
    estimated = chess_corners.Detector(make()).detect(img)
    assert estimated.noise_sigma == chess_corners.estimate_noise_sigma(img)
    supplied = chess_corners.Detector(_known(make(), 1.25)).detect(img)
    assert supplied.noise_sigma == pytest.approx(1.25)
    off = make().with_uncertainty(chess_corners.Uncertainty.off())
    quiet = chess_corners.Detector(off).detect(img)
    assert len(quiet) > 0
    assert quiet.noise_sigma is None


def test_ml_refiner_with_radon_detects_corners():
    img = _noisy_board()
    cfg = _radon().with_refiner(chess_corners.ImageRefiner.ml())
    det = chess_corners.Detector(cfg).detect(img)
    assert len(det) > 0
    assert np.isfinite(det.xy).all()
