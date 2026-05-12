"""Round-trip tests for UpscaleConfig and DetectorConfig.upscale.

Skipped when the chess_corners native extension is not built.
"""

import pytest

chess_corners = pytest.importorskip("chess_corners")

UpscaleConfig = chess_corners.UpscaleConfig
DetectorConfig = chess_corners.DetectorConfig


def test_disabled_factory():
    cfg = UpscaleConfig.disabled()
    assert cfg.kind == "disabled"
    with pytest.raises(AttributeError):
        _ = cfg.factor


def test_fixed_factory():
    cfg = UpscaleConfig.fixed(2)
    assert cfg.kind == "fixed"
    assert cfg.factor == 2


def test_to_dict_round_trip_disabled():
    cfg = UpscaleConfig.disabled()
    d = cfg.to_dict()
    assert d == {"disabled": None}
    cfg2 = UpscaleConfig.from_dict(d)
    assert cfg2.kind == "disabled"


def test_to_dict_round_trip_fixed():
    cfg = UpscaleConfig.fixed(3)
    d = cfg.to_dict()
    assert d == {"fixed": 3}
    cfg2 = UpscaleConfig.from_dict(d)
    assert cfg2.kind == "fixed"
    assert cfg2.factor == 3


def test_chess_config_upscale_attribute():
    cfg = DetectorConfig()
    # Default should be disabled.
    assert cfg.upscale.kind == "disabled"


def test_chess_config_set_upscale():
    cfg = DetectorConfig()
    cfg.upscale = UpscaleConfig.fixed(2)
    assert cfg.upscale.kind == "fixed"
    assert cfg.upscale.factor == 2


def test_chess_config_to_dict_includes_upscale():
    cfg = DetectorConfig()
    cfg.upscale = UpscaleConfig.fixed(4)
    d = cfg.to_dict()
    assert "upscale" in d
    assert d["upscale"] == {"fixed": 4}


def test_chess_config_round_trip_with_upscale():
    cfg = DetectorConfig()
    cfg.upscale = UpscaleConfig.fixed(2)
    d = cfg.to_dict()
    cfg2 = DetectorConfig.from_dict(d)
    assert cfg2.upscale.kind == "fixed"
    assert cfg2.upscale.factor == 2


def test_chess_config_json_round_trip_with_upscale():
    cfg = DetectorConfig()
    cfg.upscale = UpscaleConfig.fixed(3)
    json_str = cfg.to_json()
    cfg2 = DetectorConfig.from_json(json_str)
    assert cfg2.upscale.kind == "fixed"
    assert cfg2.upscale.factor == 3


def test_upscale_rejects_both_variants():
    with pytest.raises(chess_corners.ConfigError):
        UpscaleConfig.from_dict({"disabled": None, "fixed": 2})


def test_upscale_rejects_neither_variant():
    with pytest.raises(chess_corners.ConfigError):
        UpscaleConfig.from_dict({})
