"""Round-trip tests for UpscaleConfig and ChessConfig.upscale.

Skipped when the chess_corners native extension is not built.
"""

import pytest

chess_corners = pytest.importorskip("chess_corners")

UpscaleConfig = chess_corners.UpscaleConfig
UpscaleMode = chess_corners.UpscaleMode
ChessConfig = chess_corners.ChessConfig


def test_disabled_factory():
    cfg = UpscaleConfig.disabled()
    assert cfg.mode == UpscaleMode.DISABLED
    assert cfg.factor == 1


def test_fixed_factory():
    cfg = UpscaleConfig.fixed(2)
    assert cfg.mode == UpscaleMode.FIXED
    assert cfg.factor == 2


def test_to_dict_round_trip_disabled():
    cfg = UpscaleConfig.disabled()
    d = cfg.to_dict()
    assert d["mode"] == "disabled"
    cfg2 = UpscaleConfig.from_dict(d)
    assert cfg2.mode == UpscaleMode.DISABLED
    assert cfg2.factor == cfg.factor


def test_to_dict_round_trip_fixed():
    cfg = UpscaleConfig.fixed(3)
    d = cfg.to_dict()
    assert d["mode"] == "fixed"
    assert d["factor"] == 3
    cfg2 = UpscaleConfig.from_dict(d)
    assert cfg2.mode == UpscaleMode.FIXED
    assert cfg2.factor == 3


def test_chess_config_upscale_attribute():
    cfg = ChessConfig()
    # Default should be disabled.
    assert cfg.upscale.mode == UpscaleMode.DISABLED


def test_chess_config_set_upscale():
    cfg = ChessConfig()
    cfg.upscale = UpscaleConfig.fixed(2)
    assert cfg.upscale.mode == UpscaleMode.FIXED
    assert cfg.upscale.factor == 2


def test_chess_config_to_dict_includes_upscale():
    cfg = ChessConfig()
    cfg.upscale = UpscaleConfig.fixed(4)
    d = cfg.to_dict()
    assert "upscale" in d
    assert d["upscale"]["mode"] == "fixed"
    assert d["upscale"]["factor"] == 4


def test_chess_config_round_trip_with_upscale():
    cfg = ChessConfig()
    cfg.upscale = UpscaleConfig.fixed(2)
    d = cfg.to_dict()
    cfg2 = ChessConfig.from_dict(d)
    assert cfg2.upscale.mode == UpscaleMode.FIXED
    assert cfg2.upscale.factor == 2


def test_chess_config_json_round_trip_with_upscale():
    cfg = ChessConfig()
    cfg.upscale = UpscaleConfig.fixed(3)
    json_str = cfg.to_json()
    cfg2 = ChessConfig.from_json(json_str)
    assert cfg2.upscale.mode == UpscaleMode.FIXED
    assert cfg2.upscale.factor == 3
