"""Config serialization roundtrip tests.

Covers JSON and dict roundtrips for DetectorConfig plus its nested sub-configs,
rejection of unknown keys, and JSON / dict equivalence.
"""
from __future__ import annotations

import json

import pytest

import chess_corners


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_to_comparable_dict(cfg: chess_corners.DetectorConfig) -> dict:
    """Convert a DetectorConfig to a plain Python dict for field-level comparison."""
    return json.loads(cfg.to_json())


# ---------------------------------------------------------------------------
# Default roundtrip
# ---------------------------------------------------------------------------


def test_default_config_json_roundtrip_fields_match():
    """Parsing and re-serialising the default config must not change any field."""
    cfg = chess_corners.DetectorConfig()
    json_str = cfg.to_json()
    cfg2 = chess_corners.DetectorConfig.from_json(json_str)

    original = _cfg_to_comparable_dict(cfg)
    restored = _cfg_to_comparable_dict(cfg2)
    assert original == restored, f"roundtrip changed fields: {set(original) ^ set(restored)}"


def test_default_config_dict_roundtrip_fields_match():
    """Dict roundtrip must be lossless."""
    cfg = chess_corners.DetectorConfig()
    d = cfg.to_dict()
    cfg2 = chess_corners.DetectorConfig.from_dict(d)

    assert _cfg_to_comparable_dict(cfg) == _cfg_to_comparable_dict(cfg2)


# ---------------------------------------------------------------------------
# Nested mutation survives roundtrip
# ---------------------------------------------------------------------------


def test_chess_refiner_forstner_max_offset_survives_roundtrip():
    """A mutation in a deeply nested config field must survive JSON roundtrip."""
    fcfg = chess_corners.ForstnerConfig()
    fcfg.max_offset = 3.5
    cfg = chess_corners.DetectorConfig().with_chess(
        refiner=chess_corners.ChessRefiner.forstner(fcfg)
    )

    json_str = cfg.to_json()
    cfg2 = chess_corners.DetectorConfig.from_json(json_str)

    assert cfg2.strategy.chess.refiner.kind == "forstner"
    assert abs(cfg2.strategy.chess.refiner.payload.max_offset - 3.5) < 1e-6


def test_radon_strategy_survives_roundtrip():
    """Radon strategy params must survive a full dict roundtrip."""
    cfg = chess_corners.DetectorConfig.radon()
    radon = cfg.strategy.radon
    radon.image_upsample = 2
    cfg.strategy = chess_corners.DetectionStrategy.from_radon(radon)

    d = cfg.to_dict()
    cfg2 = chess_corners.DetectorConfig.from_dict(d)

    assert cfg2.strategy.kind == "radon"
    assert cfg2.strategy.radon.image_upsample == 2


# ---------------------------------------------------------------------------
# Unknown key rejection
# ---------------------------------------------------------------------------


def test_from_json_rejects_unknown_top_level_key():
    """from_json with an unrecognised top-level key must raise ConfigError."""
    bad = json.dumps({"bogus_key_that_does_not_exist": 1})
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_json(bad)


def test_from_dict_rejects_unknown_top_level_key():
    """from_dict with an unrecognised key must raise ConfigError."""
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict({"bogus_key_that_does_not_exist": 1})


# ---------------------------------------------------------------------------
# JSON / dict equivalence
# ---------------------------------------------------------------------------


def test_json_and_dict_produce_equivalent_configs():
    """Configs constructed from JSON and from an equivalent dict must be identical."""
    cfg = chess_corners.DetectorConfig()
    cfg.threshold = 0.17
    chess = cfg.strategy.chess
    chess.refiner = chess_corners.ChessRefiner.forstner()
    cfg.strategy = chess_corners.DetectionStrategy.from_chess(chess)

    json_cfg = chess_corners.DetectorConfig.from_json(cfg.to_json())
    dict_cfg = chess_corners.DetectorConfig.from_dict(cfg.to_dict())

    assert _cfg_to_comparable_dict(json_cfg) == _cfg_to_comparable_dict(dict_cfg)


# ---------------------------------------------------------------------------
# Bare-string unit-variant round-trips (serde JSON parity)
# ---------------------------------------------------------------------------
# serde serialises unit variants of externally-tagged enums as bare strings,
# e.g. `MultiscaleConfig::SingleScale` → `"single_scale"`. The Python
# from_dict paths must accept this form so that Rust→JSON→Python round-trips
# work end-to-end. Note: Python to_dict emits the object form
# ({"single_scale": null}) — only the input side is widened here.


def test_multiscale_single_scale_bare_string_accepted_by_from_dict():
    """from_dict must accept the bare string 'single_scale' for MultiscaleConfig."""
    ms = chess_corners.MultiscaleConfig.from_dict("single_scale")
    assert ms.kind == "single_scale"


def test_upscale_disabled_bare_string_accepted_by_from_dict():
    """from_dict must accept the bare string 'disabled' for UpscaleConfig."""
    up = chess_corners.UpscaleConfig.from_dict("disabled")
    assert up.kind == "disabled"


def test_detector_config_from_dict_with_bare_string_multiscale():
    """DetectorConfig.from_dict accepts a dict where 'multiscale' is a bare string."""
    # This is the dict you would get from json.loads(serde_json_output) when
    # MultiscaleConfig is SingleScale.
    base = chess_corners.DetectorConfig().to_dict()
    base["multiscale"] = "single_scale"
    cfg = chess_corners.DetectorConfig.from_dict(base)
    assert cfg.multiscale.kind == "single_scale"


def test_detector_config_from_dict_with_bare_string_upscale():
    """DetectorConfig.from_dict accepts a dict where 'upscale' is a bare string."""
    base = chess_corners.DetectorConfig().to_dict()
    base["upscale"] = "disabled"
    cfg = chess_corners.DetectorConfig.from_dict(base)
    assert cfg.upscale.kind == "disabled"


def test_multiscale_pyramid_bare_string_raises_clear_error():
    """A bare 'pyramid' string (payload variant) must raise ConfigError with a hint."""
    with pytest.raises(chess_corners.ConfigError, match="pyramid"):
        chess_corners.MultiscaleConfig.from_dict("pyramid")


def test_upscale_fixed_bare_string_raises_clear_error():
    """A bare 'fixed' string (payload variant) must raise ConfigError with a hint."""
    with pytest.raises(chess_corners.ConfigError, match="fixed"):
        chess_corners.UpscaleConfig.from_dict("fixed")


def test_multiscale_unknown_bare_string_raises_clear_error():
    """An unrecognised bare string for MultiscaleConfig must raise ConfigError."""
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.MultiscaleConfig.from_dict("bogus_variant")
