"""Config serialization roundtrip tests.

Covers JSON and dict roundtrips for ChessConfig plus its nested sub-configs,
rejection of unknown keys, and JSON ↔ dict equivalence.
"""
from __future__ import annotations

import json

import pytest

import chess_corners


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_to_comparable_dict(cfg: chess_corners.ChessConfig) -> dict:
    """Convert a ChessConfig to a plain Python dict for field-level comparison."""
    return json.loads(cfg.to_json())


# ---------------------------------------------------------------------------
# Default roundtrip
# ---------------------------------------------------------------------------


def test_default_config_json_roundtrip_fields_match():
    """Parsing and re-serialising the default config must not change any field."""
    cfg = chess_corners.ChessConfig()
    json_str = cfg.to_json()
    cfg2 = chess_corners.ChessConfig.from_json(json_str)

    original = _cfg_to_comparable_dict(cfg)
    restored = _cfg_to_comparable_dict(cfg2)
    assert original == restored, f"roundtrip changed fields: {set(original) ^ set(restored)}"


def test_default_config_dict_roundtrip_fields_match():
    """Dict roundtrip must be lossless."""
    cfg = chess_corners.ChessConfig()
    d = cfg.to_dict()
    cfg2 = chess_corners.ChessConfig.from_dict(d)

    assert _cfg_to_comparable_dict(cfg) == _cfg_to_comparable_dict(cfg2)


# ---------------------------------------------------------------------------
# Nested mutation survives roundtrip
# ---------------------------------------------------------------------------


def test_nested_forstner_max_offset_survives_roundtrip():
    """A mutation in a deeply nested config field must survive JSON roundtrip."""
    cfg = chess_corners.ChessConfig()
    cfg.refiner.forstner.max_offset = 3.5

    json_str = cfg.to_json()
    cfg2 = chess_corners.ChessConfig.from_json(json_str)

    assert abs(cfg2.refiner.forstner.max_offset - 3.5) < 1e-6, (
        f"forstner.max_offset not preserved: got {cfg2.refiner.forstner.max_offset}"
    )


def test_nested_radon_detector_survives_roundtrip():
    """Radon detector params must survive a full dict roundtrip."""
    cfg = chess_corners.ChessConfig()
    cfg.radon_detector.image_upsample = 2

    d = cfg.to_dict()
    cfg2 = chess_corners.ChessConfig.from_dict(d)

    assert cfg2.radon_detector.image_upsample == 2, (
        f"image_upsample not preserved: got {cfg2.radon_detector.image_upsample}"
    )


# ---------------------------------------------------------------------------
# Unknown key rejection
# ---------------------------------------------------------------------------


def test_from_json_rejects_unknown_top_level_key():
    """from_json with an unrecognised top-level key must raise ConfigError."""
    bad = json.dumps({"bogus_key_that_does_not_exist": 1})
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.ChessConfig.from_json(bad)


def test_from_dict_rejects_unknown_top_level_key():
    """from_dict with an unrecognised key must raise ConfigError."""
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.ChessConfig.from_dict({"bogus_key_that_does_not_exist": 1})


# ---------------------------------------------------------------------------
# JSON ↔ dict equivalence
# ---------------------------------------------------------------------------


def test_json_and_dict_produce_equivalent_configs():
    """Configs constructed from JSON and from an equivalent dict must be identical."""
    cfg = chess_corners.ChessConfig()
    cfg.threshold_value = 0.17
    cfg.refiner.kind = chess_corners.RefinementMethod.FORSTNER

    json_cfg = chess_corners.ChessConfig.from_json(cfg.to_json())
    dict_cfg = chess_corners.ChessConfig.from_dict(cfg.to_dict())

    assert _cfg_to_comparable_dict(json_cfg) == _cfg_to_comparable_dict(dict_cfg)
