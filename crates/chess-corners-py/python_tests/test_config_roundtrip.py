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


def test_image_refiner_saddle_point_max_offset_survives_roundtrip():
    """A mutation in a deeply nested config field must survive JSON roundtrip."""
    scfg = chess_corners.SaddlePointConfig()
    scfg.max_offset = 3.5
    cfg = chess_corners.DetectorConfig().with_refiner(
        chess_corners.ImageRefiner.saddle_point(scfg)
    )

    json_str = cfg.to_json()
    cfg2 = chess_corners.DetectorConfig.from_json(json_str)

    assert cfg2.refiner.kind == "saddle_point"
    assert abs(cfg2.refiner.payload.max_offset - 3.5) < 1e-6


def test_on_refiner_reject_survives_roundtrip_and_defaults_to_drop():
    cfg = chess_corners.DetectorConfig.chess()
    assert cfg.strategy.chess.on_refiner_reject == chess_corners.OnRefinerReject.DROP
    assert cfg.to_dict()["strategy"]["chess"]["on_refiner_reject"] == "drop"

    keep = cfg.with_refiner(chess_corners.ImageRefiner.saddle_point()).with_chess(
        on_refiner_reject=chess_corners.OnRefinerReject.KEEP_CENTER_OF_MASS
    )
    data = keep.to_dict()
    assert data["strategy"]["chess"]["on_refiner_reject"] == "keep_center_of_mass"
    restored = chess_corners.DetectorConfig.from_dict(data)
    assert (
        restored.strategy.chess.on_refiner_reject
        == chess_corners.OnRefinerReject.KEEP_CENTER_OF_MASS
    )
    assert _cfg_to_comparable_dict(restored) == _cfg_to_comparable_dict(keep)

    bad = json.loads(keep.to_json())
    bad["strategy"]["chess"]["on_refiner_reject"] = "keep"
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict(bad)


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


@pytest.mark.parametrize(
    "data, key",
    [
        ({"noise_sigma": 1.5}, "noise_sigma"),
        ({"strategy": {"chess": {"refiner": "center_of_mass"}}}, "refiner"),
        ({"strategy": {"radon": {"ray_radus": 5}}}, "ray_radus"),
        ({"strategy": {"chess": {"center_of_mass": {"radius": 2, "r": 1}}}}, "r"),
        ({"detection": {"nms": 3}}, "nms"),
        (
            {"multiscale": {"pyramid": {"levels": 3, "min_size": 128, "refinement_radius": 3, "x": 1}}},
            "x",
        ),
        ({"upscale": {"fixed": 2, "x": 1}}, "x"),
        ({"refiner": {"saddle_point": {"radius_px": 2}}}, "radius_px"),
        ({"uncertainty": {"propagated": {"noise": "estimate", "sigma": 1.0}}}, "sigma"),
    ],
)
def test_stale_keys_are_rejected_and_named_at_every_level(data, key):
    """A stale or misspelt key fails at any depth, through both entry points."""
    for load in (
        chess_corners.DetectorConfig.from_dict,
        lambda d: chess_corners.DetectorConfig.from_json(json.dumps(d)),
    ):
        with pytest.raises(chess_corners.ConfigError, match=rf"(unknown keys: {key}$|^{key} moved)"):
            load(data)


# ---------------------------------------------------------------------------
# JSON / dict equivalence
# ---------------------------------------------------------------------------


def test_json_and_dict_produce_equivalent_configs():
    """Configs constructed from JSON and from an equivalent dict must be identical."""
    cfg = chess_corners.DetectorConfig()
    cfg.threshold = 0.17
    cfg.refiner = chess_corners.ImageRefiner.saddle_point()

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


# ---------------------------------------------------------------------------
# ImageRefiner
# ---------------------------------------------------------------------------


def test_refiner_defaults_to_null_and_round_trips_both_variants():
    cfg = chess_corners.DetectorConfig.radon()
    assert json.loads(cfg.to_json())["refiner"] is None
    assert chess_corners.DetectorConfig.from_json(cfg.to_json()).refiner is None
    # An absent key also means no refiner.
    assert chess_corners.DetectorConfig.from_dict({}).refiner is None

    sp = chess_corners.SaddlePointConfig()
    sp.max_offset = 0.75
    cfg = cfg.with_refiner(chess_corners.ImageRefiner.saddle_point(sp))
    assert list(json.loads(cfg.to_json())["refiner"]) == ["saddle_point"]
    back = chess_corners.DetectorConfig.from_json(cfg.to_json())
    assert back.refiner.kind == "saddle_point"
    assert back.refiner.payload.max_offset == pytest.approx(0.75)
    assert chess_corners.DetectorConfig.from_dict(cfg.to_dict()).refiner == cfg.refiner


def test_image_refiner_dict_shapes():
    sp = chess_corners.ImageRefiner.saddle_point()
    assert chess_corners.ImageRefiner.from_dict(sp.to_dict()) == sp
    assert chess_corners.ImageRefiner.from_dict({"saddle_point": {}}) == sp
    assert repr(sp).startswith("ImageRefiner.saddle_point(")


@pytest.mark.parametrize(
    "bad",
    [
        "saddle_point",
        "center_of_mass",
        "peak_fit",
        "bogus",
        "ml",
        {"center_of_mass": {}},
        {"peak_fit": None},
        {"ml": {}},
        {"ml": None, "saddle_point": {}},
        {"saddle_point": {"unknown": 1}},
        {},
        3,
    ],
)
def test_image_refiner_rejects_unknown_shapes(bad):
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.ImageRefiner.from_dict(bad)


def test_old_per_strategy_refiner_keys_are_rejected():
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict(
            {"strategy": {"chess": {"refiner": {"saddle_point": {}}}}}
        )
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.DetectorConfig.from_dict(
            {"strategy": {"radon": {"refiner": "peak_fit"}}}
        )
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.ChessConfig.from_dict({"refiner": {"saddle_point": {}}})
    with pytest.raises(chess_corners.ConfigError):
        chess_corners.RadonConfig.from_dict({"refiner": "peak_fit"})
