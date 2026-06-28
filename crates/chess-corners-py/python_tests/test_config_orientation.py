"""Tests for OrientationMethod Python bindings.

Covers:
- Both variants round-trip through DetectorConfig.to_dict / from_dict.
- Both variants round-trip through DetectorConfig.to_json / from_json.
- Setting orientation_method changes the serialised value.
- Default orientation_method is RING_FIT.
- Unknown orientation_method string is rejected with ConfigError.
"""
from __future__ import annotations

import json

import pytest

chess_corners = pytest.importorskip("chess_corners")

OrientationMethod = chess_corners.OrientationMethod
DetectorConfig = chess_corners.DetectorConfig
ConfigError = chess_corners.ConfigError


# ---------------------------------------------------------------------------
# All variants present
# ---------------------------------------------------------------------------

ALL_VARIANTS = [
    (OrientationMethod.RING_FIT, "ring_fit"),
    (OrientationMethod.DISK_FIT, "disk_fit"),
]


@pytest.mark.parametrize("variant, key", ALL_VARIANTS)
def test_orientation_method_dict_roundtrip(variant, key):
    """Each OrientationMethod variant survives a to_dict / from_dict roundtrip."""
    cfg = DetectorConfig()
    cfg.orientation_method = variant

    d = cfg.to_dict()
    assert d["orientation_method"] == key, (
        f"to_dict serialised {variant!r} as {d['orientation_method']!r}, expected {key!r}"
    )

    cfg2 = DetectorConfig.from_dict(d)
    assert cfg2.orientation_method == variant, (
        f"from_dict did not restore {variant!r}: got {cfg2.orientation_method!r}"
    )


@pytest.mark.parametrize("variant, key", ALL_VARIANTS)
def test_orientation_method_json_roundtrip(variant, key):
    """Each OrientationMethod variant survives a to_json / from_json roundtrip."""
    cfg = DetectorConfig()
    cfg.orientation_method = variant

    json_str = cfg.to_json()
    parsed = json.loads(json_str)
    assert parsed["orientation_method"] == key, (
        f"to_json serialised {variant!r} as {parsed['orientation_method']!r}, expected {key!r}"
    )

    cfg2 = DetectorConfig.from_json(json_str)
    assert cfg2.orientation_method == variant, (
        f"from_json did not restore {variant!r}: got {cfg2.orientation_method!r}"
    )


# ---------------------------------------------------------------------------
# Default is RING_FIT
# ---------------------------------------------------------------------------


def test_default_orientation_method_is_ring_fit():
    cfg = DetectorConfig()
    assert cfg.orientation_method == OrientationMethod.RING_FIT, (
        f"default orientation_method is {cfg.orientation_method!r}, expected RING_FIT"
    )


def test_default_orientation_method_serialises_as_ring_fit():
    cfg = DetectorConfig()
    d = cfg.to_dict()
    assert d["orientation_method"] == "ring_fit", (
        f"default serialised as {d['orientation_method']!r}"
    )


# ---------------------------------------------------------------------------
# Observable change when set
# ---------------------------------------------------------------------------


def test_setting_orientation_method_changes_serialised_value():
    cfg = DetectorConfig()
    cfg.orientation_method = OrientationMethod.DISK_FIT

    d = cfg.to_dict()
    assert d["orientation_method"] == "disk_fit", (
        f"after assignment serialised as {d['orientation_method']!r}"
    )


def test_orientation_method_survives_other_field_mutations():
    cfg = DetectorConfig()
    cfg.orientation_method = OrientationMethod.DISK_FIT
    cfg.threshold = 0.25  # mutate unrelated field

    d = cfg.to_dict()
    assert d["orientation_method"] == "disk_fit"
    assert d["threshold"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Rejection of unknown string
# ---------------------------------------------------------------------------


def test_from_dict_rejects_unknown_orientation_method():
    d = DetectorConfig().to_dict()
    d["orientation_method"] = "not_a_real_method"
    with pytest.raises(ConfigError):
        DetectorConfig.from_dict(d)


def test_from_json_rejects_unknown_orientation_method():
    d = json.loads(DetectorConfig().to_json())
    d["orientation_method"] = "not_a_real_method"
    with pytest.raises(ConfigError):
        DetectorConfig.from_json(json.dumps(d))
