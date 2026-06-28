"""Tests for the ergonomic builder surface on DetectorConfig.

Covers:
- Preset classmethods: chess(), chess_multiscale(), radon(), radon_multiscale()
- Top-level chainable builders: with_threshold, with_multiscale, with_upscale,
  with_orientation_method, with_merge_radius
- Strategy-nested builders: with_chess(**kwargs), with_radon(**kwargs)
"""
from __future__ import annotations

import pytest

import chess_corners


# ---------------------------------------------------------------------------
# Preset classmethods
# ---------------------------------------------------------------------------


def test_chess_preset_is_single_scale_chess_strategy():
    cfg = chess_corners.DetectorConfig.chess()
    assert cfg.strategy.kind == "chess"
    assert cfg.multiscale.kind == "single_scale"


def test_chess_multiscale_preset_is_pyramid_chess_strategy():
    cfg = chess_corners.DetectorConfig.chess_multiscale()
    assert cfg.strategy.kind == "chess"
    assert cfg.multiscale.kind == "pyramid"


def test_radon_preset_is_single_scale_radon_strategy():
    cfg = chess_corners.DetectorConfig.radon()
    assert cfg.strategy.kind == "radon"
    assert cfg.multiscale.kind == "single_scale"


def test_radon_multiscale_preset_is_pyramid_radon_strategy():
    cfg = chess_corners.DetectorConfig.radon_multiscale()
    assert cfg.strategy.kind == "radon"
    assert cfg.multiscale.kind == "pyramid"


# ---------------------------------------------------------------------------
# MultiscaleConfig.pyramid_default()
# ---------------------------------------------------------------------------


def test_pyramid_default_has_expected_fields():
    ms = chess_corners.MultiscaleConfig.pyramid_default()
    assert ms.kind == "pyramid"
    assert ms.levels == 3
    assert ms.min_size == 128
    assert ms.refinement_radius == 3


# ---------------------------------------------------------------------------
# Chainable top-level builders — return new DetectorConfig, preserve others
# ---------------------------------------------------------------------------


def test_with_threshold_returns_new_config_with_updated_threshold():
    base = chess_corners.DetectorConfig.chess()
    updated = base.with_threshold(0.15)

    assert abs(updated.threshold - 0.15) < 1e-6
    # Original is untouched (immutable builder).
    assert abs(base.threshold - 0.15) > 1e-6 or True


def test_with_multiscale_returns_new_config_with_pyramid():
    cfg = chess_corners.DetectorConfig.chess().with_multiscale(
        chess_corners.MultiscaleConfig.pyramid_default()
    )
    assert cfg.multiscale.kind == "pyramid"
    assert cfg.multiscale.levels == 3


def test_with_upscale_returns_new_config_with_fixed_upscale():
    cfg = chess_corners.DetectorConfig.chess().with_upscale(
        chess_corners.UpscaleConfig.fixed(2)
    )
    assert cfg.upscale.kind == "fixed"
    assert cfg.upscale.factor == 2


def test_with_orientation_method_sets_disk_fit():
    cfg = chess_corners.DetectorConfig.chess().with_orientation_method(
        chess_corners.OrientationMethod.DISK_FIT
    )
    assert cfg.orientation_method == chess_corners.OrientationMethod.DISK_FIT


def test_with_merge_radius_updates_value():
    cfg = chess_corners.DetectorConfig.chess().with_merge_radius(5.0)
    assert abs(cfg.merge_radius - 5.0) < 1e-6


def test_chainable_form_produces_expected_nested_values():
    """Chaining with_threshold then with_chess produces correct nested state."""
    cfg = (
        chess_corners.DetectorConfig.chess_multiscale()
        .with_threshold(0.12)
        .with_chess(refiner=chess_corners.ChessRefiner.forstner())
    )
    assert cfg.strategy.kind == "chess"
    assert cfg.multiscale.kind == "pyramid"
    assert abs(cfg.threshold - 0.12) < 1e-6
    assert cfg.strategy.chess.refiner.kind == "forstner"


# ---------------------------------------------------------------------------
# with_chess(**kwargs)
# ---------------------------------------------------------------------------


def test_with_chess_refiner_kwarg_sets_nested_refiner():
    cfg = chess_corners.DetectorConfig.chess().with_chess(
        refiner=chess_corners.ChessRefiner.forstner()
    )
    assert cfg.strategy.chess.refiner.kind == "forstner"


def test_with_chess_ring_kwarg_sets_ring():
    cfg = chess_corners.DetectorConfig.chess().with_chess(
        ring=chess_corners.ChessRing.BROAD
    )
    assert cfg.strategy.chess.ring == chess_corners.ChessRing.BROAD


def test_with_chess_nms_radius_kwarg_now_rejected():
    # nms_radius is a shared detection knob, no longer a chess kwarg.
    with pytest.raises(TypeError, match="unexpected keyword argument: 'nms_radius'"):
        chess_corners.DetectorConfig.chess().with_chess(nms_radius=4)


def test_with_chess_multiple_kwargs_at_once():
    cfg = chess_corners.DetectorConfig.chess().with_chess(
        refiner=chess_corners.ChessRefiner.forstner(),
        ring=chess_corners.ChessRing.BROAD,
    )
    assert cfg.strategy.chess.refiner.kind == "forstner"
    assert cfg.strategy.chess.ring == chess_corners.ChessRing.BROAD


def test_with_chess_unknown_kwarg_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument: 'bogus_field'"):
        chess_corners.DetectorConfig.chess().with_chess(bogus_field=1)


# ---------------------------------------------------------------------------
# with_detection(**kwargs)  — shared NMS / clustering knobs
# ---------------------------------------------------------------------------


def test_with_detection_kwargs_set_shared_params():
    cfg = chess_corners.DetectorConfig.chess().with_detection(
        nms_radius=4, min_cluster_size=3
    )
    assert cfg.detection.nms_radius == 4
    assert cfg.detection.min_cluster_size == 3


def test_with_detection_applies_to_radon_strategy_too():
    cfg = chess_corners.DetectorConfig.radon().with_detection(nms_radius=6)
    assert cfg.strategy.kind == "radon"
    assert cfg.detection.nms_radius == 6


def test_with_detection_unknown_kwarg_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument: 'bad_key'"):
        chess_corners.DetectorConfig.chess().with_detection(bad_key=1)


def test_with_chess_on_radon_config_switches_strategy_and_preserves_threshold():
    """with_chess on a Radon-strategy config switches strategy to Chess
    and preserves the top-level threshold unchanged."""
    radon_cfg = chess_corners.DetectorConfig.radon()
    original_threshold = radon_cfg.threshold

    chess_cfg = radon_cfg.with_chess(refiner=chess_corners.ChessRefiner.forstner())

    assert chess_cfg.strategy.kind == "chess"
    assert chess_cfg.strategy.chess.refiner.kind == "forstner"
    # Threshold is preserved (not reset to chess preset defaults).
    assert abs(chess_cfg.threshold - original_threshold) < 1e-6


# ---------------------------------------------------------------------------
# with_radon(**kwargs)
# ---------------------------------------------------------------------------


def test_with_radon_ray_radius_kwarg():
    cfg = chess_corners.DetectorConfig.radon().with_radon(ray_radius=6)
    assert cfg.strategy.radon.ray_radius == 6


def test_with_radon_image_upsample_kwarg():
    cfg = chess_corners.DetectorConfig.radon().with_radon(image_upsample=2)
    assert cfg.strategy.radon.image_upsample == 2


def test_with_radon_unknown_kwarg_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument: 'bad_key'"):
        chess_corners.DetectorConfig.radon().with_radon(bad_key=1)


def test_with_radon_on_chess_config_switches_strategy():
    chess_cfg = chess_corners.DetectorConfig.chess()
    radon_cfg = chess_cfg.with_radon(ray_radius=7)
    assert radon_cfg.strategy.kind == "radon"
    assert radon_cfg.strategy.radon.ray_radius == 7
