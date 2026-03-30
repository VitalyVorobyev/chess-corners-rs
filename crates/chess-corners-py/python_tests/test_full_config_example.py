from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import chess_corners


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "run_with_full_config.py"
)
CODE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "run_with_code_config.py"
)
CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "chess_algorithm_config_example.json"
)


def _load_example_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_config_example_parser_is_lazy_about_pillow():
    before = {name for name in sys.modules if name == "PIL" or name.startswith("PIL.")}
    module = _load_example_module(EXAMPLE_PATH)
    after = {name for name in sys.modules if name == "PIL" or name.startswith("PIL.")}

    assert before == after

    cfg = module.load_chess_config(CONFIG_PATH)
    assert cfg.detector_mode is chess_corners.DetectorMode.BROAD
    assert cfg.descriptor_mode is chess_corners.DescriptorMode.CANONICAL
    assert cfg.threshold_mode is chess_corners.ThresholdMode.ABSOLUTE
    assert cfg.threshold_value == 0.5
    assert cfg.refiner.kind is chess_corners.RefinementMethod.FORSTNER
    assert cfg.refiner.forstner.max_offset == 2.0
    assert cfg.pyramid_levels == 3
    assert cfg.pyramid_min_size == 96
    assert cfg.refinement_radius == 4
    assert cfg.merge_radius == 2.5


def test_full_config_example_rejects_unknown_keys():
    module = _load_example_module(EXAMPLE_PATH)

    try:
        module.build_chess_config({"params": {"unexpected": 1}})
    except module.ConfigError as exc:
        assert "unknown keys" in str(exc)
    else:
        raise AssertionError("expected ConfigError for unknown config keys")


def test_code_config_example_is_lazy_about_pillow_and_builds_full_config():
    before = {name for name in sys.modules if name == "PIL" or name.startswith("PIL.")}
    module = _load_example_module(CODE_EXAMPLE_PATH)
    after = {name for name in sys.modules if name == "PIL" or name.startswith("PIL.")}

    assert before == after

    cfg = module.build_chess_config()
    assert cfg.detector_mode is chess_corners.DetectorMode.BROAD
    assert cfg.descriptor_mode is chess_corners.DescriptorMode.CANONICAL
    assert cfg.threshold_mode is chess_corners.ThresholdMode.ABSOLUTE
    assert cfg.threshold_value == 0.5
    assert cfg.refiner.kind is chess_corners.RefinementMethod.FORSTNER
    assert cfg.refiner.center_of_mass.radius == 2
    assert cfg.refiner.forstner.max_offset == 2.0
    assert cfg.refiner.saddle_point.max_offset == 1.75
    assert cfg.pyramid_levels == 3
    assert cfg.pyramid_min_size == 96


def test_public_package_exposes_py_typed():
    package_root = Path(chess_corners.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
