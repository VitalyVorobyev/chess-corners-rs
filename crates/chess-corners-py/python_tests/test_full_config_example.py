from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import chess_corners


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "run_with_full_config.py"
)
CODE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "run_with_code_config.py"
)


# A pure detector-config JSON (no CLI runtime keys) exercising every
# top-level branch of the 0.10.0 shape.
SAMPLE_CONFIG = {
    "strategy": {
        "chess": {
            "ring": "broad",
            "descriptor_ring": "canonical",
            "nms_radius": 3,
            "min_cluster_size": 1,
            "refiner": {
                "forstner": {
                    "radius": 2,
                    "min_trace": 25.0,
                    "min_det": 0.001,
                    "max_condition_number": 50.0,
                    "max_offset": 2.0,
                },
            },
        },
    },
    "threshold": {"absolute": 0.5},
    "multiscale": {
        "pyramid": {
            "levels": 3,
            "min_size": 96,
            "refinement_radius": 4,
        },
    },
    "upscale": {"disabled": None},
    "orientation_method": "ring_fit",
    "merge_radius": 2.5,
}


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

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(SAMPLE_CONFIG, tmp)
        tmp_path = Path(tmp.name)
    try:
        cfg = module.load_chess_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    assert cfg.strategy.kind == "chess"
    assert cfg.strategy.chess.ring is chess_corners.ChessRing.BROAD
    assert (
        cfg.strategy.chess.descriptor_ring
        is chess_corners.DescriptorRing.CANONICAL
    )
    assert cfg.threshold.kind == "absolute"
    assert cfg.threshold.value == 0.5
    assert cfg.strategy.chess.refiner.kind == "forstner"
    assert cfg.strategy.chess.refiner.payload.max_offset == 2.0
    ms = cfg.multiscale
    assert ms.kind == "pyramid"
    assert ms.levels == 3
    assert ms.min_size == 96
    assert ms.refinement_radius == 4
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
    assert cfg.strategy.kind == "chess"
    assert cfg.strategy.chess.ring is chess_corners.ChessRing.BROAD
    assert (
        cfg.strategy.chess.descriptor_ring
        is chess_corners.DescriptorRing.CANONICAL
    )
    assert cfg.threshold.kind == "absolute"
    assert cfg.threshold.value == 0.5
    assert cfg.strategy.chess.refiner.kind == "forstner"
    assert cfg.strategy.chess.refiner.payload.max_offset == 2.0
    ms = cfg.multiscale
    assert ms.kind == "pyramid"
    assert ms.levels == 3
    assert ms.min_size == 96


def test_public_package_exposes_py_typed():
    package_root = Path(chess_corners.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
