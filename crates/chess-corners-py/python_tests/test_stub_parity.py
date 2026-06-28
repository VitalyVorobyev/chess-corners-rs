"""Parity guard: every public callable on chess_corners.Detector at runtime
must have a matching ``def <name>`` entry in the shipped ``__init__.pyi``
stub.  Catches future stub drift without any third-party dependencies.
"""

import inspect
import re
from pathlib import Path

import chess_corners


def _runtime_detector_methods() -> set[str]:
    """Return the set of public callable names on Detector.

    Includes ``__init__``; excludes all other dunder names.
    """
    names: set[str] = set()
    for name in dir(chess_corners.Detector):
        if name.startswith("__") and name.endswith("__") and name != "__init__":
            continue
        try:
            attr = getattr(chess_corners.Detector, name)
        except AttributeError:
            continue
        if callable(attr):
            names.add(name)
    return names


def _stub_detector_method_names(pyi_text: str) -> set[str]:
    """Parse the ``class Detector:`` block in *pyi_text* and collect
    ``def <name>`` entries.
    """
    # Find the start of 'class Detector:'
    class_match = re.search(r"^class Detector:", pyi_text, re.MULTILINE)
    if class_match is None:
        raise AssertionError("class Detector: not found in .pyi stub")

    # Slice from that line to either the next top-level 'class ' or EOF
    block_start = class_match.start()
    next_class = re.search(r"^class ", pyi_text[class_match.end():], re.MULTILINE)
    if next_class is not None:
        block_end = class_match.end() + next_class.start()
    else:
        block_end = len(pyi_text)

    detector_block = pyi_text[block_start:block_end]
    return set(re.findall(r"^\s+def (\w+)", detector_block, re.MULTILINE))


def _locate_pyi() -> Path:
    """Return the path to chess_corners/__init__.pyi.

    Tries the installed location first (sibling of the .py/__init__), then
    falls back to the in-tree source path.
    """
    # Installed location: same directory as __init__.py / __init__.pyi
    pkg_file = Path(chess_corners.__file__)
    candidate = pkg_file.with_name("__init__.pyi")
    if candidate.exists():
        return candidate

    # Source tree fallback: crates/chess-corners-py/python/chess_corners/__init__.pyi
    # Walk up from this test file to find the repo root.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        source = (
            ancestor
            / "crates"
            / "chess-corners-py"
            / "python"
            / "chess_corners"
            / "__init__.pyi"
        )
        if source.exists():
            return source

    raise FileNotFoundError(
        "__init__.pyi not found next to the installed package or in the "
        "source tree. Run `maturin develop` to install the package first."
    )


def test_detector_stub_covers_all_runtime_methods() -> None:
    """Every public callable on the runtime Detector class must appear in the
    stub's ``class Detector:`` block as a ``def <name>`` entry.
    """
    pyi_path = _locate_pyi()
    pyi_text = pyi_path.read_text(encoding="utf-8")

    runtime_methods = _runtime_detector_methods()
    stub_methods = _stub_detector_method_names(pyi_text)

    missing = runtime_methods - stub_methods
    assert not missing, (
        f"Runtime Detector methods not documented in {pyi_path}:\n"
        + "\n".join(f"  - {m}" for m in sorted(missing))
        + "\nAdd the missing method(s) to the 'class Detector:' block in "
        "__init__.pyi to keep the stub in sync."
    )
