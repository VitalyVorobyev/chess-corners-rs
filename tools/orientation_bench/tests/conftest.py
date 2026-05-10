"""Path bootstrapping for the orientation_bench test suite.

The benchmark lives under `tools/orientation_bench/`, which is not on
`sys.path` by default. We add `tools/` so the package is importable as
``orientation_bench`` when pytest is invoked from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
