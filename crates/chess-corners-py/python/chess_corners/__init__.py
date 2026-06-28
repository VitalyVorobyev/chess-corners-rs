"""Python-first public API for the chess_corners detector.

The config classes (`DetectorConfig`, `ChessConfig`, `RadonConfig`,
`ChessRefiner`, `RadonRefiner`, ...) and enums (`ChessRing`,
`PeakFitMode`, `OrientationMethod`) are native PyO3
types defined in the compiled `_native` extension. They expose
attribute access, classmethod factories
(`DetectorConfig.multiscale()`, `Threshold.relative(...)`,
`MultiscaleConfig.pyramid(...)`, `ChessRefiner.forstner(cfg)`), and
`to_dict` / `from_dict` / `to_json` / `from_json` helpers.

The detector itself is the :class:`Detector` PyClass - build once,
call :meth:`Detector.detect` repeatedly to amortise pyramid / upscale
buffer allocations across frames.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from . import _native
from ._native import (
    CenterOfMassConfig,
    ChessConfig,
    ChessRefiner,
    ChessRing,
    ConfigError,
    DetectionParams,
    DetectionStrategy,
    Detector,
    DetectorConfig,
    ForstnerConfig,
    MultiscaleConfig,
    OrientationMethod,
    PeakFitMode,
    RadonConfig,
    RadonPeakConfig,
    RadonRefiner,
    SaddlePointConfig,
    Threshold,
    UpscaleConfig,
)


def _print(self: Any, *, file: TextIO | None = None, indent: int = 2, sort_keys: bool = True) -> None:
    """Pretty-print a config object, using ``rich`` when available."""

    if file is None:
        try:
            from rich.console import Console
        except ImportError:
            pass
        else:
            Console().print(self)
            return
    print(self.pretty(indent=indent, sort_keys=sort_keys), file=file or sys.stdout)


def _rich_console(self: Any, console: Any, options: Any) -> Any:
    try:
        from rich.panel import Panel
        from rich.pretty import Pretty
    except ImportError:
        yield self.pretty()
        return
    yield Panel(Pretty(self.to_dict(), expand_all=True), title=type(self).__name__)


for _cls in (
    CenterOfMassConfig,
    ForstnerConfig,
    SaddlePointConfig,
    RadonPeakConfig,
    DetectionParams,
    Threshold,
    MultiscaleConfig,
    UpscaleConfig,
    ChessRefiner,
    RadonRefiner,
    ChessConfig,
    RadonConfig,
    DetectionStrategy,
    DetectorConfig,
):
    _cls.print = _print  # type: ignore[attr-defined]
    _cls.__rich_console__ = _rich_console  # type: ignore[attr-defined]


__all__ = [
    "CenterOfMassConfig",
    "ChessConfig",
    "ChessRefiner",
    "ChessRing",
    "ConfigError",
    "DetectionParams",
    "DetectionStrategy",
    "Detector",
    "DetectorConfig",
    "ForstnerConfig",
    "MultiscaleConfig",
    "OrientationMethod",
    "PeakFitMode",
    "RadonConfig",
    "RadonPeakConfig",
    "RadonRefiner",
    "SaddlePointConfig",
    "Threshold",
    "UpscaleConfig",
]
