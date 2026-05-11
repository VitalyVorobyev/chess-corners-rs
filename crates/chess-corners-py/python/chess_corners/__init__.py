"""Python-first public API for the chess_corners detector.

The config classes (`DetectorConfig`, `RefinerConfig`, …) and enums
(`ChessRing`, `RefinementMethod`, …) are native PyO3 types defined
in the compiled `_native` extension. They expose attribute access,
classmethod factories (`DetectorConfig.multiscale()`,
`Threshold.relative(...)`, `DetectionStrategy.radon(...)`), and
`to_dict` / `from_dict` / `to_json` / `from_json` helpers.

The detector itself is the :class:`Detector` PyClass — build once,
call :meth:`Detector.detect` repeatedly to amortise pyramid / upscale
buffer allocations across frames.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from . import _native
from ._native import (
    CenterOfMassConfig,
    DetectorConfig,
    ChessRing,
    ChessStrategy,
    ConfigError,
    DescriptorMode,
    DetectionStrategy,
    Detector,
    ForstnerConfig,
    MultiscaleParams,
    OrientationMethod,
    PeakFitMode,
    RadonPeakConfig,
    RadonStrategy,
    RefinementMethod,
    RefinerConfig,
    SaddlePointConfig,
    Threshold,
    UpscaleConfig,
    UpscaleMode,
)

# Backwards-compat alias for callers on 0.9.x. Will be removed in 0.11.0.
ChessConfig = DetectorConfig


def _print(self: Any, *, file: TextIO | None = None, indent: int = 2, sort_keys: bool = True) -> None:
    """Pretty-print a config object, using ``rich`` when available.

    Attached to every config class below so that
    ``ChessConfig().print()`` keeps working from earlier versions.
    """

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
    Threshold,
    MultiscaleParams,
    ChessStrategy,
    RadonStrategy,
    DetectionStrategy,
    RefinerConfig,
    UpscaleConfig,
    DetectorConfig,
):
    _cls.print = _print  # type: ignore[attr-defined]
    _cls.__rich_console__ = _rich_console  # type: ignore[attr-defined]


__all__ = [
    "CenterOfMassConfig",
    "ChessConfig",
    "ChessRing",
    "ChessStrategy",
    "ConfigError",
    "DescriptorMode",
    "DetectionStrategy",
    "DetectorConfig",
    "Detector",
    "ForstnerConfig",
    "MultiscaleParams",
    "OrientationMethod",
    "PeakFitMode",
    "RadonPeakConfig",
    "RadonStrategy",
    "RefinerConfig",
    "RefinementMethod",
    "SaddlePointConfig",
    "Threshold",
    "UpscaleConfig",
    "UpscaleMode",
]
