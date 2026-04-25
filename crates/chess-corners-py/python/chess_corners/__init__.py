"""Python-first public API for the chess_corners detector.

The config classes (`ChessConfig`, `RefinerConfig`, …) and enums
(`DetectorMode`, `RefinementMethod`, …) are native PyO3 types defined
in the compiled `_native` extension. They expose the same surface as
the prior pure-Python dataclasses — attribute access, classmethod
factories (`ChessConfig.multiscale()`), `to_dict` / `from_dict` /
`to_json` / `from_json`, identity-comparable enum members
(`cfg.detector_mode is DetectorMode.BROAD`) — but the FFI no longer
serializes through JSON: `find_chess_corners(image, cfg)` hands the
typed object directly to Rust.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from . import _native
from ._native import (
    CenterOfMassConfig,
    ChessConfig,
    ConfigError,
    DescriptorMode,
    DetectorMode,
    ForstnerConfig,
    PeakFitMode,
    RadonDetectorParams,
    RadonPeakConfig,
    RefinementMethod,
    RefinerConfig,
    SaddlePointConfig,
    ThresholdMode,
)


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
    RadonDetectorParams,
    RefinerConfig,
    ChessConfig,
):
    _cls.print = _print  # type: ignore[attr-defined]
    _cls.__rich_console__ = _rich_console  # type: ignore[attr-defined]


def find_chess_corners(image: Any, cfg: ChessConfig | None = None) -> Any:
    """Detect chessboard corners in a 2D C-contiguous uint8 NumPy image."""

    return _native.find_chess_corners(image, cfg)


def radon_heatmap(image: Any, cfg: ChessConfig | None = None) -> Any:
    """Compute the whole-image Radon detector heatmap.

    Returns a 2D ``float32`` NumPy array at *working resolution* — that
    is, ``height * upscale * radon_image_upsample`` rows by the same in
    columns. The working-to-input scale factor is
    ``cfg.upscale.factor * cfg.radon_detector.image_upsample`` (each
    clamped to its supported range).

    Parameters
    ----------
    image:
        2D C-contiguous ``uint8`` NumPy array of shape ``(H, W)``.
    cfg:
        Optional :class:`ChessConfig`. The ``upscale`` and
        ``radon_detector`` fields are honoured; other fields are
        ignored because no corner detection is performed.
    """

    return _native.radon_heatmap(image, cfg)


if hasattr(_native, "find_chess_corners_with_ml"):

    def find_chess_corners_with_ml(image: Any, cfg: ChessConfig | None = None) -> Any:
        """Detect chessboard corners using the ML-backed refiner pipeline."""

        return _native.find_chess_corners_with_ml(image, cfg)


__all__ = [
    "CenterOfMassConfig",
    "ChessConfig",
    "ConfigError",
    "DescriptorMode",
    "DetectorMode",
    "ForstnerConfig",
    "PeakFitMode",
    "RadonDetectorParams",
    "RadonPeakConfig",
    "RefinerConfig",
    "RefinementMethod",
    "SaddlePointConfig",
    "ThresholdMode",
    "find_chess_corners",
    "radon_heatmap",
]

if hasattr(_native, "find_chess_corners_with_ml"):
    __all__.append("find_chess_corners_with_ml")
