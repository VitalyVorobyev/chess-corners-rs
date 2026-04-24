"""Python-first public API for the chess_corners detector."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

from . import _native


JsonDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a config dictionary or JSON document is invalid."""


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be an object")
    return value


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"{path} has unknown keys: {joined}")


def _expect_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    return value


def _expect_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _expect_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    return value


def _enum_value(enum_type: type[Enum], value: Any, path: str) -> Enum:
    raw = _expect_string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ConfigError(f"{path} must be one of: {allowed}") from exc


class _PrettyMixin:
    """Shared JSON/pretty-print helpers for public config objects."""

    def to_dict(self) -> JsonDict:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = None, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    def pretty(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return self.to_json(indent=indent, sort_keys=sort_keys)

    def print(
        self,
        *,
        file: TextIO | None = None,
        indent: int = 2,
        sort_keys: bool = True,
    ) -> None:
        if file is None:
            try:
                from rich.console import Console
            except ImportError:
                pass
            else:
                Console().print(self)
                return
        print(self.pretty(indent=indent, sort_keys=sort_keys), file=file or sys.stdout)

    def __str__(self) -> str:
        return self.pretty()

    def __rich_console__(self, console: Any, options: Any) -> Any:
        try:
            from rich.panel import Panel
            from rich.pretty import Pretty
        except ImportError:
            yield self.pretty()
            return

        yield Panel(Pretty(self.to_dict(), expand_all=True), title=type(self).__name__)


class DetectorMode(str, Enum):
    """High-level detector mode: ChESS variants or whole-image Radon."""

    CANONICAL = "canonical"
    BROAD = "broad"
    RADON = "radon"


class DescriptorMode(str, Enum):
    """High-level descriptor mode for orientation/response sampling."""

    FOLLOW_DETECTOR = "follow_detector"
    CANONICAL = "canonical"
    BROAD = "broad"


class ThresholdMode(str, Enum):
    """Threshold interpretation for candidate selection."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class RefinementMethod(str, Enum):
    """Subpixel refinement algorithm applied to each candidate."""

    CENTER_OF_MASS = "center_of_mass"
    FORSTNER = "forstner"
    SADDLE_POINT = "saddle_point"
    RADON_PEAK = "radon_peak"


@dataclass
class CenterOfMassConfig(_PrettyMixin):
    """Classic center-of-mass refiner parameters."""

    radius: int = 2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CenterOfMassConfig:
        mapping = _expect_mapping(data, "center_of_mass")
        _reject_unknown_keys(mapping, {"radius"}, "center_of_mass")
        cfg = cls()
        if "radius" in mapping:
            cfg.radius = _expect_int(mapping["radius"], "center_of_mass.radius")
        return cfg

    def to_dict(self) -> JsonDict:
        return {"radius": self.radius}


@dataclass
class ForstnerConfig(_PrettyMixin):
    """Förstner refiner parameters."""

    radius: int = 2
    min_trace: float = 25.0
    min_det: float = 1e-3
    max_condition_number: float = 50.0
    max_offset: float = 1.5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForstnerConfig:
        mapping = _expect_mapping(data, "forstner")
        _reject_unknown_keys(
            mapping,
            {
                "radius",
                "min_trace",
                "min_det",
                "max_condition_number",
                "max_offset",
            },
            "forstner",
        )
        cfg = cls()
        if "radius" in mapping:
            cfg.radius = _expect_int(mapping["radius"], "forstner.radius")
        if "min_trace" in mapping:
            cfg.min_trace = _expect_float(mapping["min_trace"], "forstner.min_trace")
        if "min_det" in mapping:
            cfg.min_det = _expect_float(mapping["min_det"], "forstner.min_det")
        if "max_condition_number" in mapping:
            cfg.max_condition_number = _expect_float(
                mapping["max_condition_number"],
                "forstner.max_condition_number",
            )
        if "max_offset" in mapping:
            cfg.max_offset = _expect_float(mapping["max_offset"], "forstner.max_offset")
        return cfg

    def to_dict(self) -> JsonDict:
        return {
            "radius": self.radius,
            "min_trace": self.min_trace,
            "min_det": self.min_det,
            "max_condition_number": self.max_condition_number,
            "max_offset": self.max_offset,
        }


@dataclass
class SaddlePointConfig(_PrettyMixin):
    """Quadratic saddle-point refiner parameters."""

    radius: int = 2
    det_margin: float = 1e-3
    max_offset: float = 1.5
    min_abs_det: float = 1e-4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SaddlePointConfig:
        mapping = _expect_mapping(data, "saddle_point")
        _reject_unknown_keys(
            mapping,
            {"radius", "det_margin", "max_offset", "min_abs_det"},
            "saddle_point",
        )
        cfg = cls()
        if "radius" in mapping:
            cfg.radius = _expect_int(mapping["radius"], "saddle_point.radius")
        if "det_margin" in mapping:
            cfg.det_margin = _expect_float(mapping["det_margin"], "saddle_point.det_margin")
        if "max_offset" in mapping:
            cfg.max_offset = _expect_float(mapping["max_offset"], "saddle_point.max_offset")
        if "min_abs_det" in mapping:
            cfg.min_abs_det = _expect_float(mapping["min_abs_det"], "saddle_point.min_abs_det")
        return cfg

    def to_dict(self) -> JsonDict:
        return {
            "radius": self.radius,
            "det_margin": self.det_margin,
            "max_offset": self.max_offset,
            "min_abs_det": self.min_abs_det,
        }


class PeakFitMode(str, Enum):
    """Subpixel peak-fit mode used by the Radon refiner and detector."""

    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"


@dataclass
class RadonPeakConfig(_PrettyMixin):
    """Local Duda-Frese Radon refiner parameters."""

    ray_radius: int = 2
    patch_radius: int = 3
    image_upsample: int = 2
    response_blur_radius: int = 1
    peak_fit: PeakFitMode = PeakFitMode.GAUSSIAN
    min_response: float = 0.0
    max_offset: float = 1.5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RadonPeakConfig:
        mapping = _expect_mapping(data, "radon_peak")
        _reject_unknown_keys(
            mapping,
            {
                "ray_radius",
                "patch_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "min_response",
                "max_offset",
            },
            "radon_peak",
        )
        cfg = cls()
        if "ray_radius" in mapping:
            cfg.ray_radius = _expect_int(mapping["ray_radius"], "radon_peak.ray_radius")
        if "patch_radius" in mapping:
            cfg.patch_radius = _expect_int(mapping["patch_radius"], "radon_peak.patch_radius")
        if "image_upsample" in mapping:
            cfg.image_upsample = _expect_int(
                mapping["image_upsample"], "radon_peak.image_upsample"
            )
        if "response_blur_radius" in mapping:
            cfg.response_blur_radius = _expect_int(
                mapping["response_blur_radius"], "radon_peak.response_blur_radius"
            )
        if "peak_fit" in mapping:
            cfg.peak_fit = _enum_value(PeakFitMode, mapping["peak_fit"], "radon_peak.peak_fit")
        if "min_response" in mapping:
            cfg.min_response = _expect_float(mapping["min_response"], "radon_peak.min_response")
        if "max_offset" in mapping:
            cfg.max_offset = _expect_float(mapping["max_offset"], "radon_peak.max_offset")
        return cfg

    def to_dict(self) -> JsonDict:
        return {
            "ray_radius": self.ray_radius,
            "patch_radius": self.patch_radius,
            "image_upsample": self.image_upsample,
            "response_blur_radius": self.response_blur_radius,
            "peak_fit": self.peak_fit.value,
            "min_response": self.min_response,
            "max_offset": self.max_offset,
        }


@dataclass
class RadonDetectorParams(_PrettyMixin):
    """Whole-image Duda-Frese Radon detector parameters."""

    ray_radius: int = 4
    image_upsample: int = 2
    response_blur_radius: int = 1
    peak_fit: PeakFitMode = PeakFitMode.GAUSSIAN
    threshold_rel: float = 0.01
    threshold_abs: Optional[float] = None
    nms_radius: int = 4
    min_cluster_size: int = 2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RadonDetectorParams:
        mapping = _expect_mapping(data, "radon_detector")
        _reject_unknown_keys(
            mapping,
            {
                "ray_radius",
                "image_upsample",
                "response_blur_radius",
                "peak_fit",
                "threshold_rel",
                "threshold_abs",
                "nms_radius",
                "min_cluster_size",
            },
            "radon_detector",
        )
        cfg = cls()
        if "ray_radius" in mapping:
            cfg.ray_radius = _expect_int(mapping["ray_radius"], "radon_detector.ray_radius")
        if "image_upsample" in mapping:
            cfg.image_upsample = _expect_int(
                mapping["image_upsample"], "radon_detector.image_upsample"
            )
        if "response_blur_radius" in mapping:
            cfg.response_blur_radius = _expect_int(
                mapping["response_blur_radius"], "radon_detector.response_blur_radius"
            )
        if "peak_fit" in mapping:
            cfg.peak_fit = _enum_value(
                PeakFitMode, mapping["peak_fit"], "radon_detector.peak_fit"
            )
        if "threshold_rel" in mapping:
            cfg.threshold_rel = _expect_float(
                mapping["threshold_rel"], "radon_detector.threshold_rel"
            )
        if "threshold_abs" in mapping:
            value = mapping["threshold_abs"]
            cfg.threshold_abs = (
                None if value is None else _expect_float(value, "radon_detector.threshold_abs")
            )
        if "nms_radius" in mapping:
            cfg.nms_radius = _expect_int(mapping["nms_radius"], "radon_detector.nms_radius")
        if "min_cluster_size" in mapping:
            cfg.min_cluster_size = _expect_int(
                mapping["min_cluster_size"], "radon_detector.min_cluster_size"
            )
        return cfg

    def to_dict(self) -> JsonDict:
        return {
            "ray_radius": self.ray_radius,
            "image_upsample": self.image_upsample,
            "response_blur_radius": self.response_blur_radius,
            "peak_fit": self.peak_fit.value,
            "threshold_rel": self.threshold_rel,
            "threshold_abs": self.threshold_abs,
            "nms_radius": self.nms_radius,
            "min_cluster_size": self.min_cluster_size,
        }


@dataclass
class RefinerConfig(_PrettyMixin):
    """Flat public refiner configuration with default-initialized leaves."""

    kind: RefinementMethod = RefinementMethod.CENTER_OF_MASS
    center_of_mass: CenterOfMassConfig = field(default_factory=CenterOfMassConfig)
    forstner: ForstnerConfig = field(default_factory=ForstnerConfig)
    saddle_point: SaddlePointConfig = field(default_factory=SaddlePointConfig)
    radon_peak: RadonPeakConfig = field(default_factory=RadonPeakConfig)

    @classmethod
    def center_of_mass_config(cls) -> RefinerConfig:
        return cls(kind=RefinementMethod.CENTER_OF_MASS)

    @classmethod
    def forstner_config(cls) -> RefinerConfig:
        return cls(kind=RefinementMethod.FORSTNER)

    @classmethod
    def saddle_point_config(cls) -> RefinerConfig:
        return cls(kind=RefinementMethod.SADDLE_POINT)

    @classmethod
    def radon_peak_config(cls) -> RefinerConfig:
        return cls(kind=RefinementMethod.RADON_PEAK)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RefinerConfig:
        mapping = _expect_mapping(data, "refiner")
        _reject_unknown_keys(
            mapping,
            {"kind", "center_of_mass", "forstner", "saddle_point", "radon_peak"},
            "refiner",
        )
        cfg = cls()
        if "kind" in mapping:
            cfg.kind = _enum_value(RefinementMethod, mapping["kind"], "refiner.kind")
        if "center_of_mass" in mapping:
            cfg.center_of_mass = CenterOfMassConfig.from_dict(
                _expect_mapping(mapping["center_of_mass"], "refiner.center_of_mass")
            )
        if "forstner" in mapping:
            cfg.forstner = ForstnerConfig.from_dict(
                _expect_mapping(mapping["forstner"], "refiner.forstner")
            )
        if "saddle_point" in mapping:
            cfg.saddle_point = SaddlePointConfig.from_dict(
                _expect_mapping(mapping["saddle_point"], "refiner.saddle_point")
            )
        if "radon_peak" in mapping:
            cfg.radon_peak = RadonPeakConfig.from_dict(
                _expect_mapping(mapping["radon_peak"], "refiner.radon_peak")
            )
        return cfg

    def to_dict(self) -> JsonDict:
        return {
            "kind": self.kind.value,
            "center_of_mass": self.center_of_mass.to_dict(),
            "forstner": self.forstner.to_dict(),
            "saddle_point": self.saddle_point.to_dict(),
            "radon_peak": self.radon_peak.to_dict(),
        }


@dataclass
class ChessConfig(_PrettyMixin):
    """Canonical detector configuration used across Rust, Python, and CLI JSON."""

    detector_mode: DetectorMode = DetectorMode.CANONICAL
    descriptor_mode: DescriptorMode = DescriptorMode.FOLLOW_DETECTOR
    threshold_mode: ThresholdMode = ThresholdMode.RELATIVE
    threshold_value: float = 0.2
    nms_radius: int = 2
    min_cluster_size: int = 2
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    pyramid_levels: int = 1
    pyramid_min_size: int = 128
    refinement_radius: int = 3
    merge_radius: float = 3.0
    radon_detector: RadonDetectorParams = field(default_factory=RadonDetectorParams)

    @classmethod
    def single_scale(cls) -> ChessConfig:
        return cls()

    @classmethod
    def multiscale(cls) -> ChessConfig:
        return cls(pyramid_levels=3, pyramid_min_size=128)

    @classmethod
    def radon(cls) -> ChessConfig:
        """Preset that selects the whole-image Radon detector."""

        return cls(detector_mode=DetectorMode.RADON, pyramid_levels=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChessConfig:
        mapping = _expect_mapping(data, "config")
        _reject_unknown_keys(
            mapping,
            {
                "detector_mode",
                "descriptor_mode",
                "threshold_mode",
                "threshold_value",
                "nms_radius",
                "min_cluster_size",
                "refiner",
                "pyramid_levels",
                "pyramid_min_size",
                "refinement_radius",
                "merge_radius",
                "radon_detector",
            },
            "config",
        )

        cfg = cls()
        if "detector_mode" in mapping:
            cfg.detector_mode = _enum_value(
                DetectorMode,
                mapping["detector_mode"],
                "config.detector_mode",
            )
        if "descriptor_mode" in mapping:
            cfg.descriptor_mode = _enum_value(
                DescriptorMode,
                mapping["descriptor_mode"],
                "config.descriptor_mode",
            )
        if "threshold_mode" in mapping:
            cfg.threshold_mode = _enum_value(
                ThresholdMode,
                mapping["threshold_mode"],
                "config.threshold_mode",
            )
        if "threshold_value" in mapping:
            cfg.threshold_value = _expect_float(mapping["threshold_value"], "config.threshold_value")
        if "nms_radius" in mapping:
            cfg.nms_radius = _expect_int(mapping["nms_radius"], "config.nms_radius")
        if "min_cluster_size" in mapping:
            cfg.min_cluster_size = _expect_int(
                mapping["min_cluster_size"],
                "config.min_cluster_size",
            )
        if "refiner" in mapping:
            cfg.refiner = RefinerConfig.from_dict(
                _expect_mapping(mapping["refiner"], "config.refiner")
            )
        if "pyramid_levels" in mapping:
            cfg.pyramid_levels = _expect_int(mapping["pyramid_levels"], "config.pyramid_levels")
        if "pyramid_min_size" in mapping:
            cfg.pyramid_min_size = _expect_int(
                mapping["pyramid_min_size"],
                "config.pyramid_min_size",
            )
        if "refinement_radius" in mapping:
            cfg.refinement_radius = _expect_int(
                mapping["refinement_radius"],
                "config.refinement_radius",
            )
        if "merge_radius" in mapping:
            cfg.merge_radius = _expect_float(mapping["merge_radius"], "config.merge_radius")
        if "radon_detector" in mapping:
            cfg.radon_detector = RadonDetectorParams.from_dict(
                _expect_mapping(mapping["radon_detector"], "config.radon_detector")
            )
        return cfg

    @classmethod
    def from_json(cls, text: str) -> ChessConfig:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"failed to parse config JSON: {exc}") from exc
        return cls.from_dict(_expect_mapping(data, "config"))

    @classmethod
    def from_json_file(cls, path: str | Path) -> ChessConfig:
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"failed to read config {file_path}: {exc}") from exc
        return cls.from_json(text)

    def to_dict(self) -> JsonDict:
        return {
            "detector_mode": self.detector_mode.value,
            "descriptor_mode": self.descriptor_mode.value,
            "threshold_mode": self.threshold_mode.value,
            "threshold_value": self.threshold_value,
            "nms_radius": self.nms_radius,
            "min_cluster_size": self.min_cluster_size,
            "refiner": self.refiner.to_dict(),
            "pyramid_levels": self.pyramid_levels,
            "pyramid_min_size": self.pyramid_min_size,
            "refinement_radius": self.refinement_radius,
            "merge_radius": self.merge_radius,
            "radon_detector": self.radon_detector.to_dict(),
        }


def find_chess_corners(image: Any, cfg: ChessConfig | None = None) -> Any:
    """Detect chessboard corners in a 2D C-contiguous uint8 NumPy image."""

    return _native.find_chess_corners(image, None if cfg is None else cfg.to_json())


if hasattr(_native, "find_chess_corners_with_ml"):
    def find_chess_corners_with_ml(image: Any, cfg: ChessConfig | None = None) -> Any:
        """Detect chessboard corners using the ML-backed refiner pipeline."""

        return _native.find_chess_corners_with_ml(
            image,
            None if cfg is None else cfg.to_json(),
        )


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
]

if hasattr(_native, "find_chess_corners_with_ml"):
    __all__.append("find_chess_corners_with_ml")
