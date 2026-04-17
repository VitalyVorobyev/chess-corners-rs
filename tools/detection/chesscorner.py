""" ChESS corner object """

from dataclasses import dataclass, field, fields
import json
from pathlib import Path


@dataclass
class ChESSAxis:
    """ One of two local grid axes at a corner. """
    angle: float
    sigma: float


@dataclass
class ChESSCorner:
    """ ChESS corner object """
    x: float
    y: float
    response: float
    contrast: float = 0.0
    fit_rms: float = 0.0
    axes: list[ChESSAxis] = field(default_factory=list)
    scale: float | None = None

    @property
    def orientation(self) -> float:
        """Backwards-compatible single-axis orientation (the first grid axis)."""
        if self.axes:
            return self.axes[0].angle
        return 0.0

    def to_json(self):
        payload = {
            "x": self.x,
            "y": self.y,
            "response": self.response,
            "contrast": self.contrast,
            "fit_rms": self.fit_rms,
            "axes": [{"angle": a.angle, "sigma": a.sigma} for a in self.axes],
        }
        if self.scale is not None:
            payload["scale"] = self.scale
        return json.dumps(payload)

    @staticmethod
    def from_dict(data: dict) -> "ChESSCorner":
        corner_fields = {f.name for f in fields(ChESSCorner)}
        kwargs = {k: v for k, v in data.items() if k in corner_fields}
        axes = [ChESSAxis(**a) for a in data.get("axes", [])]
        kwargs["axes"] = axes
        return ChESSCorner(**kwargs)

    @staticmethod
    def from_json(json_str: str) -> "ChESSCorner":
        return ChESSCorner.from_dict(json.loads(json_str))


def load_corners(json_path: Path) -> list[ChESSCorner]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    corners = data.get("corners", [])
    return [ChESSCorner.from_dict(c) for c in corners]
