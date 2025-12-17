""" ChESS corner object """

from dataclasses import dataclass, fields
import json
from pathlib import Path

@dataclass
class ChESSCorner:
    """ ChESS corner object """
    x: float
    y: float
    response: float
    orientation: float
    scale: float | None = None

    def to_json(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return ChESSCorner(**data)

def load_corners(json_path: Path) -> list[ChESSCorner]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    corners = data.get("corners", [])
    corner_fields = {f.name for f in fields(ChESSCorner)}
    corners = [
        ChESSCorner(**{k: v for k, v in c.items() if k in corner_fields})
        for c in corners
    ]
    return corners
