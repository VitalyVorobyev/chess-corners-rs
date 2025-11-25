#!/usr/bin/env python3
"""
Quickly overlay detected ChESS corners on top of an input image.

Example:
    python tools/plot_corners.py testdata/images/Cam1.png \
        --json testdata/images/Cam1.corners.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def load_corners(json_path: Path):
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    corners = data.get("corners", [])
    xs = [float(c["x"]) for c in corners]
    ys = [float(c["y"]) for c in corners]
    meta = {
        "downsample": data.get("downsample", 1),
        "width": data.get("width"),
        "height": data.get("height"),
        "image": data.get("image"),
    }
    return xs, ys, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay ChESS corners on an image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--json",
        type=Path,
        help="Corners JSON produced by dump_corners (defaults to <image>.corners.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path for overlay PNG (defaults to <image>.corners_overlay.png).",
    )
    args = parser.parse_args()

    image_path = args.image
    json_path = args.json or image_path.with_suffix(".corners.json")
    out_path = args.out or image_path.with_suffix(".corners_overlay.png")

    img = Image.open(image_path)
    xs, ys, meta = load_corners(json_path)

    # Scale coordinates if the JSON was produced at a different resolution.
    scale_x = scale_y = 1.0
    if meta.get("width") and meta.get("height") and meta["width"] > 0 and meta["height"] > 0:
        scale_x = img.width / float(meta["width"])
        scale_y = img.height / float(meta["height"])
    elif meta.get("downsample", 1) and meta["downsample"] > 1:
        # Fallback: assume coordinates are in the downsampled space.
        scale_x = scale_y = float(meta["downsample"])

    if not math.isclose(scale_x, 1.0) or not math.isclose(scale_y, 1.0):
        xs = [x * scale_x for x in xs]
        ys = [y * scale_y for y in ys]

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.scatter(xs, ys, s=16, facecolors="none", edgecolors="red", linewidths=0.7)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved overlay to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
