#!/usr/bin/env python3
"""
Quickly overlay detected ChESS corners on top of an input image.

Supports JSON produced by:
- legacy examples (dump_corners / dump_corners_multiscale)
- the `chess-cli` application (DetectionDump format)

Examples:
    # Legacy example output
    python tools/plot_corners.py testdata/images/Cam1.png \
        --json testdata/images/Cam1.corners.json

    # CLI output that carries the image path inside the JSON
    python tools/plot_corners.py --json output.json
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
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Input image path (optional when --json is provided and contains an 'image' field).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help=(
            "Corners JSON produced by the examples or chess-cli. "
            "If omitted, defaults to <image>.corners.json or <image>.multiscale.corners.json."
        ),
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Prefer multiscale dump naming (<image>.multiscale.corners.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path for overlay PNG (defaults to <image>.corners_overlay.png).",
    )
    args = parser.parse_args()

    image_path = args.image
    if args.json:
        json_path = args.json
        xs, ys, meta = load_corners(json_path)
        if image_path is None:
            img_field = meta.get("image")
            if not img_field:
                raise SystemExit(
                    "When --json is used without an image path, the JSON must contain an 'image' field."
                )
            image_path = Path(img_field)
    else:
        if image_path is None:
            raise SystemExit("Either an image path or --json must be provided.")
        primary = image_path.with_suffix(".corners.json")
        alt = image_path.with_suffix(".multiscale.corners.json")
        json_path = alt if args.multi else primary
        if not json_path.exists():
            json_path = primary if json_path == alt else alt
        xs, ys, meta = load_corners(json_path)

    out_path = args.out or image_path.with_suffix(".corners_overlay.png")

    img = Image.open(image_path)

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
