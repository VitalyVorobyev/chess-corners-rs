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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_corners(json_path: Path):
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    corners = data.get("corners", [])
    xs = [float(c["x"]) for c in corners]
    ys = [float(c["y"]) for c in corners]
    phases = [int(c.get("phase", 0)) for c in corners]
    orientations = []
    for c in corners:
        o = c.get("orientation")
        orientations.append(float(o) if o is not None else None)
    meta = {
        "downsample": data.get("downsample", 1),
        "width": data.get("width"),
        "height": data.get("height"),
        "image": data.get("image"),
    }
    return xs, ys, phases, orientations, meta


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
    parser.add_argument(
        "--show-orientation",
        action="store_true",
        help="Draw orientation arrows (when orientations are present in the JSON).",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=15.0,
        help="Arrow length in pixels for orientation arrows.",
    )
    args = parser.parse_args()

    image_path = args.image
    if args.json:
        json_path = args.json
        xs, ys, phases, orientations, meta = load_corners(json_path)
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
        xs, ys, phases, orientations, meta = load_corners(json_path)

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
        # Phases and orientations are scale-invariant; no change needed.

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)

    phase_palette = ["#e63946", "#1d3557", "#2a9d8f", "#f4a261"]
    colors = [phase_palette[p % len(phase_palette)] for p in phases]

    unique_vals, counts = np.unique(phases, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        print(val, "occurs", cnt, "times")

    ax.scatter(xs, ys, s=16, facecolors="none", edgecolors=colors, linewidths=0.7)

    unique_phases = sorted(set(phases))
    if unique_phases:
        handles = []
        labels = []
        for p in unique_phases:
            handles.append(
                plt.Line2D(
                    [], [], marker="o", linestyle="", markeredgecolor=phase_palette[p % len(phase_palette)], markerfacecolor="none"
                )
            )
            labels.append(f"phase {p}")
        ax.legend(handles, labels, title="Phase", loc="lower right", framealpha=0.6)

    if args.show_orientation:
        arrow_xs = []
        arrow_ys = []
        arrow_us = []
        arrow_vs = []
        arrow_colors = []
        for x, y, o, c in zip(xs, ys, orientations, colors):
            if o is None:
                continue
            arrow_xs.append(x)
            arrow_ys.append(y)
            arrow_us.append(math.cos(o) * args.arrow_length)
            arrow_vs.append(math.sin(o) * args.arrow_length)
            arrow_colors.append(c)

        if arrow_xs:
            ax.quiver(
                arrow_xs,
                arrow_ys,
                arrow_us,
                arrow_vs,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=arrow_colors,
                width=0.002,
            )

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    if False:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved overlay to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
