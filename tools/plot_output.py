#!/usr/bin/env python3
"""
Render an overlay of chess-corners detections from a JSON output file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from detection.chesscorner import load_corners
from plotting.chess_plot import plot_chess_corners


def _load_image(path: Path):
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

    img = plt.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img


def _auto_arrow_length(corners, img_shape) -> float:
    """Pick a reasonable arrow length from median corner spacing."""
    import numpy as np

    if len(corners) < 2:
        h, w = img_shape[:2]
        return float(max(12, min(w, h) * 0.03))

    xy = np.array([(c.x, c.y) for c in corners], dtype=float)
    # For each corner, distance to its nearest neighbor.
    n = xy.shape[0]
    nearest = np.empty(n)
    for i in range(n):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        d[i] = np.inf
        nearest[i] = d.min()
    median = float(np.median(nearest))
    # Arrows half the typical spacing keep neighbouring arrows clearly
    # separated while staying visible.
    return max(6.0, 0.45 * median)


def _resolve_image_path(output_json: Path, override: Path | None) -> Path:
    if override is not None:
        return override

    with output_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    image_value = data.get("image")
    if not image_value:
        raise SystemExit("output.json does not include an image path; pass --image")

    img_path = Path(image_value)
    if img_path.is_absolute():
        return img_path

    candidate = output_json.parent / img_path
    if candidate.exists():
        return candidate

    return img_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot chess-corners detections from a chess-corners output JSON."
    )
    parser.add_argument("output_json", type=Path, help="Path to output.json.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Override image path (defaults to image in output.json).",
    )
    parser.add_argument("--out", type=Path, help="Optional path to save the overlay PNG.")
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Disable grid-axis orientation arrows.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=None,
        help=(
            "Arrow length for grid-axis overlays, in image pixels. "
            "Auto-picked from median nearest-neighbor spacing if omitted."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI (default 180).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure (save only if --out is set).",
    )
    args = parser.parse_args()

    output_json = args.output_json
    if not output_json.exists():
        raise SystemExit(f"output.json not found: {output_json}")

    img_path = _resolve_image_path(output_json, args.image)
    if not img_path.exists():
        raise SystemExit(f"image not found: {img_path}")

    corners = load_corners(output_json)
    img = _load_image(img_path)

    arrow_length = args.arrow_length
    if arrow_length is None:
        arrow_length = _auto_arrow_length(corners, img.shape[:2])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray")
    plot_chess_corners(
        ax,
        corners,
        show_orientation=not args.no_orientation,
        arrow_length=arrow_length,
    )
    ax.set_axis_off()
    if not args.no_orientation and corners:
        ax.legend(loc="lower right", framealpha=0.8, fontsize=8)
    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved overlay to {args.out}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
