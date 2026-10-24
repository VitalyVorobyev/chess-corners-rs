// Overlay data builders: turn parsed detector output into the items the
// stage2d layers draw, and the pixel buffer behind the heatmap.

import type { MarkerShape, PointSetItem } from "@vitavision/stage2d";
import type { DetectedCorner, HeatmapData } from "../types/chess-corners";

export const OVERLAY_COLORS = {
  axis0: "rgb(80, 200, 255)",
  axis1: "rgb(255, 150, 60)",
  heatmap: "rgb(120, 200, 90)",
} as const;

/** Google "Turbo" colormap, polynomial approximation. `t` in [0, 1]. */
export function turbo(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const r =
    0.13572138 +
    x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g =
    0.09140261 +
    x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b =
    0.1066733 +
    x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  const byte = (v: number) => Math.round(255 * Math.min(1, Math.max(0, v)));
  return [byte(r), byte(g), byte(b)];
}

export function turboCss(t: number): string {
  const [r, g, b] = turbo(t);
  return `rgb(${r}, ${g}, ${b})`;
}

// --- Orientation markers --------------------------------------------------
// stage2d markers are sized in screen pixels. Both read the item's `angle`
// (radians, clockwise from +x); the wedge reads its σ from `angle2`.

const AXIS_HALF = 9;

/** An undirected axis: a line through the corner, `AXIS_HALF` px each way. */
export const AXIS_MARKER: MarkerShape = {
  paint: "line",
  size: AXIS_HALF,
  path: (x, y, u, angle) => {
    const dx = Math.cos(angle) * AXIS_HALF * u;
    const dy = Math.sin(angle) * AXIS_HALF * u;
    return `M${x - dx} ${y - dy}L${x + dx} ${y + dy}`;
  },
};

/** Both ±1σ wedges of an undirected axis (σ = `angle2`, clamped so a weak corner stays legible). */
export const WEDGE_MARKER: MarkerShape = {
  paint: "line",
  size: AXIS_HALF,
  path: (x, y, u, angle, sigma = 0) => {
    const s = Math.min(0.6, Math.max(0, sigma));
    const r = AXIS_HALF * u;
    let d = "";
    for (const base of [angle, angle + Math.PI]) {
      const x1 = x + Math.cos(base - s) * r;
      const y1 = y + Math.sin(base - s) * r;
      const x2 = x + Math.cos(base + s) * r;
      const y2 = y + Math.sin(base + s) * r;
      d += `M${x} ${y}L${x1} ${y1}A${r} ${r} 0 0 1 ${x2} ${y2}Z`;
    }
    return d;
  },
};

/** Two items per corner, one per axis, in the axis colours; a `wedge` item carries σ instead of drawing the axis. */
export function axisItems(corners: readonly DetectedCorner[], wedge: boolean): PointSetItem[] {
  return corners.flatMap((c, i) =>
    c.axes.map((axis, k) => ({
      id: `${i}:${k}`,
      x: c.x,
      y: c.y,
      kind: wedge ? "wedge" : "axis",
      angle: axis.angle,
      ...(wedge ? { angle2: axis.sigma } : {}),
      color: k === 0 ? OVERLAY_COLORS.axis0 : OVERLAY_COLORS.axis1,
    })),
  );
}

/**
 * The dense response colormapped (Turbo) to RGBA. Detector responses are
 * sharply peaked at corners, so a perceptual gamma lift surfaces weaker
 * structure while true zero (flat regions) stays transparent.
 */
export function heatmapRgba({ data }: HeatmapData): Uint8ClampedArray {
  let max = 0;
  for (let i = 0; i < data.length; i++) max = Math.max(max, data[i]!);
  const inv = max > 0 ? 1 / max : 0;
  const px = new Uint8ClampedArray(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    const tl = Math.pow(Math.min(1, Math.max(0, data[i]! * inv)), 0.55);
    const [r, g, b] = turbo(tl);
    px.set([r, g, b, Math.round(210 * tl)], i * 4);
  }
  return px;
}
