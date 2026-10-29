// Overlay data builders: turn parsed detector output into the items the
// stage2d layers draw, and the pixel buffer behind the heatmap.

import type { MarkerShape, PointSetItem } from "@vitavision/stage2d";
import type { DetectedCorner, HeatmapData } from "../types/chess-corners";

export const OVERLAY_COLORS = {
  axis0: "rgb(80, 200, 255)",
  axis1: "rgb(255, 150, 60)",
  heatmap: "rgb(120, 200, 90)",
} as const;

// Viridis (polynomial approximation): perceptually uniform, monotone in
// lightness, and legible over both light and dark image content. The two
// extremes are trimmed so the darkest step does not vanish on a black pixel
// and the lightest does not wash out on a white one.
const VIRIDIS = [
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
] as const;
const RAMP_RANGE = [0.12, 0.92] as const;
/** Ellipses of one colour share a stage2d path, so the continuous ramp is quantised to this many steps. */
export const RAMP_STEPS = 24;

/** Sequential ramp, `t` in [0, 1]: low values dark blue-purple, high values yellow. */
export function ramp(t: number): [number, number, number] {
  const x = RAMP_RANGE[0] + Math.min(1, Math.max(0, t)) * (RAMP_RANGE[1] - RAMP_RANGE[0]);
  return [0, 1, 2].map((ch) => {
    let v = 0;
    for (let k = VIRIDIS.length - 1; k >= 0; k--) v = v * x + VIRIDIS[k]![ch]!;
    return Math.round(255 * Math.min(1, Math.max(0, v)));
  }) as [number, number, number];
}

/** CSS colour of `t`, snapped to one of `RAMP_STEPS` levels. */
export function rampCss(t: number): string {
  const q = Math.round(Math.min(1, Math.max(0, t)) * (RAMP_STEPS - 1)) / (RAMP_STEPS - 1);
  const [r, g, b] = ramp(q);
  return `rgb(${r}, ${g}, ${b})`;
}

// --- Orientation markers --------------------------------------------------
// stage2d markers are sized in screen pixels. Both read the item's `angle`
// (radians, clockwise from +x); the wedge reads its half-width from `angle2`.

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

/** Both ± fit-standard-error wedges of an undirected axis (half-width `angle2`, clamped so a weak corner stays legible). */
export const WEDGE_MARKER: MarkerShape = {
  paint: "line",
  size: AXIS_HALF,
  path: (x, y, u, angle, halfWidth = 0) => {
    const s = Math.min(0.6, Math.max(0, halfWidth));
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

/** Two items per corner, one per axis, in the axis colours; a `wedge` item carries the fit standard error instead of drawing the axis. */
export function axisItems(corners: readonly DetectedCorner[], wedge: boolean): PointSetItem[] {
  return corners.flatMap((c, i) =>
    c.axes.map((axis, k) => ({
      id: `${i}:${k}`,
      x: c.x,
      y: c.y,
      kind: wedge ? "wedge" : "axis",
      angle: axis.angle,
      ...(wedge ? { angle2: axis.fitStdError } : {}),
      color: k === 0 ? OVERLAY_COLORS.axis0 : OVERLAY_COLORS.axis1,
    })),
  );
}

/**
 * The dense response colormapped (ramp) to RGBA. Detector responses are
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
    const [r, g, b] = ramp(tl);
    px.set([r, g, b, Math.round(210 * tl)], i * 4);
  }
  return px;
}
