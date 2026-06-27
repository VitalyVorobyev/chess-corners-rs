// Overlay-layer builders: turn parsed detector output into draw callbacks for
// CanvasViewport. Each layer's `draw` receives a ctx already in image
// coordinates, plus the current zoom `scale` so marker sizes stay roughly
// constant on screen.

import type { OverlayLayer } from "./CanvasViewport";
import type { DetectedCorner, HeatmapData } from "../types/chess-corners";

export const OVERLAY_COLORS = {
  axis0: "rgb(80, 200, 255)",
  axis1: "rgb(255, 150, 60)",
  heatmap: "rgb(120, 200, 90)",
} as const;

/** Google "Turbo" colormap, polynomial approximation. `t` in [0, 1]. */
function turbo(t: number): [number, number, number] {
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
  return [
    Math.round(255 * Math.min(1, Math.max(0, r))),
    Math.round(255 * Math.min(1, Math.max(0, g))),
    Math.round(255 * Math.min(1, Math.max(0, b))),
  ];
}

/** Filled dots on every corner, colored by normalized response (Turbo). */
export function cornersLayer(
  corners: DetectedCorner[],
  visible: boolean,
): OverlayLayer {
  let max = 0;
  for (const c of corners) {
    const a = Math.abs(c.response);
    if (a > max) max = a;
  }
  const inv = max > 0 ? 1 / max : 0;
  return {
    id: "corners",
    visible,
    draw: (ctx, scale) => {
      const r = 2.6 / Math.sqrt(scale);
      ctx.lineWidth = Math.max(0.6 / scale, 0.25);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
      for (const c of corners) {
        const [rr, gg, bb] = turbo(Math.min(1, Math.abs(c.response) * inv));
        ctx.fillStyle = `rgb(${rr}, ${gg}, ${bb})`;
        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    },
  };
}

/** Two short segments per corner — one per orientation axis (undirected). */
export function axesLayer(
  corners: DetectedCorner[],
  visible: boolean,
): OverlayLayer {
  return {
    id: "axes",
    visible,
    draw: (ctx, scale) => {
      const L = 9 / Math.sqrt(scale);
      ctx.lineWidth = Math.max(1.2 / scale, 0.4);
      ctx.lineCap = "round";
      for (const c of corners) {
        for (let k = 0; k < 2; k++) {
          const axis = c.axes[k]!;
          const dx = Math.cos(axis.angle) * L;
          const dy = Math.sin(axis.angle) * L;
          ctx.strokeStyle = k === 0 ? OVERLAY_COLORS.axis0 : OVERLAY_COLORS.axis1;
          ctx.beginPath();
          ctx.moveTo(c.x - dx, c.y - dy);
          ctx.lineTo(c.x + dx, c.y + dy);
          ctx.stroke();
        }
      }
    },
  };
}

/** Faint angular wedges showing each axis's ±1σ orientation uncertainty. */
export function sigmaLayer(
  corners: DetectedCorner[],
  visible: boolean,
): OverlayLayer {
  return {
    id: "sigma",
    visible,
    draw: (ctx, scale) => {
      const L = 9 / Math.sqrt(scale);
      for (const c of corners) {
        for (let k = 0; k < 2; k++) {
          const axis = c.axes[k]!;
          // Clamp the displayed half-angle so a low-confidence corner does not
          // flood the view; the wedge stays a legible uncertainty cue.
          const s = Math.min(0.6, Math.max(0, axis.sigma));
          if (s <= 1e-4) continue;
          ctx.fillStyle =
            k === 0 ? "rgba(80, 200, 255, 0.16)" : "rgba(255, 150, 60, 0.16)";
          // Undirected axis: mirror the wedge to both ends.
          for (const base of [axis.angle, axis.angle + Math.PI]) {
            ctx.beginPath();
            ctx.moveTo(c.x, c.y);
            ctx.arc(c.x, c.y, L, base - s, base + s);
            ctx.closePath();
            ctx.fill();
          }
        }
      }
    },
  };
}

/**
 * Semi-transparent dense response/heatmap, colormapped (Turbo) and scaled to
 * fill the base image. The colormapped canvas is built once when the layer is
 * constructed (i.e. once per detection), not on every animation frame.
 */
export function responseHeatmapLayer(
  heatmap: HeatmapData | null,
  imgW: number,
  imgH: number,
  visible: boolean,
): OverlayLayer {
  let cached: HTMLCanvasElement | null = null;
  if (heatmap && heatmap.width > 0 && heatmap.height > 0 && imgW > 0 && imgH > 0) {
    const { data, width, height } = heatmap;
    let max = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!;
      if (v > max) max = v;
    }
    const inv = max > 0 ? 1 / max : 0;
    const cv = document.createElement("canvas");
    cv.width = width;
    cv.height = height;
    const ctx = cv.getContext("2d");
    if (ctx) {
      const img = ctx.createImageData(width, height);
      const px = img.data;
      for (let i = 0; i < data.length; i++) {
        const t = Math.min(1, Math.max(0, data[i]! * inv));
        // Detector responses are sharply peaked at corners, so a linear ramp
        // leaves the field almost entirely transparent. A perceptual gamma
        // lift surfaces the weaker-but-nonzero structure while keeping true
        // zero (flat regions) transparent.
        const tl = Math.pow(t, 0.55);
        const [r, g, b] = turbo(tl);
        const o = i * 4;
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = Math.round(210 * tl);
      }
      ctx.putImageData(img, 0, 0);
      cached = cv;
    }
  }
  return {
    id: "heatmap",
    visible,
    draw: (ctx) => {
      if (!cached) return;
      const prev = ctx.imageSmoothingEnabled;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(cached, 0, 0, cached.width, cached.height, 0, 0, imgW, imgH);
      ctx.imageSmoothingEnabled = prev;
    },
  };
}
