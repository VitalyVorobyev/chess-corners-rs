// The image stage with the detector overlays, built on @vitavision/stage2d.
//
// Detector coordinates are pixel-centre (pixel i covers [i-0.5, i+0.5]), which
// is also stage2d's convention, so items are passed through unshifted.

import { useMemo, useState } from "react";
import {
  EllipseSet,
  HeatmapLayer,
  ImageLayer,
  ImageStage,
  PointSet,
  StageReadout,
  StageToolbar,
  type EllipseSetItem,
  type PointSetItem,
  type StageView,
} from "@vitavision/stage2d";
import { AXIS_MARKER, axisItems, heatmapRgba, OVERLAY_COLORS, rampCss, WEDGE_MARKER } from "./overlays";
import { covarianceEllipse, formatSigma, predictedSigma, sigmaFraction, type UncertaintySummary } from "../lib/uncertainty";
import type { DetectedCorner, HeatmapData } from "../types/chess-corners";

export type LayerId = "heatmap" | "corners" | "axes" | "wedges" | "ellipses";
export type CornerColoring = "response" | "sigma";

export interface ViewerImage {
  url: string;
  name: string;
  width: number;
  height: number;
}

interface ViewerProps {
  image: ViewerImage | null;
  corners: readonly DetectedCorner[];
  heatmap: HeatmapData | null;
  visible: Record<LayerId, boolean>;
  coloring: CornerColoring;
  /** Multiplier applied to the 1σ ellipse semi-axes. */
  exaggeration: number;
  summary: UncertaintySummary;
}

const MARKERS = { axis: AXIS_MARKER, wedge: WEDGE_MARKER };
const DEG = 180 / Math.PI;

export function Viewer({ image, corners, heatmap, visible, coloring, exaggeration, summary }: ViewerProps) {
  const [view, setView] = useState<StageView | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const cornerItems = useMemo<PointSetItem[]>(() => {
    const maxResponse = Math.max(0, ...corners.map((c) => Math.abs(c.response)));
    return corners.map((c, i) => {
      if (coloring === "response") {
        const t = maxResponse > 0 ? Math.abs(c.response) / maxResponse : 0;
        return { id: i, x: c.x, y: c.y, kind: "dot", color: rampCss(t) };
      }
      // No covariance: a cross in the label colour, distinct from the σ ramp.
      if (!c.covariance) return { id: i, x: c.x, y: c.y, kind: "cross", color: "var(--stage-label)" };
      return { id: i, x: c.x, y: c.y, kind: "dot", color: rampCss(sigmaFraction(predictedSigma(c.covariance), summary.sigmaRange)) };
    });
  }, [corners, coloring, summary.sigmaRange]);

  const ellipseItems = useMemo<EllipseSetItem[]>(
    () =>
      corners.flatMap((c, i) => {
        if (!c.covariance) return [];
        const e = covarianceEllipse(c.covariance);
        return [
          {
            id: i,
            x: c.x,
            y: c.y,
            rx: e.sigmaMajor * exaggeration,
            ry: e.sigmaMinor * exaggeration,
            angle: e.angle,
            color: rampCss(sigmaFraction(predictedSigma(c.covariance), summary.sigmaRange)),
          },
        ];
      }),
    [corners, exaggeration, summary.sigmaRange],
  );
  const axes = useMemo(() => axisItems(corners, false), [corners]);
  const wedges = useMemo(() => axisItems(corners, true), [corners]);
  const heatmapPixels = useMemo(() => (heatmap ? heatmapRgba(heatmap) : null), [heatmap]);

  const hoveredCorner = hovered === null ? undefined : corners[hovered];
  const hoveredEllipse = hoveredCorner?.covariance ? covarianceEllipse(hoveredCorner.covariance) : null;

  if (!image) return <div className="grid h-full place-items-center text-sm text-fg-muted">Choose a sample or upload an image.</div>;

  return (
    <ImageStage
      className="h-full"
      image={image}
      view={view}
      onView={setView}
      initialView="fit"
      toolbar={<StageToolbar />}
      readout={<StageReadout cursor={cursor} />}
      onHover={setCursor}
      banner={
        <div className="flex flex-col items-start gap-1.5">
          {visible.ellipses && ellipseItems.length > 0 && (
            <div className="rounded-control border border-line bg-overlay/85 px-2 py-1 text-xs text-fg-muted backdrop-blur">
              1σ position ellipses, enlarged {Math.round(exaggeration)}×
            </div>
          )}
          {hoveredCorner && (
            <dl className="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 rounded-panel border border-line bg-overlay/90 px-3 py-2 font-mono text-xs text-fg tabular-nums backdrop-blur">
              <dt className="text-fg-muted">x, y</dt>
              <dd>
                {hoveredCorner.x.toFixed(2)}, {hoveredCorner.y.toFixed(2)}
              </dd>
              <dt className="text-fg-muted">response</dt>
              <dd>{hoveredCorner.response.toFixed(1)}</dd>
              <dt style={{ color: OVERLAY_COLORS.axis0 }}>axis 0</dt>
              <dd>
                {(hoveredCorner.axes[0].angle * DEG).toFixed(1)}° ± {(hoveredCorner.axes[0].sigma * DEG).toFixed(1)}°
              </dd>
              <dt style={{ color: OVERLAY_COLORS.axis1 }}>axis 1</dt>
              <dd>
                {(hoveredCorner.axes[1].angle * DEG).toFixed(1)}° ± {(hoveredCorner.axes[1].sigma * DEG).toFixed(1)}°
              </dd>
              {hoveredEllipse ? (
                <>
                  <dt className="text-fg-muted">σ major</dt>
                  <dd>{formatSigma(hoveredEllipse.sigmaMajor)} px</dd>
                  <dt className="text-fg-muted">σ minor</dt>
                  <dd>{formatSigma(hoveredEllipse.sigmaMinor)} px</dd>
                  <dt className="text-fg-muted">ellipse angle</dt>
                  <dd>{(hoveredEllipse.angle * DEG).toFixed(1)}°</dd>
                </>
              ) : (
                <>
                  <dt className="text-fg-muted">covariance</dt>
                  <dd>none</dd>
                </>
              )}
            </dl>
          )}
        </div>
      }
    >
      <ImageLayer src={image.url} alt={image.name} />
      {visible.heatmap && heatmap && heatmapPixels && (
        <HeatmapLayer rgba={heatmapPixels} width={heatmap.width} height={heatmap.height} alt="Dense detector response" />
      )}
      {visible.wedges && <PointSet items={wedges} markers={MARKERS} pickable={false} halo={false} label="Axis uncertainty wedges" />}
      {visible.axes && <PointSet items={axes} markers={MARKERS} pickable={false} label="Orientation axes" />}
      {visible.ellipses && ellipseItems.length > 0 && (
        <EllipseSet items={ellipseItems} role="model" pickable={false} hoveredId={hovered} minRadius={1.5} label="Position uncertainty ellipses" />
      )}
      {visible.corners && (
        <PointSet
          items={cornerItems}
          hoveredId={hovered}
          onHoverChange={(id) => setHovered(typeof id === "number" ? id : null)}
          label="Detected corners"
        />
      )}
    </ImageStage>
  );
}
