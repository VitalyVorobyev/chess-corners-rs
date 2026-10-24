// Right-hand inspector: image source, detector settings, layers and the
// uncertainty summary. Built from @vitavision/ui and @vitavision/workbench.

import type { ReactNode } from "react";
import { Badge, Button, Checkbox, DensityProvider, Disclosure, Field, InfoHint, NumberInput, SegmentedControl, Section, Select, Slider, Switch } from "@vitavision/ui";
import { FileDrop } from "@vitavision/workbench";
import { OVERLAY_COLORS, turboCss } from "./overlays";
import type { CornerColoring, LayerId } from "./Viewer";
import { formatSigma, type UncertaintySummary } from "../lib/uncertainty";
import { defaultThreshold, type DetectResult } from "../lib/detector";
import type { ChessRefinerKind, DetectorSettings, OrientationKind, Strategy } from "../types/chess-corners";

export const SAMPLES = [
  { label: "Small", url: "./samples/small.png" },
  { label: "Mid", url: "./samples/mid.png" },
  { label: "Large", url: "./samples/large.png" },
] as const;

export type ViewMode = "detection" | "uncertainty";

export const EXAGGERATION_LOG_RANGE = [1, 3] as const; // 10× … 1000×

interface InspectorProps {
  settings: DetectorSettings;
  onSettings: (patch: Partial<DetectorSettings>) => void;
  imageSize: { width: number; height: number } | null;
  onFile: (file: File) => void;
  onSample: (url: string) => void;
  loading: boolean;
  result: DetectResult | null;
  error: string | null;
  detecting: boolean;
  mode: ViewMode;
  onMode: (mode: ViewMode) => void;
  visible: Record<LayerId, boolean>;
  onVisible: (id: LayerId, on: boolean) => void;
  coloring: CornerColoring;
  /** log10 of the ellipse exaggeration factor. */
  exaggerationLog: number;
  onExaggerationLog: (v: number) => void;
  summary: UncertaintySummary;
}

const COVARIANCE_SOURCE = "Only the ChESS detector with the Saddle point refiner reports a position covariance.";

export function Inspector(p: InspectorProps) {
  const s = p.settings;
  const chess = s?.strategy === "chess";
  const hasCovariance = p.summary.withCovariance > 0;

  return (
    <DensityProvider value="compact">
      <div className="flex flex-col gap-5 p-3">
        <Section title="Image">
          <FileDrop accept="image/*" multiple={false} overlay buttonLabel="Upload image…" overlayMessage="Drop image to load" onFiles={(f) => f[0] && p.onFile(f[0])} />
          <div className="grid grid-cols-3 gap-1">
            {SAMPLES.map((x) => (
              <Button key={x.url} variant="secondary" size="sm" onClick={() => p.onSample(x.url)}>
                {x.label}
              </Button>
            ))}
          </div>
          <p className="font-mono text-xs text-fg-muted">
            {p.loading ? "Loading…" : p.imageSize ? `${p.imageSize.width} × ${p.imageSize.height} px` : "No image"}
          </p>
        </Section>

        <div className="flex flex-wrap gap-1.5" aria-live="polite">
          {p.error ? (
            <Badge tone="defect">{p.error}</Badge>
          ) : p.detecting ? (
            <Badge tone="info">detecting…</Badge>
          ) : p.result ? (
            <>
              <Badge>{p.result.corners.length} corners</Badge>
              <Badge>{p.result.timeMs.toFixed(1)} ms</Badge>
            </>
          ) : null}
        </div>

        {(
          <>
            <Section title="Detector">
              <Field label="Strategy">
                <Select
                  value={s.strategy}
                  onValueChange={(v) => p.onSettings({ strategy: v as Strategy, threshold: defaultThreshold(v as Strategy) })}
                  options={[
                    { value: "chess", label: "ChESS" },
                    { value: "radon", label: "Radon" },
                  ]}
                />
              </Field>
              <Field
                label="Orientation"
                annotation={<InfoHint>RingFit fits a 16-sample ring; DiskFit is a full-disk crossing-line estimator for strongly warped corners.</InfoHint>}
              >
                <Select
                  value={s.orientation}
                  onValueChange={(v) => p.onSettings({ orientation: v as OrientationKind })}
                  options={[
                    { value: "ringFit", label: "Ring fit" },
                    { value: "diskFit", label: "Disk fit" },
                  ]}
                />
              </Field>
              {chess && (
                <Field label="Refiner">
                  <Select
                    value={s.chessRefiner}
                    onValueChange={(v) => p.onSettings({ chessRefiner: v as ChessRefinerKind })}
                    options={[
                      { value: "centerOfMass", label: "Center of mass" },
                      { value: "forstner", label: "Förstner" },
                      { value: "saddlePoint", label: "Saddle point", note: "reports covariance" },
                    ]}
                  />
                </Field>
              )}
              <Field
                label={chess ? "Threshold" : "Relative threshold"}
                annotation={
                  <InfoHint>
                    {chess
                      ? "Absolute response floor: corners are kept when their raw ChESS response exceeds this value."
                      : "Fraction of the per-frame maximum Radon response."}
                  </InfoHint>
                }
                as="group"
              >
                <Slider
                  aria-label="Threshold"
                  value={s.threshold}
                  min={0}
                  max={chess ? 100 : 0.5}
                  step={chess ? 1 : 0.01}
                  readout={chess ? s.threshold.toFixed(0) : s.threshold.toFixed(2)}
                  onValueChange={(v) => p.onSettings({ threshold: v })}
                />
              </Field>
              <Switch
                checked={s.multiscale}
                onCheckedChange={(v) => p.onSettings({ multiscale: v })}
                label={
                  <span className="inline-flex items-center gap-1.5">
                    Multiscale (coarse-to-fine pyramid)
                    <InfoHint>Runs the detector across an image pyramid: more robust to scale, at extra cost.</InfoHint>
                  </span>
                }
              />
              <Disclosure summary="Advanced">
                <div className="flex flex-col gap-2">
                  <Field label="NMS radius">
                    <NumberInput unit="px" precision={0} min={1} value={s.nmsRadius} onValueChange={(v) => p.onSettings({ nmsRadius: Math.round(v) })} />
                  </Field>
                  <Field label="Min cluster">
                    <NumberInput precision={0} min={1} value={s.minClusterSize} onValueChange={(v) => p.onSettings({ minClusterSize: Math.round(v) })} />
                  </Field>
                </div>
              </Disclosure>
            </Section>

            <Section title="Layers">
              <SegmentedControl
                aria-label="View"
                value={p.mode}
                onValueChange={(v) => p.onMode(v as ViewMode)}
                options={[
                  { value: "detection", label: "Detection" },
                  { value: "uncertainty", label: "Uncertainty" },
                ]}
              />
              <div className="flex flex-col gap-1.5">
                <LayerCheckbox id="corners" label={p.coloring === "sigma" ? "Corners (colored by predicted σ)" : "Corners (colored by response)"} p={p} />
                <LayerCheckbox id="axes" label="Orientation axes" swatch={OVERLAY_COLORS.axis0} p={p} />
                <LayerCheckbox id="wedges" label="Axis σ wedges" swatch={OVERLAY_COLORS.axis1} p={p} />
                <LayerCheckbox id="ellipses" label="Position 1σ ellipses" swatch="var(--stage-model)" p={p} />
                <LayerCheckbox id="heatmap" label="Response heatmap" swatch={OVERLAY_COLORS.heatmap} p={p} />
              </div>
            </Section>

            <Section
              title="Position uncertainty"
              actions={<InfoHint>{COVARIANCE_SOURCE}</InfoHint>}
            >
              {p.result && !hasCovariance && p.result.corners.length > 0 && (
                <div className="flex flex-col items-start gap-1.5 text-xs text-fg-muted">
                  <p>No corner has a covariance. {COVARIANCE_SOURCE}</p>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() =>
                      p.onSettings({ strategy: "chess", chessRefiner: "saddlePoint", threshold: defaultThreshold("chess") })
                    }
                  >
                    Use ChESS + Saddle point
                  </Button>
                </div>
              )}
              <Field
                label="Ellipse exaggeration"
                annotation={<InfoHint>The ellipses are the predicted 1σ regions, drawn N times larger so they are visible; the hover readout is not scaled.</InfoHint>}
                as="group"
              >
                <Slider
                  aria-label="Ellipse exaggeration"
                  value={p.exaggerationLog}
                  min={EXAGGERATION_LOG_RANGE[0]}
                  max={EXAGGERATION_LOG_RANGE[1]}
                  step={0.01}
                  readout={`${Math.round(10 ** p.exaggerationLog)}×`}
                  onValueChange={p.onExaggerationLog}
                />
              </Field>
              {hasCovariance && p.summary.sigmaRange && <SigmaLegend range={p.summary.sigmaRange} />}
              <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 text-xs">
                <Row label="with covariance" value={p.summary.withCovariance} />
                <Row label="without covariance" value={p.summary.withoutCovariance} marker />
                <Row label="median predicted σ" value={p.summary.medianSigma === null ? "—" : `${formatSigma(p.summary.medianSigma)} px`} />
              </dl>
            </Section>
          </>
        )}
      </div>
    </DensityProvider>
  );
}

function LayerCheckbox({ id, label, swatch, p }: { id: LayerId; label: string; swatch?: string; p: InspectorProps }) {
  return (
    <Checkbox
      checked={p.visible[id]}
      onCheckedChange={(on) => p.onVisible(id, on)}
      label={
        <span className="inline-flex items-center gap-2">
          {swatch && <span className="size-2.5 shrink-0 rounded-[2px]" style={{ background: swatch }} aria-hidden />}
          {label}
        </span>
      }
    />
  );
}

function Row({ label, value, marker }: { label: string; value: ReactNode; marker?: boolean }) {
  return (
    <>
      <dt className="inline-flex items-center gap-1.5 text-fg-muted">
        {marker && (
          <svg viewBox="-5 -5 10 10" className="size-2.5 text-fg" aria-hidden>
            <path d="M-3.5 -3.5L3.5 3.5M-3.5 3.5L3.5 -3.5" stroke="currentColor" strokeWidth="1.6" fill="none" />
          </svg>
        )}
        {label}
      </dt>
      <dd className="text-right font-mono text-fg tabular-nums">{value}</dd>
    </>
  );
}

/** Colour ramp for predicted σ, in px, matching the corner colours. */
function SigmaLegend({ range: [lo, hi] }: { range: [number, number] }) {
  const stops = Array.from({ length: 9 }, (_, i) => turboCss(i / 8)).join(", ");
  return (
    <figure className="flex flex-col gap-1" aria-label="Predicted σ colour scale">
      <div className="h-2.5 rounded-[2px] border border-line" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <figcaption className="flex justify-between font-mono text-[11px] text-fg-muted tabular-nums">
        <span>{formatSigma(lo)} px</span>
        <span className="font-sans">predicted σ</span>
        <span>≥ {formatSigma(hi)} px</span>
      </figcaption>
    </figure>
  );
}
