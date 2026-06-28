// Single-screen workspace: interactive viewport (left) + config panel (right).
// No backend — detection runs synchronously in WASM, re-run (debounced) on
// every config change.
//
// Image sources: user file upload (or drag-drop) and bundled samples.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { CanvasViewport, type HitPoint } from "../components/CanvasViewport";
import { LayerToggles } from "../components/LayerToggles";
import { InfoTip } from "../components/InfoTip";
import {
  axesLayer,
  cornersLayer,
  OVERLAY_COLORS,
  responseHeatmapLayer,
  sigmaLayer,
} from "../components/overlays";
import { useDebounced } from "../hooks/useDebounced";
import { useImageBitmapFromUrl } from "../hooks/useImageBitmap";
import {
  defaultSettings,
  detect,
  type DetectResult,
} from "../lib/detector";
import { loadImage, rgbaFromBitmap, type ImageData as ImgData } from "../lib/image-utils";
import type {
  ChessRefinerKind,
  DetectorSettings,
  OrientationKind,
  Strategy,
} from "../types/chess-corners";

// ---------------------------------------------------------------------------
// Bundled sample gallery
// ---------------------------------------------------------------------------

interface SampleEntry {
  label: string;
  url: string;
}

const SAMPLES: SampleEntry[] = [
  { label: "Small", url: "./samples/small.png" },
  { label: "Mid", url: "./samples/mid.png" },
  { label: "Large", url: "./samples/large.png" },
];

// ---------------------------------------------------------------------------
// Tooltip data
// ---------------------------------------------------------------------------

interface TooltipData {
  x: number;
  y: number;
  response: number;
  a0: number;
  s0: number;
  a1: number;
  s1: number;
}

const DEG = 180 / Math.PI;

interface WorkspaceProps {
  ready: boolean;
}

export function Workspace({ ready }: WorkspaceProps) {
  // --- image state ---
  const [imgData, setImgData] = useState<ImgData | null>(null);
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null);
  const [loadingImg, setLoadingImg] = useState(false);

  // --- sample loading via URL ---
  const [sampleUrl, setSampleUrl] = useState<string | null>(null);
  const sampleBitmap = useImageBitmapFromUrl(sampleUrl);

  useEffect(() => {
    if (!sampleBitmap.bitmap || !ready) return;
    const bm = sampleBitmap.bitmap;
    setBitmap(bm);
    setImgData(rgbaFromBitmap(bm));
  }, [sampleBitmap.bitmap, ready]);

  // --- config ---
  const [settings, setSettings] = useState<DetectorSettings | null>(null);
  useEffect(() => {
    if (ready && !settings) setSettings(defaultSettings());
  }, [ready, settings]);

  const update = useCallback((patch: Partial<DetectorSettings>) => {
    setSettings((s) => (s ? { ...s, ...patch } : s));
  }, []);

  // Debounce config so rapid slider drags don't hammer WASM.
  const debouncedSettings = useDebounced(settings, 300);

  // --- detection output ---
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);

  // --- layer visibility ---
  const [visible, setVisible] = useState<Record<string, boolean>>({
    heatmap: false,
    corners: true,
    axes: true,
    sigma: false,
  });
  const showHeatmap = visible["heatmap"] ?? false;

  // --- run detection (re-runs on image / debounced-config / heatmap change) ---
  useEffect(() => {
    if (!ready || !imgData || !debouncedSettings) return;
    setDetecting(true);
    setError(null);
    // Defer so the "detecting…" state paints before WASM blocks the thread.
    const id = setTimeout(() => {
      try {
        const res = detect(
          imgData.rgba,
          imgData.width,
          imgData.height,
          debouncedSettings,
          showHeatmap,
        );
        setResult(res);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
        setResult(null);
      } finally {
        setDetecting(false);
      }
    }, 0);
    return () => clearTimeout(id);
  }, [ready, imgData, debouncedSettings, showHeatmap]);

  // --- file handling ---
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setLoadingImg(true);
    setSampleUrl(null);
    try {
      const { bitmap: bm, data } = await loadImage(file);
      setBitmap(bm);
      setImgData(data);
      setResult(null);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingImg(false);
    }
  }, []);

  const handleSampleClick = useCallback((entry: SampleEntry) => {
    setSampleUrl(entry.url);
    setResult(null);
    setError(null);
  }, []);

  // --- overlay layers ---
  const corners = useMemo(() => result?.corners ?? [], [result]);

  const layers = useMemo(() => {
    const w = imgData?.width ?? 0;
    const h = imgData?.height ?? 0;
    return [
      responseHeatmapLayer(result?.heatmap ?? null, w, h, visible["heatmap"] ?? false),
      sigmaLayer(corners, visible["sigma"] ?? false),
      axesLayer(corners, visible["axes"] ?? true),
      cornersLayer(corners, visible["corners"] ?? true),
    ];
  }, [result, imgData, corners, visible]);

  const hitPoints = useMemo<HitPoint<TooltipData>[]>(
    () =>
      corners.map((c) => ({
        x: c.x,
        y: c.y,
        data: {
          x: c.x,
          y: c.y,
          response: c.response,
          a0: c.axes[0].angle,
          s0: c.axes[0].sigma,
          a1: c.axes[1].angle,
          s1: c.axes[1].sigma,
        },
      })),
    [corners],
  );

  const [draggingOver, setDraggingOver] = useState(false);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* ---------------------------------------------------------------- */}
      {/* Viewport                                                          */}
      {/* ---------------------------------------------------------------- */}
      <div
        style={{ flex: 1, minWidth: 0, position: "relative" }}
        onDrop={(e) => {
          e.preventDefault();
          setDraggingOver(false);
          const file = e.dataTransfer.files[0];
          if (file) void handleFile(file);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDraggingOver(true);
        }}
        onDragLeave={() => setDraggingOver(false)}
      >
        <CanvasViewport
          image={bitmap}
          layers={layers}
          hitPoints={hitPoints}
          renderTooltip={(d) => (
            <>
              <div style={{ color: "var(--text)" }}>
                x {d.x.toFixed(2)} · y {d.y.toFixed(2)}
              </div>
              <div style={{ color: "var(--text-muted)" }}>
                response {d.response.toFixed(1)}
              </div>
              <div style={{ color: OVERLAY_COLORS.axis0 }}>
                axis 0 {(d.a0 * DEG).toFixed(1)}° ± {(d.s0 * DEG).toFixed(1)}°
              </div>
              <div style={{ color: OVERLAY_COLORS.axis1 }}>
                axis 1 {(d.a1 * DEG).toFixed(1)}° ± {(d.s1 * DEG).toFixed(1)}°
              </div>
            </>
          )}
        />
        {draggingOver && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "color-mix(in srgb, var(--bg0) 85%, transparent)",
              border: "2px dashed var(--accent)",
              borderRadius: "var(--radius)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
              fontSize: 15,
              color: "var(--accent)",
            }}
          >
            Drop image to load
          </div>
        )}
        {(sampleBitmap.loading || loadingImg) && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "color-mix(in srgb, var(--bg0) 70%, transparent)",
              color: "var(--text-muted)",
              fontSize: 13,
            }}
          >
            Loading…
          </div>
        )}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Right panel                                                       */}
      {/* ---------------------------------------------------------------- */}
      <aside
        style={{
          width: 300,
          flexShrink: 0,
          borderLeft: "1px solid var(--border)",
          background: "var(--bg1)",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "var(--s4)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--s4)",
          }}
        >
          {/* Image source */}
          <div>
            <div className="label" style={{ marginBottom: "var(--s2)" }}>Image</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--s2)" }}>
              <button
                className="btn"
                onClick={() => fileInputRef.current?.click()}
                style={{ width: "100%", justifyContent: "center" }}
              >
                Upload file…
              </button>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "var(--s1)" }}>
                {SAMPLES.map((s) => (
                  <button
                    key={s.url}
                    className="btn"
                    style={{ fontSize: 11, padding: "4px 8px", justifyContent: "center" }}
                    onClick={() => handleSampleClick(s)}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
              {imgData && (
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {imgData.width} × {imgData.height} px
                </div>
              )}
            </div>
          </div>

          {/* Stats */}
          {(result != null || error || detecting) && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s1)" }}>
              {error ? (
                <span className="chip err">{error}</span>
              ) : detecting ? (
                <span className="chip warn">detecting…</span>
              ) : result ? (
                <>
                  <span className="chip">{result.corners.length} corners</span>
                  <span className="chip">{result.timeMs.toFixed(1)} ms</span>
                </>
              ) : null}
            </div>
          )}

          {settings ? (
            <>
              {/* Strategy */}
              <div>
                <div className="label" style={{ marginBottom: "var(--s2)" }}>Detector</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "var(--s2)" }}>
                  <SelectRow
                    label="Strategy"
                    value={settings.strategy}
                    options={[
                      ["chess", "ChESS"],
                      ["radon", "Radon"],
                    ]}
                    onChange={(v) => update({ strategy: v as Strategy })}
                  />
                  <SelectRow
                    label="Orientation"
                    value={settings.orientation}
                    options={[
                      ["ringFit", "Ring fit"],
                      ["diskFit", "Disk fit"],
                    ]}
                    onChange={(v) => update({ orientation: v as OrientationKind })}
                    info="RingFit fits a 16-sample ring; DiskFit is a full-disk crossing-line estimator for strongly warped corners."
                  />
                  {settings.strategy === "chess" && (
                    <SelectRow
                      label="Refiner"
                      value={settings.chessRefiner}
                      options={[
                        ["centerOfMass", "Center of mass"],
                        ["forstner", "Förstner"],
                        ["saddlePoint", "Saddle point"],
                      ]}
                      onChange={(v) => update({ chessRefiner: v as ChessRefinerKind })}
                    />
                  )}
                </div>
              </div>

              {/* Threshold */}
              <div>
                <div className="label" style={{ marginBottom: "var(--s2)" }}>
                  Threshold
                  <InfoTip text="Relative acceptance threshold: corners are kept when their response is at least this fraction of the per-frame maximum response." />
                </div>
                <SliderRow
                  label="Relative"
                  value={settings.thresholdRel}
                  min={0}
                  max={0.5}
                  step={0.01}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => update({ thresholdRel: v })}
                />
              </div>

              {/* Multiscale */}
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  fontSize: 12,
                  cursor: "pointer",
                  color: "var(--text-muted)",
                }}
              >
                <input
                  type="checkbox"
                  checked={settings.multiscale}
                  onChange={(e) => update({ multiscale: e.target.checked })}
                  style={{ accentColor: "var(--accent)" }}
                />
                Multiscale (coarse-to-fine pyramid)
                <InfoTip text="Runs the detector across an image pyramid — more robust to scale, at extra cost." />
              </label>

              {/* Advanced */}
              <details>
                <summary style={{ cursor: "pointer", listStyle: "revert" }}>
                  <span className="label">Advanced</span>
                </summary>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "var(--s2)",
                    marginTop: "var(--s2)",
                  }}
                >
                  <NumberRow
                    label="NMS radius"
                    value={settings.nmsRadius}
                    step={1}
                    integer
                    min={1}
                    onChange={(v) => update({ nmsRadius: v })}
                  />
                  <NumberRow
                    label="Min cluster"
                    value={settings.minClusterSize}
                    step={1}
                    integer
                    min={1}
                    onChange={(v) => update({ minClusterSize: v })}
                  />
                </div>
              </details>

              {/* Layers */}
              <div>
                <div className="label" style={{ marginBottom: "var(--s2)" }}>Layers</div>
                <LayerToggles
                  toggles={[
                    { id: "corners", label: "Corners (colored by response)", checked: visible["corners"] ?? true },
                    { id: "axes", label: "Orientation axes", checked: visible["axes"] ?? true, swatch: OVERLAY_COLORS.axis0 },
                    { id: "sigma", label: "Axis σ wedges", checked: visible["sigma"] ?? false, swatch: OVERLAY_COLORS.axis1 },
                    { id: "heatmap", label: "Response heatmap", checked: visible["heatmap"] ?? false, swatch: OVERLAY_COLORS.heatmap },
                  ]}
                  onChange={(id, checked) => setVisible((v) => ({ ...v, [id]: checked }))}
                />
              </div>
            </>
          ) : (
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
              Initializing detector…
            </div>
          )}
        </div>
      </aside>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared mini-components
// ---------------------------------------------------------------------------

function NumberRow({
  label,
  value,
  step,
  integer,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  step: number;
  integer?: boolean;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label
      style={{
        display: "grid",
        gridTemplateColumns: "90px 1fr",
        alignItems: "center",
        gap: "var(--s2)",
        fontSize: 12,
        color: "var(--text-muted)",
      }}
    >
      {label}
      <input
        className="input"
        type="number"
        step={step}
        value={value}
        min={min}
        max={max}
        style={{ padding: "3px 6px", fontSize: 12 }}
        onChange={(e) => {
          const v = e.target.valueAsNumber;
          if (!Number.isNaN(v)) onChange(integer ? Math.round(v) : v);
        }}
      />
    </label>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label
      style={{
        display: "grid",
        gridTemplateColumns: "70px 1fr 36px",
        alignItems: "center",
        gap: "var(--s2)",
        fontSize: 12,
        color: "var(--text-muted)",
      }}
    >
      {label}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ accentColor: "var(--accent)", width: "100%" }}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
      <span style={{ fontFamily: "var(--font-mono)", textAlign: "right" }}>{format(value)}</span>
    </label>
  );
}

function SelectRow({
  label,
  value,
  options,
  onChange,
  info,
}: {
  label: string;
  value: string;
  options: readonly (readonly [string, string])[];
  onChange: (v: string) => void;
  info?: string;
}) {
  return (
    <label
      style={{
        display: "grid",
        gridTemplateColumns: "90px 1fr",
        alignItems: "center",
        gap: "var(--s2)",
        fontSize: 12,
        color: "var(--text-muted)",
      }}
    >
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
        {label}
        {info && <InfoTip text={info} />}
      </span>
      <select
        className="select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ fontSize: 12 }}
      >
        {options.map(([k, l]) => (
          <option key={k} value={k}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}
