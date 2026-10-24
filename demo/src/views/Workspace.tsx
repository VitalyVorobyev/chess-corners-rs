// Single-screen workspace: stage (left) + inspector (right). No backend:
// detection runs synchronously in WASM and re-runs (debounced) on every
// config change.

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { AppShell } from "@vitavision/workbench";
import { Inspector, SAMPLES, type ViewMode } from "../components/Inspector";
import { Viewer, type CornerColoring, type LayerId, type ViewerImage } from "../components/Viewer";
import { useDebounced } from "../hooks/useDebounced";
import { defaultSettings, detect, estimateNoise, type DetectResult } from "../lib/detector";
import { decodeRgba, type ImageData as ImgData } from "../lib/image-utils";
import { summarize } from "../lib/uncertainty";
import type { DetectorSettings } from "../types/chess-corners";

type Source = ViewerImage & ImgData;

const PRESETS: Record<ViewMode, Record<LayerId, boolean>> = {
  detection: { heatmap: false, corners: true, axes: true, wedges: false, ellipses: false },
  uncertainty: { heatmap: false, corners: true, axes: false, wedges: false, ellipses: true },
};

export function Workspace({ header }: { header: ReactNode }) {
  const [source, setSource] = useState<Source | null>(null);
  const [loading, setLoading] = useState(false);

  const [settings, setSettings] = useState<DetectorSettings>(defaultSettings);
  const update = useCallback((patch: Partial<DetectorSettings>) => setSettings((s) => ({ ...s, ...patch })), []);
  // Debounce config so rapid slider drags don't hammer WASM.
  const debouncedSettings = useDebounced(settings, 300);

  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);

  const [mode, setMode] = useState<ViewMode>("detection");
  const [visible, setVisible] = useState(PRESETS.detection);
  const [exaggerationLog, setExaggerationLog] = useState(2); // 100×
  const coloring: CornerColoring = mode === "uncertainty" ? "sigma" : "response";

  const changeMode = useCallback((m: ViewMode) => {
    setMode(m);
    setVisible(PRESETS[m]);
  }, []);

  // --- image loading (bundled sample or user file) ---
  const load = useCallback(async (blob: Blob, name: string) => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await decodeRgba(blob);
      setSource((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { ...data, name, url: URL.createObjectURL(blob) };
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSample = useCallback(
    async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      await load(await res.blob(), url.split("/").pop() ?? url);
    },
    [load],
  );

  useEffect(() => {
    loadSample(SAMPLES[1].url).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [loadSample]);

  // --- detection ---
  const showHeatmap = visible.heatmap;
  useEffect(() => {
    if (!source) return;
    setDetecting(true);
    setError(null);
    // Defer so the "detecting…" state paints before WASM blocks the thread.
    const id = setTimeout(() => {
      try {
        setResult(detect(source.rgba, source.width, source.height, debouncedSettings, showHeatmap));
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
        setResult(null);
      } finally {
        setDetecting(false);
      }
    }, 0);
    return () => clearTimeout(id);
  }, [source, debouncedSettings, showHeatmap]);

  const corners = useMemo(() => result?.corners ?? [], [result]);
  // Same gray image the detector sees; skipped under Off, which makes no estimate.
  const noiseEstimate = useMemo(
    () => (source && settings.uncertainty === "propagated" ? estimateNoise(source.rgba, source.width, source.height) : null),
    [source, settings.uncertainty],
  );
  const summary = useMemo(() => summarize(corners), [corners]);

  return (
    <AppShell
      header={header}
      rightLabel="Inspector"
      rightSize={{ defaultSize: 320, minSize: 260 }}
      main={
        <Viewer
          key={source?.url}
          image={source}
          corners={corners}
          heatmap={result?.heatmap ?? null}
          visible={visible}
          coloring={coloring}
          exaggeration={10 ** exaggerationLog}
          summary={summary}
        />
      }
      right={
        <Inspector
          settings={settings}
          onSettings={update}
          imageSize={source}
          onFile={(f) => void load(f, f.name)}
          onSample={(u) => void loadSample(u).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))}
          loading={loading}
          result={result}
          error={error}
          detecting={detecting}
          mode={mode}
          onMode={changeMode}
          visible={visible}
          onVisible={(id, on) => setVisible((v) => ({ ...v, [id]: on }))}
          coloring={coloring}
          exaggerationLog={exaggerationLog}
          onExaggerationLog={setExaggerationLog}
          summary={summary}
          noiseEstimate={noiseEstimate}
        />
      }
    />
  );
}
