// Typed wrapper over the raw @vitavision/chess-corners WASM API.
//
// A single persistent `ChessDetector` is reused across frames so its pyramid
// scratch buffers survive (the multiscale path reallocates only when the
// multiscale settings change). Each detect rebuilds a typed `DetectorConfig`
// from the UI settings and commits it via `applyConfig`.

import init, {
  ChessDetector,
  DetectorConfig,
  OrientationMethod,
  ChessRefiner,
  CenterOfMassConfig,
  ForstnerConfig,
  SaddlePointConfig,
  RadonRefiner,
  Uncertainty,
  estimateNoiseSigmaRgba,
} from "@vitavision/chess-corners";

import type {
  DetectedCorner,
  DetectorSettings,
  HeatmapData,
  ChessRefinerKind,
  RadonRefinerKind,
  Strategy,
} from "../types/chess-corners";

let detector: ChessDetector | null = null;
let initialized = false;
let loading: Promise<void> | null = null;

/**
 * Load the WASM module once. The promise is shared: a second `init()` while the
 * first is in flight (React StrictMode double-runs effects) would instantiate a
 * second module and strand the first detector on a replaced instance, which
 * traps with `unreachable` on its first call.
 */
export function initialize(): Promise<void> {
  loading ??= init().then(() => {
    detector = new ChessDetector();
    initialized = true;
  });
  return loading;
}

export function isReady(): boolean {
  return initialized;
}

// ---------------------------------------------------------------------------
// Defaults (read from the WASM presets so the initial run matches the library)
// ---------------------------------------------------------------------------

function mapChessRefiner(kind: string): ChessRefinerKind {
  switch (kind) {
    case "forstner":
      return "forstner";
    case "saddle_point":
      return "saddlePoint";
    default:
      return "centerOfMass";
  }
}

/** Return the preset threshold for the given strategy (reads from the WASM preset). */
export function defaultThreshold(strategy: Strategy): number {
  return strategy === "radon" ? DetectorConfig.radon().threshold : DetectorConfig.chess().threshold;
}

/** Seed the UI from the WASM `chess()` / `radon()` presets. */
export function defaultSettings(): DetectorSettings {
  const chess = DetectorConfig.chess();
  return {
    strategy: "chess",
    multiscale: false,
    threshold: chess.threshold,
    orientation: "ringFit",
    chessRefiner: mapChessRefiner(chess.strategy.chess.refiner.kind),
    radonRefiner: "peakFit",
    uncertainty: "propagated",
    noiseSigma: null,
    nmsRadius: chess.detection.nmsRadius,
    minClusterSize: chess.detection.minClusterSize,
  };
}

// ---------------------------------------------------------------------------
// Config construction
// ---------------------------------------------------------------------------

function buildChessRefiner(kind: ChessRefinerKind): ChessRefiner {
  switch (kind) {
    case "centerOfMass":
      return ChessRefiner.withCenterOfMass(new CenterOfMassConfig());
    case "saddlePoint":
      return ChessRefiner.withSaddlePoint(new SaddlePointConfig());
    case "forstner":
    default:
      return ChessRefiner.withForstner(new ForstnerConfig());
  }
}

function buildRadonRefiner(kind: RadonRefinerKind): RadonRefiner {
  return kind === "saddlePoint" ? RadonRefiner.withSaddlePoint(new SaddlePointConfig()) : RadonRefiner.withPeakFit();
}

/** The per-image noise σ (gray levels) the detector derives when none is supplied; `null` when it declines. */
export function estimateNoise(rgba: Uint8Array, width: number, height: number): number | null {
  return estimateNoiseSigmaRgba(rgba, width, height) ?? null;
}

function buildConfig(s: DetectorSettings): DetectorConfig {
  let cfg =
    s.strategy === "radon"
      ? s.multiscale
        ? DetectorConfig.radonMultiscale()
        : DetectorConfig.radon()
      : s.multiscale
        ? DetectorConfig.chessMultiscale()
        : DetectorConfig.chess();

  cfg = cfg.withThreshold(s.threshold);
  cfg = cfg.withOrientationMethod(
    s.orientation === "diskFit"
      ? OrientationMethod.DiskFit
      : OrientationMethod.RingFit,
  );
  cfg =
    s.strategy === "radon"
      ? cfg.withRadonRefiner(buildRadonRefiner(s.radonRefiner))
      : cfg.withChessRefiner(buildChessRefiner(s.chessRefiner));
  cfg = cfg.withUncertainty(s.uncertainty === "off" ? Uncertainty.Off : Uncertainty.Propagated);
  if (s.noiseSigma !== null) cfg = cfg.withNoiseSigma(s.noiseSigma);
  cfg = cfg.withDetection({
    nmsRadius: s.nmsRadius,
    minClusterSize: s.minClusterSize,
  });
  return cfg;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const CORNER_STRIDE = 7;

const COV_STRIDE = 3;

/** `cov` is stride 3 `[xx, xy, yy]`, NaN for a corner without a covariance. */
function parseCorners(flat: Float32Array, cov: Float32Array): DetectedCorner[] {
  const out: DetectedCorner[] = [];
  for (let o = 0; o + CORNER_STRIDE <= flat.length; o += CORNER_STRIDE) {
    const c = (o / CORNER_STRIDE) * COV_STRIDE;
    const xx = cov[c];
    out.push({
      x: flat[o]!,
      y: flat[o + 1]!,
      response: flat[o + 2]!,
      axes: [
        { angle: flat[o + 3]!, sigma: flat[o + 4]! },
        { angle: flat[o + 5]!, sigma: flat[o + 6]! },
      ],
      covariance:
        xx === undefined || Number.isNaN(xx)
          ? null
          : { xx, xy: cov[c + 1]!, yy: cov[c + 2]! },
    });
  }
  return out;
}

export interface DetectResult {
  corners: DetectedCorner[];
  /** Dense response/heatmap, only present when `computeHeatmap` was true. */
  heatmap: HeatmapData | null;
  /** Wall-clock time for the `detect` call itself, in milliseconds. */
  timeMs: number;
}

/**
 * Apply `settings` to the persistent detector and detect corners in an RGBA
 * frame. When `computeHeatmap` is true, additionally compute the dense
 * response map (ChESS) or Radon heatmap for overlay rendering.
 */
export function detect(
  rgba: Uint8Array,
  width: number,
  height: number,
  settings: DetectorSettings,
  computeHeatmap: boolean,
): DetectResult {
  if (!detector) throw new Error("Detector not initialized");
  detector.applyConfig(buildConfig(settings));

  const t0 = performance.now();
  const flat = detector.detectRgba(rgba, width, height);
  const timeMs = performance.now() - t0;
  const corners = parseCorners(flat, detector.positionCovariance());

  let heatmap: HeatmapData | null = null;
  if (computeHeatmap) {
    if (settings.strategy === "radon") {
      const data = detector.diagnosticsRadonHeatmapRgba(rgba, width, height);
      heatmap = {
        data,
        width: detector.diagnosticsRadonHeatmapWidth(),
        height: detector.diagnosticsRadonHeatmapHeight(),
      };
    } else {
      const data = detector.diagnosticsResponseRgba(rgba, width, height);
      heatmap = {
        data,
        width: detector.diagnosticsResponseWidth(),
        height: detector.diagnosticsResponseHeight(),
      };
    }
  }

  return { corners, heatmap, timeMs };
}
