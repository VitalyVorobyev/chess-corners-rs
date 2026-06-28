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
  RadonRefiner,
  CenterOfMassConfig,
  ForstnerConfig,
  SaddlePointConfig,
  RadonPeakConfig,
} from "@vitavision/chess-corners";

import type {
  DetectedCorner,
  DetectorSettings,
  HeatmapData,
  ChessRefinerKind,
  RadonRefinerKind,
} from "../types/chess-corners";

let detector: ChessDetector | null = null;
let initialized = false;

export async function initialize(): Promise<void> {
  if (!initialized) {
    await init();
    detector = new ChessDetector();
    initialized = true;
  }
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

function mapRadonRefiner(kind: string): RadonRefinerKind {
  return kind === "center_of_mass" ? "centerOfMass" : "radonPeak";
}

/** Seed the UI from the WASM `chess()` / `radon()` presets. */
export function defaultSettings(): DetectorSettings {
  const chess = DetectorConfig.chess();
  const radon = DetectorConfig.radon();
  return {
    strategy: "chess",
    multiscale: false,
    thresholdRel: 0.05,
    orientation: "ringFit",
    chessRefiner: mapChessRefiner(chess.strategy.chess.refiner.kind),
    radonRefiner: mapRadonRefiner(radon.strategy.radon.refiner.kind),
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
  return kind === "centerOfMass"
    ? RadonRefiner.withCenterOfMass(new CenterOfMassConfig())
    : RadonRefiner.withRadonPeak(new RadonPeakConfig());
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

  cfg = cfg.withThreshold(s.thresholdRel);
  cfg = cfg.withOrientationMethod(
    s.orientation === "diskFit"
      ? OrientationMethod.DiskFit
      : OrientationMethod.RingFit,
  );
  cfg =
    s.strategy === "radon"
      ? cfg.withRadonRefiner(buildRadonRefiner(s.radonRefiner))
      : cfg.withChessRefiner(buildChessRefiner(s.chessRefiner));
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

function parseCorners(flat: Float32Array): DetectedCorner[] {
  const out: DetectedCorner[] = [];
  for (let o = 0; o + CORNER_STRIDE <= flat.length; o += CORNER_STRIDE) {
    out.push({
      x: flat[o]!,
      y: flat[o + 1]!,
      response: flat[o + 2]!,
      axes: [
        { angle: flat[o + 3]!, sigma: flat[o + 4]! },
        { angle: flat[o + 5]!, sigma: flat[o + 6]! },
      ],
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
  const corners = parseCorners(flat);

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
