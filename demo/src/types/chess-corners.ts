// Demo-internal domain types for the chess-corners WASM detector.
//
// `detect` / `detectRgba` return a flat `Float32Array` with stride 7 per
// corner: `[x, y, response, axis0_angle, axis0_sigma, axis1_angle,
// axis1_sigma]`. The demo parses that into the structured shapes below
// (see `lib/detector.ts`).

/** One orientation axis: undirected angle (radians) + its 1σ uncertainty. */
export interface Axis {
  /** Undirected axis angle in radians (image frame: x right, y down). */
  angle: number;
  /** 1σ uncertainty of `angle`, in radians. */
  sigma: number;
}

/** A single detected corner with subpixel position and two orientation axes. */
export interface DetectedCorner {
  /** Image x in pixels. */
  x: number;
  /** Image y in pixels. */
  y: number;
  /** Detector response / strength at this corner. */
  response: number;
  /** The two estimated orientation axes. */
  axes: [Axis, Axis];
  /** Position covariance in px², or `null` when the refiner reports none. */
  covariance: PositionCovariance | null;
}

/** Symmetric 2×2 position covariance `[[xx, xy], [xy, yy]]`, in px². */
export interface PositionCovariance {
  xx: number;
  xy: number;
  yy: number;
}

/** Dense detector response/heatmap at working resolution (row-major). */
export interface HeatmapData {
  data: Float32Array;
  width: number;
  height: number;
}

/** Which dense detector drives the pipeline. */
export type Strategy = "chess" | "radon";

/** Two-axis orientation-fit method. */
export type OrientationKind = "ringFit" | "diskFit";

/** Subpixel refiner for the ChESS strategy. */
export type ChessRefinerKind = "centerOfMass" | "saddlePoint";

/** Radon subpixel refiner. */
export type RadonRefinerKind = "peakFit" | "saddlePoint";

/** Position-uncertainty tier. */
export type UncertaintyKind = "off" | "propagated";

/** UI-facing detector configuration, mapped to the typed WASM config tree. */
export interface DetectorSettings {
  strategy: Strategy;
  multiscale: boolean;
  /** Acceptance threshold. ChESS: absolute response floor (0–100). Radon: fraction of per-frame max (0–0.5). */
  threshold: number;
  orientation: OrientationKind;
  chessRefiner: ChessRefinerKind;
  radonRefiner: RadonRefinerKind;
  uncertainty: UncertaintyKind;
  /** Pixel-noise σ in gray levels supplied by the user; `null` lets the detector estimate it per image. */
  noiseSigma: number | null;
  /** Non-maximum-suppression radius (px). */
  nmsRadius: number;
  /** Minimum cluster size for a detection to survive. */
  minClusterSize: number;
}
