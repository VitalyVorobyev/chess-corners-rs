// Position-covariance geometry and summary statistics (book Part VIII §8.2).
//
// The detector reports a 2×2 covariance per corner in px². Its 1σ ellipse has
// semi-axes sqrt(λ1), sqrt(λ2) for the covariance eigenvalues, and the angle of
// the major axis is ½·atan2(2·xy, xx − yy) in the image frame (x right, y down,
// clockwise on screen).

import type { DetectedCorner, PositionCovariance } from "../types/chess-corners";

export interface CovarianceEllipse {
  /** 1σ along the major axis, px. */
  sigmaMajor: number;
  /** 1σ along the minor axis, px. */
  sigmaMinor: number;
  /** Major-axis angle in radians, clockwise on screen. */
  angle: number;
}

export function covarianceEllipse({ xx, xy, yy }: PositionCovariance): CovarianceEllipse {
  const mean = (xx + yy) / 2;
  const spread = Math.hypot((xx - yy) / 2, xy);
  return {
    sigmaMajor: Math.sqrt(Math.max(mean + spread, 0)),
    sigmaMinor: Math.sqrt(Math.max(mean - spread, 0)),
    angle: 0.5 * Math.atan2(2 * xy, xx - yy),
  };
}

/** Predicted isotropic σ = sqrt((xx + yy) / 2), px. */
export function predictedSigma({ xx, yy }: PositionCovariance): number {
  return Math.sqrt(Math.max((xx + yy) / 2, 0));
}

export interface UncertaintySummary {
  withCovariance: number;
  withoutCovariance: number;
  /** Median predicted σ over corners with a covariance, px; `null` when there is none. */
  medianSigma: number | null;
  /** Colour-scale range, px: smallest σ to the 95th percentile (so one outlier does not wash out the ramp). */
  sigmaRange: [number, number] | null;
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

export function summarize(corners: readonly DetectedCorner[]): UncertaintySummary {
  const sigmas = corners
    .flatMap((c) => (c.covariance ? [predictedSigma(c.covariance)] : []))
    .sort((a, b) => a - b);
  return {
    withCovariance: sigmas.length,
    withoutCovariance: corners.length - sigmas.length,
    medianSigma: sigmas.length ? quantile(sigmas, 0.5) : null,
    sigmaRange: sigmas.length ? [sigmas[0]!, quantile(sigmas, 0.95)] : null,
  };
}

/** Format a σ in px: sub-pixel values need two significant digits to tell corners apart. */
export function formatSigma(sigma: number): string {
  return sigma >= 1 ? sigma.toFixed(2) : sigma.toPrecision(2);
}
