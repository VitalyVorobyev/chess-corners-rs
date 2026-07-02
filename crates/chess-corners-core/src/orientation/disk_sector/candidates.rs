//! Candidate angle generation and pair-pruning infrastructure.

use core::f32::consts::PI;

use super::data::DiskData;
use super::geometry::{line_delta, valid_pair, wrap_pi};

pub(super) const MAX_CANDIDATES: usize = 64;
const HIST_BINS: usize = 72;

/// Upper bound on the number of valid pairs we can stage in the pruning
/// buffer. With `MAX_CANDIDATES = 64` the worst case is
/// `64·63/2 = 2016`. The `PairScore` struct is 16 bytes so this
/// allocates ~32 KB on the stack — comfortably small but not free.
pub(super) const MAX_PAIRS: usize = MAX_CANDIDATES * (MAX_CANDIDATES - 1) / 2;

/// Compact record used by the pair-pruning step. Sized at 16 bytes so a
/// `MAX_PAIRS` array stays around 32 KB on the stack.
#[derive(Clone, Copy, Default)]
pub(super) struct PairScore {
    pub(super) i: u16,
    pub(super) j: u16,
    pub(super) edge_score: f32,
    pub(super) forced: bool,
    pub(super) _pad: [u8; 7],
}

/// Fixed-capacity set of candidate line angles.
#[derive(Clone, Copy)]
pub(super) struct CandidateSet {
    pub(super) n: usize,
    pub(super) angles: [f32; MAX_CANDIDATES],
}

impl CandidateSet {
    pub(super) fn new() -> Self {
        Self {
            n: 0,
            angles: [0.0; MAX_CANDIDATES],
        }
    }

    /// Add `angle` (wrapped to `[0, π)`) unless an existing candidate
    /// is closer than `min_sep` in undirected-line distance.
    pub(super) fn push_unique(&mut self, angle: f32, min_sep: f32) {
        if self.n >= MAX_CANDIDATES {
            return;
        }
        let a = wrap_pi(angle);
        for &existing in &self.angles[..self.n] {
            if line_delta(a, existing) <= min_sep {
                return;
            }
        }
        self.angles[self.n] = a;
        self.n += 1;
    }
}

/// Build the candidate angle set from three sources:
/// 1. gradient-histogram peaks (top-8 after smoothing),
/// 2. ±0°, ±4°, ±8° offsets around the baseline seed pair `(base0, base1)`,
/// 3. a coarse global 30°-grid.
pub(super) fn histogram_candidates(data: &DiskData, base0: f32, base1: f32) -> CandidateSet {
    let mut candidates = CandidateSet::new();
    let mut hist = [0.0f32; HIST_BINS];
    for i in 0..data.n {
        let weight = data.grad_weights[i];
        if weight <= 0.0 {
            continue;
        }
        let idx = ((data.grad_angles[i] / PI) * HIST_BINS as f32).floor() as usize % HIST_BINS;
        hist[idx] += weight;
    }

    let mut smooth = [0.0f32; HIST_BINS];
    for i in 0..HIST_BINS {
        let prev = if i == 0 { HIST_BINS - 1 } else { i - 1 };
        let next = if i + 1 == HIST_BINS { 0 } else { i + 1 };
        smooth[i] = 0.25 * hist[prev] + 0.5 * hist[i] + 0.25 * hist[next];
    }

    let mut max_v = 0.0f32;
    for &v in &smooth {
        max_v = max_v.max(v);
    }
    if max_v > 0.0 {
        let mut used = [false; HIST_BINS];
        for _ in 0..8 {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..HIST_BINS {
                if used[i] {
                    continue;
                }
                if best.is_none_or(|(_, bv)| smooth[i] > bv) {
                    best = Some((i, smooth[i]));
                }
            }
            let Some((best_i, best_v)) = best else {
                break;
            };
            if best_v < max_v * 0.12 {
                break;
            }
            used[best_i] = true;
            let angle = (best_i as f32 + 0.5) * PI / HIST_BINS as f32;
            candidates.push_unique(angle, 4.0_f32.to_radians());
        }
    }

    for seed in [base0, base1] {
        for off_deg in [-8.0f32, -4.0, 0.0, 4.0, 8.0] {
            candidates.push_unique(seed + off_deg.to_radians(), 1.0_f32.to_radians());
        }
    }

    let mut deg = 0;
    while deg < 180 {
        candidates.push_unique((deg as f32).to_radians(), 1.0_f32.to_radians());
        deg += 30;
    }

    candidates
}

/// Force-include the pair closest to the `(base0, base1)` seed.
/// Returns `(i, j)` with `i < j`, or `None` if fewer than two distinct
/// candidates exist.
pub(super) fn forced_seed_pair(
    candidates: &CandidateSet,
    base0: f32,
    base1: f32,
) -> Option<(usize, usize)> {
    if candidates.n < 2 {
        return None;
    }
    let i = closest_candidate(candidates, base0)?;
    let j = closest_candidate(candidates, base1)?;
    if i == j {
        return None;
    }
    Some(if i < j { (i, j) } else { (j, i) })
}

/// Force-include the pair of the two strongest-individual edge
/// alignments. Naturally includes the strongest histogram peaks even
/// when the pair-balance term in `combine_edge_pair` would otherwise
/// dilute their combined score.
pub(super) fn forced_histogram_pair(
    candidates: &CandidateSet,
    edge_align: &[f32; MAX_CANDIDATES],
) -> Option<(usize, usize)> {
    if candidates.n < 2 {
        return None;
    }
    let mut top0 = 0usize;
    let mut top1 = 1usize;
    if edge_align[1] > edge_align[0] {
        top0 = 1;
        top1 = 0;
    }
    for c in 2..candidates.n {
        if edge_align[c] > edge_align[top0] {
            top1 = top0;
            top0 = c;
        } else if edge_align[c] > edge_align[top1] {
            top1 = c;
        }
    }
    if top0 == top1 {
        return None;
    }
    Some(if top0 < top1 {
        (top0, top1)
    } else {
        (top1, top0)
    })
}

/// Linear search for the candidate index closest to `target` in
/// undirected-line distance. Returns `None` for an empty set.
pub(super) fn closest_candidate(candidates: &CandidateSet, target: f32) -> Option<usize> {
    if candidates.n == 0 {
        return None;
    }
    let target_w = wrap_pi(target);
    let mut best_i = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..candidates.n {
        let d = line_delta(candidates.angles[c], target_w);
        if d < best_d {
            best_d = d;
            best_i = c;
        }
    }
    Some(best_i)
}

/// Return `true` when `forced` matches `(i, j)`.
#[inline]
pub(super) fn matches_pair(forced: Option<(usize, usize)>, i: usize, j: usize) -> bool {
    matches!(forced, Some((fi, fj)) if fi == i && fj == j)
}

/// Return all `(i, j)` valid pairs from `candidates` that survive the
/// separation filter, with their edge scores and forced flag.
pub(super) fn build_pair_pool(
    candidates: &CandidateSet,
    edge_align: &[f32; MAX_CANDIDATES],
    grad_total: f32,
    force_seed: Option<(usize, usize)>,
    force_hist: Option<(usize, usize)>,
    pool: &mut [PairScore; MAX_PAIRS],
) -> usize {
    let mut n_pairs = 0usize;
    for i in 0..candidates.n {
        for j in (i + 1)..candidates.n {
            let a0 = candidates.angles[i];
            let a1 = candidates.angles[j];
            if !valid_pair(a0, a1) {
                continue;
            }
            let edge_score = combine_edge_pair(edge_align[i], edge_align[j], grad_total);
            let forced = matches_pair(force_seed, i, j) || matches_pair(force_hist, i, j);
            pool[n_pairs] = PairScore {
                i: i as u16,
                j: j as u16,
                edge_score,
                forced,
                _pad: [0; 7],
            };
            n_pairs += 1;
        }
    }
    n_pairs
}

/// Bit-identical to the tail of `edge_pair_score` after both
/// `s0 /= total` and `s1 /= total` have been applied. `s_a` / `s_b`
/// are the un-normalised candidate sums; `total` is `Σ w_p`.
#[inline]
pub(super) fn combine_edge_pair(s_a: f32, s_b: f32, total: f32) -> f32 {
    if total <= 1e-9 {
        return 0.0;
    }
    let a = s_a / total;
    let b = s_b / total;
    let balance = 2.0 * a.min(b) / (a + b).max(1e-9);
    (a + b) * balance
}
