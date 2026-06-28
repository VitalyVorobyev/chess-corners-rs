//! Top-fit selection and local refinement of the best disk pair.

use super::candidates::{
    build_pair_pool, combine_edge_pair, forced_histogram_pair, forced_seed_pair,
    histogram_candidates, CandidateSet, PairScore, MAX_CANDIDATES, MAX_PAIRS,
};
use super::data::DiskData;
use super::geometry::{signed_line_delta, wrap_pi};
use super::score::{
    fit_is_better, precompute_edge_alignments, score_pair, score_pair_cached, Fit, MAX_TANH_PIXELS,
    WIDTHS_PX,
};

/// Number of best fits kept across the candidate-pair phase to seed
/// refinement. Lowered from 4 to 2 once pair pruning made the top
/// pre-refinement fits much higher quality on average; refining only
/// the strongest two captures essentially all of the recoverable
/// angular precision on the synthetic gate while halving refinement
/// cost.
const TOP_FITS: usize = 2;

/// Maximum number of candidate pairs that get the expensive cached
/// tanh scoring after pruning. The full pair set can reach ~500 when
/// the histogram, baseline-offset, and global-grid candidate sources
/// all contribute. Most pairs have weak or contradictory gradient
/// support; ranking by per-pair edge alignment and keeping the top 24
/// leaves headroom for both crossing-line directions and a few
/// off-axis safety nets without paying for the noise.
const MAX_TOP_PAIRS: usize = 24;

/// Run the full best-pair search across all `WIDTHS_PX` using the
/// tanh-cache path. Falls back to the uncached path if `data.n`
/// exceeds [`MAX_TANH_PIXELS`] or there are fewer than 2 candidates.
pub(super) fn best_disk_fit(data: &DiskData, base0: f32, base1: f32) -> Option<Fit> {
    let candidates = histogram_candidates(data, wrap_pi(base0), wrap_pi(base1));
    let mut top: [Option<Fit>; TOP_FITS] = [None; TOP_FITS];

    // Out-of-envelope inputs fall back to the un-cached path. The
    // debug-assert guards developer mistakes; in production the
    // current `MAX_SUPPORT_RADIUS = 8` keeps `data.n ≤ ~200`.
    debug_assert!(
        data.n <= MAX_TANH_PIXELS,
        "disk pixel count {} exceeds MAX_TANH_PIXELS = {MAX_TANH_PIXELS}; \
         raise the cache size or fall back",
        data.n,
    );
    if data.n > MAX_TANH_PIXELS || candidates.n < 2 {
        return best_disk_fit_uncached(data, &candidates, &mut top);
    }

    // Step 1 — per-candidate edge alignment. `edge_pair_score(a, b)`
    // factors as `combine(s_a, s_b, total)`, so each candidate's `s_c`
    // is computed once (~200 grad samples × ~30 candidates = ~6k exps)
    // instead of once per pair (~30² × 200 = ~180k exps).
    let mut edge_align = [0.0f32; MAX_CANDIDATES];
    let grad_total = precompute_edge_alignments(data, &candidates, &mut edge_align);

    // Step 2 — build the candidate-pair pool with edge scores.
    // Force-include the pair closest to the (base0, base1) seed, and
    // the pair of the two strongest-individual edge alignments — both
    // are known-good seeds we never want to drop.
    let force_seed = forced_seed_pair(&candidates, base0, base1);
    let force_hist = forced_histogram_pair(&candidates, &edge_align);

    let mut pair_pool: [PairScore; MAX_PAIRS] = [PairScore::default(); MAX_PAIRS];
    let n_pairs = build_pair_pool(
        &candidates,
        &edge_align,
        grad_total,
        force_seed,
        force_hist,
        &mut pair_pool,
    );
    if n_pairs == 0 {
        return None;
    }

    // Step 3 — partial-sort the top-K pairs (deterministic comparator).
    let cmp = |a: &PairScore, b: &PairScore| -> core::cmp::Ordering {
        // forced first, then edge_score descending, then (i, j) ascending.
        b.forced
            .cmp(&a.forced)
            .then_with(|| b.edge_score.total_cmp(&a.edge_score))
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    };
    let k = n_pairs.min(MAX_TOP_PAIRS);
    if k < n_pairs {
        // O(n) selection followed by a small O(k log k) sort gives
        // deterministic ordering of the kept pairs.
        pair_pool[..n_pairs].select_nth_unstable_by(k - 1, cmp);
    }
    pair_pool[..k].sort_by(cmp);

    // Step 4 — only candidates that appear in the top-K pairs need
    // their per-(width, pixel) tanh values precomputed. Typical: 12
    // active candidates out of ~30, ~3× saving on tanh count vs the
    // dense Phase-1 cache.
    let mut active = [false; MAX_CANDIDATES];
    for p in &pair_pool[..k] {
        active[p.i as usize] = true;
        active[p.j as usize] = true;
    }

    // Step 5 — for each width, precompute the per-active-candidate
    // tanh row, then score the kept pairs.
    let mut tanh_cache = [0.0f32; MAX_CANDIDATES * MAX_TANH_PIXELS];
    for &width in &WIDTHS_PX {
        for ((c, &is_active), &theta) in active
            .iter()
            .enumerate()
            .zip(candidates.angles.iter())
            .take(candidates.n)
        {
            if !is_active {
                continue;
            }
            let (s, co) = theta.sin_cos();
            let row_start = c * MAX_TANH_PIXELS;
            let row = &mut tanh_cache[row_start..row_start + data.n];
            for (p, slot) in row.iter_mut().enumerate() {
                let d = -s * data.xs[p] + co * data.ys[p];
                // `d / width` (not `d * inv_w`) matches the original
                // `score_pair` arithmetic bit-for-bit.
                *slot = (d / width).tanh();
            }
        }
        for p in &pair_pool[..k] {
            let i = p.i as usize;
            let j = p.j as usize;
            let a0 = candidates.angles[i];
            let a1 = candidates.angles[j];
            let h0 = &tanh_cache[i * MAX_TANH_PIXELS..i * MAX_TANH_PIXELS + data.n];
            let h1 = &tanh_cache[j * MAX_TANH_PIXELS..j * MAX_TANH_PIXELS + data.n];
            if let Some(fit) = score_pair_cached(data, a0, a1, width, p.edge_score, h0, h1) {
                insert_top(&mut top, fit);
            }
        }
    }

    // Step 6 — refine top fits. Refinement still uses the un-cached
    // `score_pair` because the perturbed angles aren't in the
    // candidate set; the per-call cost is now lower thanks to the
    // sufficient-stat residual.
    let mut best = top[0]?;
    for seed in top.iter().flatten().copied() {
        let refined = refine_fit(data, seed);
        if fit_is_better(refined, best) {
            best = refined;
        }
    }
    Some(best)
}

/// Slow path retained for the (currently unreachable) case where
/// `data.n` exceeds the per-candidate tanh cache. Same algorithm and
/// outputs as the cached implementation.
pub(super) fn best_disk_fit_uncached(
    data: &DiskData,
    candidates: &CandidateSet,
    top: &mut [Option<Fit>; TOP_FITS],
) -> Option<Fit> {
    for i in 0..candidates.n {
        let a0 = candidates.angles[i];
        for j in i + 1..candidates.n {
            let a1 = candidates.angles[j];
            if !super::geometry::valid_pair(a0, a1) {
                continue;
            }
            let edge_score = super::score::edge_pair_score(data, a0, a1);
            for width in WIDTHS_PX {
                if let Some(fit) = score_pair(data, a0, a1, width, edge_score) {
                    insert_top(top, fit);
                }
            }
        }
    }
    let mut best = top[0]?;
    for seed in top.iter().flatten().copied() {
        let refined = refine_fit(data, seed);
        if fit_is_better(refined, best) {
            best = refined;
        }
    }
    Some(best)
}

/// Insert `fit` into the sorted top-K buffer.
pub(super) fn insert_top(top: &mut [Option<Fit>; TOP_FITS], fit: Fit) {
    let mut pos = TOP_FITS;
    for (i, existing) in top.iter().enumerate() {
        match existing {
            Some(v) if fit_is_better(fit, *v) => {
                pos = i;
                break;
            }
            None => {
                pos = i;
                break;
            }
            _ => {}
        }
    }
    if pos == TOP_FITS {
        return;
    }
    for i in (pos + 1..TOP_FITS).rev() {
        top[i] = top[i - 1];
    }
    top[pos] = Some(fit);
}

/// Local grid search around `seed` to refine the pair angles.
/// Step sizes: 1°, 0.5°, 0.25°.
///
/// Dispatches to the row-cached implementation whenever the disk fits the
/// per-axis scratch (`data.n <= MAX_TANH_PIXELS`, always true for the
/// production `MAX_SUPPORT_RADIUS = 8`). The scalar reference handles the
/// out-of-envelope fallback and is the bit-exact oracle the cached path
/// is tested against.
pub(super) fn refine_fit(data: &DiskData, seed: Fit) -> Fit {
    if data.n <= MAX_TANH_PIXELS {
        refine_fit_cached(data, seed)
    } else {
        refine_fit_scalar(data, seed)
    }
}

/// Row-cached refinement.
///
/// Each greedy 3×3 step evaluates eight neighbours that share only three
/// distinct θ₁ and three distinct θ₂ angles, yet the scalar reference
/// recomputes both `tanh` factors *and* the full gradient-alignment edge
/// score for all eight. Here the per-angle `tanh` rows and (un-normalised)
/// edge sums are built once per step — rebuilt only when the greedy update
/// moves the center — and reused across the eight combinations through the
/// bit-exact [`score_pair_cached`] / [`combine_edge_pair`] paths. The
/// `tanh`/`exp` count per step drops from `8·2·n` to `≤(1+moves)·3·2·n`.
/// Output is bit-identical to [`refine_fit_scalar`].
fn refine_fit_cached(data: &DiskData, seed: Fit) -> Fit {
    let n = data.n;
    let mut best = seed;
    let total = edge_weight_total(data);
    // Three θ₁-side and three θ₂-side rows for offsets [-step, 0, +step]
    // around the current best, with their matching un-normalised edge
    // sums. Stack-only; no per-corner heap allocation.
    let mut t1_rows = [[0.0f32; MAX_TANH_PIXELS]; 3];
    let mut t2_rows = [[0.0f32; MAX_TANH_PIXELS]; 3];
    let mut t1_edge = [0.0f32; 3];
    let mut t2_edge = [0.0f32; 3];

    for step_deg in [1.0f32, 0.5, 0.25] {
        let step = step_deg.to_radians();
        let offsets = [-step, 0.0, step];
        // Rebuild the cached rows at the start of each step and whenever a
        // greedy update moves `best` (so the rows always describe the
        // angles `score_pair_cached` is about to be called with).
        let mut stale = true;
        for (i0, &d0) in offsets.iter().enumerate() {
            for (i1, &d1) in offsets.iter().enumerate() {
                if d0 == 0.0 && d1 == 0.0 {
                    continue;
                }
                if stale {
                    build_axis_rows(
                        data,
                        best.theta1,
                        &offsets,
                        best.width,
                        &mut t1_rows,
                        &mut t1_edge,
                    );
                    build_axis_rows(
                        data,
                        best.theta2,
                        &offsets,
                        best.width,
                        &mut t2_rows,
                        &mut t2_edge,
                    );
                    stale = false;
                }
                let edge = combine_edge_pair(t1_edge[i0], t2_edge[i1], total);
                if let Some(fit) = score_pair_cached(
                    data,
                    best.theta1 + d0,
                    best.theta2 + d1,
                    best.width,
                    edge,
                    &t1_rows[i0][..n],
                    &t2_rows[i1][..n],
                ) {
                    if fit_is_better(fit, best) {
                        best = fit;
                        stale = true;
                    }
                }
            }
        }
    }
    best
}

/// Scalar reference refinement: recomputes the `tanh` model and the
/// gradient-alignment edge score from scratch for every candidate. Used
/// when `data.n > MAX_TANH_PIXELS` (out of the cached envelope) and as the
/// bit-exact oracle for [`refine_fit_cached`].
fn refine_fit_scalar(data: &DiskData, seed: Fit) -> Fit {
    let mut best = seed;
    // Step pattern dropped from `[2.0, 1.0, 0.5, 0.25]` to
    // `[1.0, 0.5, 0.25]`. The 2° step's role was to recover from a
    // rough top-pair seed; pair pruning now elevates higher-quality
    // seeds, so the coarse step's hit rate is too low to pay for itself
    // on the synthetic gate.
    for step_deg in [1.0f32, 0.5, 0.25] {
        let step = step_deg.to_radians();
        for d0 in [-step, 0.0, step] {
            for d1 in [-step, 0.0, step] {
                if d0 == 0.0 && d1 == 0.0 {
                    continue;
                }
                if let Some(fit) =
                    score_pair(data, best.theta1 + d0, best.theta2 + d1, best.width, -1.0)
                {
                    if fit_is_better(fit, best) {
                        best = fit;
                    }
                }
            }
        }
    }
    best
}

/// Σ of the per-pixel gradient magnitudes. Matches the `total`
/// accumulation order in [`super::score::edge_pair_score`] so the cached
/// edge score is bit-identical to the scalar one.
#[inline]
fn edge_weight_total(data: &DiskData) -> f32 {
    let mut total = 0.0f32;
    for p in 0..data.n {
        total += data.grad_weights[p];
    }
    total
}

/// Fill `rows[k]` / `edges[k]` for `θ = base + offsets[k]`, k ∈ {0,1,2}:
///
/// - `rows[k][p] = tanh((-sinθ·xₚ + cosθ·yₚ) / width)`
/// - `edges[k]   = Σₚ wₚ · exp(-½(Δₚ/σ)²)` (un-normalised, σ = 4°)
///
/// Both expressions match `score_pair` / `edge_pair_score` operation for
/// operation (same factor order, same accumulation order), so feeding the
/// rows to [`score_pair_cached`] and the edge sums to [`combine_edge_pair`]
/// reproduces the scalar refinement bit-for-bit.
fn build_axis_rows(
    data: &DiskData,
    base: f32,
    offsets: &[f32; 3],
    width: f32,
    rows: &mut [[f32; MAX_TANH_PIXELS]; 3],
    edges: &mut [f32; 3],
) {
    let n = data.n;
    let sigma = 4.0_f32.to_radians();
    for (k, &off) in offsets.iter().enumerate() {
        let theta = base + off;
        let (s, c) = theta.sin_cos();
        let row = &mut rows[k];
        let mut s_e = 0.0f32;
        for p in 0..n {
            let d = -s * data.xs[p] + c * data.ys[p];
            row[p] = (d / width).tanh();
            let dd = signed_line_delta(data.grad_angles[p], theta);
            s_e += data.grad_weights[p] * (-0.5 * (dd / sigma) * (dd / sigma)).exp();
        }
        edges[k] = s_e;
    }
}
