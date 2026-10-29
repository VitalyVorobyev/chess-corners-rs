//! Order-preserving per-item map with per-worker scratch.

use core::borrow::BorrowMut;

/// Map `op` over `items`, in item order, threading mutable scratch through
/// the calls.
///
/// Sequentially (no `rayon` feature, a one-thread pool, or fewer than
/// `2 · min_len` items) every call borrows `scratch`. Under `rayon` the
/// items are split into jobs of at least `min_len`; each job gets its own
/// scratch from `fork(scratch)` and the results are collected in item
/// order. `op` must not let its result depend on what earlier calls left in
/// the scratch, so both ways return the same values bit for bit.
///
/// `min_len` trades scheduling overhead against parallelism: pick it so a
/// job of `min_len` items outweighs a fork and a rayon task.
pub(crate) fn map_with_scratch<T, R, S, W>(
    items: &[T],
    min_len: usize,
    scratch: &mut S,
    fork: impl Fn(&S) -> W + Sync + Send,
    op: impl Fn(&mut S, &T) -> R + Sync + Send,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    S: ?Sized + Sync,
    W: BorrowMut<S>,
{
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let min_len = min_len.max(1);
        if items.len() >= 2 * min_len && rayon::current_num_threads() > 1 {
            let proto: &S = scratch;
            return items
                .par_iter()
                .with_min_len(min_len)
                .map_init(|| fork(proto), |w, item| op(w.borrow_mut(), item))
                .collect();
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = (min_len, &fork);
    items.iter().map(|item| op(scratch, item)).collect()
}
