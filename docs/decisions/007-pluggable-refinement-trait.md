# ADR-007: Pluggable Refinement Trait Design

## Status

Accepted (retrospective, introduced in v0.2.0)

## Context

Subpixel corner refinement runs in a tight loop over all detected candidates. The refinement strategy must be swappable (different applications benefit from different refiners) while meeting hard performance constraints:

- No heap allocations per corner
- Scratch buffers must be reusable across calls
- The pipeline must know why a corner was rejected (not just that it was)

## Decision

Define a `CornerRefiner` trait:

```rust
pub trait CornerRefiner {
    fn radius(&self) -> i32;
    fn refine(&mut self, seed_xy: [f32; 2], ctx: RefineContext<'_>) -> RefineResult;
}
```

Key design choices:

- **`&mut self`**: allows refiners to maintain internal scratch buffers (e.g., gradient matrices for Forstner) without per-call allocation.
- **`RefineContext`**: provides optional `ImageView` and `ResponseMap` so each refiner can choose its input source.
- **`RefineResult`** includes `RefineStatus` enum (`Accepted`, `Rejected`, `OutOfBounds`, `IllConditioned`) alongside the refined position and score.
- **`RefinerKind`** enum holds serializable configuration; `Refiner::from_kind()` constructs the runtime implementation.

Default behavior is preserved: CenterOfMass is the legacy default and must reproduce prior results unless the user opts in to a different refiner.

## Consequences

**Positive:**
- Zero hot-path allocations via mutable self with internal buffers
- Status enum enables per-reason filtering (e.g., keep ill-conditioned corners for diagnostics)
- Easy to add new refiners by implementing the trait
- Configuration is serializable and config-file friendly

**Negative:**
- Mutable self means refiners are not `Send` without wrapping (not an issue since refinement is per-thread)
- Status enum adds a branch per corner (negligible cost)
- Default behavior constraint limits optimization of CenterOfMass
