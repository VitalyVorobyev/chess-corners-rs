# ADR-003: Determinism Guarantee with Parallel Execution

## Status

Accepted (retrospective, enforced since introduction of rayon support)

## Context

Enabling `rayon` for parallel response computation and refinement improves throughput significantly on multi-core systems. However, parallel iteration can produce non-deterministic output ordering depending on thread scheduling, which makes testing, debugging, and reproducibility difficult.

The project's AGENTS.md states: "Avoid nondeterministic iteration ordering in outputs." Calibration and computer vision applications often require bitwise reproducibility.

## Decision

When parallelism is used, always sort the final results by stable keys before returning. The sort key uses response value (descending), then x coordinate (ascending), then y coordinate (ascending).

This guarantee is documented as a project invariant: **same inputs produce the same outputs regardless of thread scheduling or feature flags**.

## Consequences

**Positive:**
- Users get reproducible results for testing and debugging
- Bitwise identical output between `rayon` and non-`rayon` builds
- No surprises when toggling parallelism features

**Negative:**
- Slight overhead from sorting after parallel collection
- Sort key must be stable and unambiguous (floating point comparisons require care)
