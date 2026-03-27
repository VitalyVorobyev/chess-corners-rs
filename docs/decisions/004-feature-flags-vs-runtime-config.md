# ADR-004: Feature Flags for Performance vs Runtime Config for Behavior

## Status

Accepted (retrospective, established pattern throughout project)

## Context

The project needs to control both algorithmic behavior (thresholds, NMS radius, refiner selection, pyramid levels) and performance strategies (SIMD vectorization, multi-threading, parallel downsampling). These two categories have fundamentally different characteristics:

- Performance choices affect compile-time code generation and dependencies
- Behavioral choices should be tunable without recompilation

Mixing the two in a single mechanism (all features, or all runtime) would either bloat the binary with unused codepaths or force recompilation for simple parameter changes.

## Decision

- **Compile-time feature flags** control performance-only choices: `rayon` (threading), `simd` (vectorization), `par_pyramid` (parallel downsampling), `tracing` (diagnostics).
- **Runtime configuration** via `ChessParams` and `CoarseToFineParams` controls algorithmic behavior: thresholds, NMS radius, refiner kind, pyramid levels, merge radius.

The documented invariant is: **features only affect performance and observability, not numerical results**.

## Consequences

**Positive:**
- No runtime overhead for unused performance paths (dead code elimination)
- Users can tune detection behavior without recompilation
- Testing can verify numerical equivalence across feature combinations

**Negative:**
- SIMD acceleration requires nightly Rust (gated behind `simd` feature)
- Users must recompile to switch between rayon/non-rayon builds
- Feature matrix grows combinatorially (tested in CI)
