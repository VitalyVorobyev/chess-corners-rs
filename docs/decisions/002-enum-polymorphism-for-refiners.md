# ADR-002: Enum-Based Polymorphism for Refiners

## Status

Accepted (retrospective, introduced in v0.2.0)

## Context

The project supports multiple subpixel refinement strategies (CenterOfMass, Forstner, SaddlePoint) with different configuration knobs. The runtime dispatch mechanism needed to be:

- Zero-cost or near-zero-cost in the hot path (called per corner)
- Exhaustive at compile time (adding a new refiner should not silently fall through)
- Serializable for configuration files

The standard Rust alternatives are: trait objects (`dyn CornerRefiner`), enum dispatch, or generics with monomorphization.

## Decision

Use an enum `Refiner` with per-variant structs for dispatch, paired with a `CornerRefiner` trait for the shared interface:

```rust
pub enum Refiner {
    CenterOfMass(CenterOfMassRefiner),
    Forstner(ForstnerRefiner),
    SaddlePoint(SaddlePointRefiner),
}
```

A separate `RefinerKind` enum holds serializable configuration and constructs `Refiner` instances via `Refiner::from_kind()`. The `CornerRefiner` trait exists for static-dispatch bounds but is not used as `dyn`.

## Consequences

**Positive:**
- Zero-cost dispatch via `match` (no vtable indirection)
- Exhaustive matching catches missing variants at compile time
- `RefinerKind` is `Serialize`/`Deserialize`-friendly for config files
- Predictable performance: no hidden allocations or indirection

**Negative:**
- Adding a new refiner requires touching the enum (closed set by design)
- External consumers cannot plug in custom refiners without modifying crate code
- Slightly more boilerplate than `dyn` dispatch
