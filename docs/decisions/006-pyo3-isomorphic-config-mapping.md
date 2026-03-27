# ADR-006: PyO3 Isomorphic Config Mapping

## Status

Accepted (retrospective, introduced in v0.3.0)

## Context

Python bindings need to expose the same configuration surface as the Rust API. Two approaches were considered:

1. **Dict-based config** -- accept Python dicts, parse into Rust structs. Flexible but loses type safety and IDE support.
2. **Isomorphic classes** -- mirror each Rust config struct as a `#[pyclass]` with matching getters/setters. More boilerplate but type-safe.

## Decision

Mirror each Rust config struct (`ChessParams`, `CoarseToFineParams`, `PyramidParams`, refiner configs) as a `#[pyclass]` with getters/setters that match Rust field names. Bidirectional conversion methods (`from_rust()` / `to_rust()`) bridge the Python-Rust boundary.

Python users access configuration via familiar attribute syntax:
```python
cfg = chess_corners.ChessConfig()
cfg.threshold_rel = 0.15
cfg.pyramid_num_levels = 3
```

## Consequences

**Positive:**
- Python users get IDE autocomplete and type checking
- Type safety at the FFI boundary (invalid types rejected by PyO3)
- Familiar attribute-access API matches Rust struct field access
- Configuration is validated at the boundary, not deep in Rust code

**Negative:**
- Boilerplate-heavy: each Rust struct change requires parallel Python-side update
- Config structs are duplicated across the FFI boundary
- Nested config access (e.g., `cfg.multiscale.pyramid.num_levels`) requires wrapper classes at each level
