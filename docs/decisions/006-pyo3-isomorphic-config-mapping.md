# ADR-006: PyO3 Isomorphic Config Mapping

## Status

Superseded by ADR-007

## Context

This decision record documents the old Python binding design.

The previous approach mirrored Rust config structs directly through PyO3
`#[pyclass]` wrappers (`ChessParams`, `CoarseToFineParams`, `PyramidParams`,
refiner configs, etc.).

## Decision

That design has been retired.

## Consequences

Why it was superseded:

- it leaked Rust internal structure into Python
- nested objects like `cfg.multiscale.pyramid` made the public API harder to use
- editor discoverability and docstrings were still poor in practice
- keeping parallel PyO3 wrapper classes in sync created too much boilerplate

See ADR-007 for the replacement design.
