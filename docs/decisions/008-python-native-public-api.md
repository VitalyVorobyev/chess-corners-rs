# ADR-007: Python-Native Public API With Private Native Module

## Status

Accepted

## Context

The original Python bindings mirrored Rust config structs directly through PyO3
classes. That made the Python API:

- overly coupled to Rust internal layout
- harder to discover in editors
- awkward to print and serialize
- harder to evolve without duplicating wrapper boilerplate

At the same time, the Rust public API also moved to a flatter canonical config
shape centered on `ChessConfig`.

## Decision

Adopt a mixed Rust/Python package:

- `chess_corners` is a pure-Python public API
- `chess_corners._native` is a private PyO3 extension module

The public Python package owns:

- enums
- config dataclasses
- JSON helpers
- printing helpers
- public wrapper functions with Python signatures and docstrings

The native module owns only:

- NumPy validation
- conversion to Rust `ChessConfig` via canonical JSON
- detector execution

The canonical algorithm config schema is shared across:

- Rust `ChessConfig`
- Python `ChessConfig`
- CLI JSON algorithm fields
- examples and docs

## Consequences

Positive:

- Python users get editor-visible fields, signatures, docstrings, and typed configs
- printing, JSON, and Rich integration live naturally on the Python side
- the Rust internals can evolve without forcing a one-to-one Python mirror
- the public schema is unified across Rust, Python, and CLI usage

Negative:

- config translation now crosses the native boundary as JSON rather than direct struct wrappers
- some validation logic exists both in Python and Rust
- the private/public split is a deliberate packaging design, not a thin wrapper
