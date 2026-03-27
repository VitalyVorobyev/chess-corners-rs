# ADR-001: Layered Crate Architecture (Core vs Facade)

## Status

Accepted (retrospective, established at project inception)

## Context

The project needs to serve two audiences: users who want a simple, ergonomic API for chessboard corner detection, and advanced users (or embedded targets) who need low-level access to the response computation and detection algorithms without heavy dependencies.

A single crate would force all users to accept the same dependency footprint and API complexity. Separating concerns allows the core algorithms to remain lean and `no_std`-compatible while the facade provides convenience.

## Decision

Use a layered published-crate structure:

- **`box-image-pyramid`**: Standalone fixed-2x grayscale pyramid construction with reusable buffers. Independent of the chess-specific crates so it can be reused separately.

- **`chess-corners-core`**: Low-level, performance-oriented crate containing all algorithmic implementations (ring response, detection, refinement). Minimal dependencies (`log` only mandatory). Supports `no_std` with `alloc`. Treats its public API as "sharp tools" with fewer stability guarantees.

- **`chess-corners`**: High-level facade that depends on `chess-corners-core`. Adds ergonomic wrappers (`ChessConfig`, `ChessParams`), the multiscale/pyramid pipeline, `image` crate integration, ML refiner gating, and the CLI binary. This is the stability boundary for public API.

**Rule:** `chess-corners-core` must never depend on `chess-corners`. `box-image-pyramid` must remain independent of the chess-specific crates. Core algorithms and hot-path chess logic live in `core`; reusable pyramid construction lives in `box-image-pyramid`; convenience, builders, and feature gating live in the facade.

## Consequences

**Positive:**
- Standalone pyramid construction can be reused without pulling in chess-specific code
- Core can be used in `no_std`/embedded contexts without pulling in the facade
- Users who only need raw response maps avoid heavy transitive dependencies
- Feature flags propagate upward naturally (e.g., facade's `rayon` enables core and pyramid acceleration where appropriate)
- Clear ownership: core owns correctness/performance, facade owns ergonomics/stability

**Negative:**
- Re-export management: facade must carefully select which core types to re-export
- More crates to version and publish (though automated via CI)
- Some code that conceptually belongs together is split across crate boundaries
