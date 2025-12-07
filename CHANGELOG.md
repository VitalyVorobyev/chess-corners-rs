# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2]

### Added

- Test images in `./testimages`
- Examples are tested and documented
- `tools/README.md` describing the Python benchmarking and visualization helpers.
- Basic contributor guidance (`CONTRIBUTING.md`) for running tests, docs, and feature-matrix checks.

### Changed

- Default `ChessParams` and `CoarseToFineParams` are adjusted to work in most practical cases out of the box
- CLI config examples and documentation now use numeric `radius` / `descriptor_radius` fields; these are validated and mapped to the underlying ChESS ring radii while still allowing boolean overrides from CLI flags.
- Public detector helpers (`find_chess_corners_u8`, `find_chess_corners`, `find_chess_corners_image`) are now marked `#[must_use]` and document their preconditions.
- Crate metadata now points to the `chess-corners-rs` docs site and exposes search keywords on crates.io.
