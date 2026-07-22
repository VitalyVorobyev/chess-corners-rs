# Roadmap

Milestones toward `v1.0.0`. Task IDs link to [`BACKLOG.md`](BACKLOG.md);
design detail lives in the linked `design/` docs.

**Sequencing:** `M1 → M2 → M3 (API surface) → M4 site → M5 C++/vcpkg → M6 hardening → tag 1.0.0`.
M3 froze the API surface; M6 re-opened it for a final coherence/tech-debt pass
before the freeze becomes a semver-locked contract. `SWEEP-*` (public-surface
dev-history cleanup) and `SOLID-*` (DRY/cohesion) run continuously, folded into
the M2/M3/M6 windows.

All milestones are **done**. The `1.0.0` **release act** (`API-09`) shipped:
`chess-corners`, `chess-corners-core`, and `box-image-pyramid` are published
on crates.io and PyPI at `1.0.0`. Two residuals carried forward: the npm
package `@vitavision/chess-corners` never got its `wasm-v1.0.0` tag and is
still at `0.11.2`, and the vcpkg port SHA512 is still a placeholder
(`CPP-05`). The workspace is now in the `1.1.0` **release act**: version bump
to `1.1.0`, `cargo-semver-checks` flipped from advisory to blocking (baseline
auto-detected from the published `1.0.0`), the `image` codec-free dependency
fix for #68, and an ONNX runtime bump + MSRV `1.91`. The npm publish and
vcpkg SHA512 residuals carry into this release act too.

## Milestones

| M | Goal | Tasks | Status | Outcome |
|---|------|-------|--------|---------|
| M1 | KB backbone: `docs/` knowledge base, algorithm index, design docs | `DOCS-01` | done | `docs/{README,ROADMAP,BACKLOG}` + `design/` landed; references swept. |
| M2 | Perf: bench every atomic hot path, profiling automation, CI regression gate | `PERF-01..12`, `SOLID-01` | done | Atomic benches + `tools/profile.sh` + CI bench gate (≤2% median drift, `bench-gate.yml`); baselines in `tools/perf/`. See [`design/perf-profiling.md`](design/perf-profiling.md). |
| M3 | Freeze a minimal semver-stable public surface | `API-01..07` | done | Fields dropped, config dedup, sealed traits, `#[non_exhaustive]`, MSRV stated; re-opened + re-frozen by M6. See [`design/api-v1.0.md`](design/api-v1.0.md). |
| M4 | GitHub Pages site: landing → book → API → demo → performance | `SITE-01..06` | done | `/`, `/book/`, `/api/`, `/demo/`, `/performance/` assembled by `docs.yml`; `scripts/build-site.sh` reproduces locally. See [`design/site-architecture.md`](design/site-architecture.md). |
| M5 | vcpkg-installable C/C++ binding | `CPP-01..07` | done | `chess-corners-capi` + cbindgen header + C++ header + CMake `find_package`; vcpkg port is a verified-local draft (registry finalize at release). See [`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md). |
| M6 | Design hardening before the freeze becomes semver-locked | `DEBT-01..05` | done | Deleted `unstable`/`low_level` escape hatches; config lowering exposed as `DetectorConfig` methods; argmax sentinel → `Option`; facade `config.rs` split. Detection bit-stable. |

## Release act (in progress)

`API-09` (the `1.0.0` release act) is done: `chess-corners`,
`chess-corners-core`, and `box-image-pyramid` were tagged and published to
crates.io, and the Python bindings published to PyPI, at `1.0.0`. Two
residuals were not completed and carry into `1.1.0`:

- **npm publish** — no `wasm-v1.0.0` tag was ever pushed, so
  `@vitavision/chess-corners` is still at `0.11.2` on npm (`DEBT-11`).
- **vcpkg SHA512** — `ports/chess-corners/portfile.cmake` still has the
  placeholder `SHA512 0` (`CPP-05`).

The `1.1.0` release act (this cycle) bumps the workspace version
`1.0.0 → 1.1.0`, flips `cargo-semver-checks` from advisory to a blocking CI
gate now that `1.0.0` is a valid crates.io baseline (`API-08`), removes the
`image` crate's default-features codec bundle (fixes #68, `DEBT-09`), and
bumps the ONNX runtime + MSRV to `1.91` (`DEBT-10`). The npm publish and
vcpkg SHA512 residuals remain open and are expected to close alongside or
shortly after `1.1.0`, since they don't require a new Rust release.

## Continuous

- **`SWEEP-*`** — keep public surfaces (book/README/rustdoc/CHANGELOG) free of
  dev-history/lineage/origin references.
- **`SOLID-*`** — DRY/cohesion cleanup (shared test utils, dispatch, fixtures).
- **`SKILL-*`** — keep the reusable user-level skill set current with the
  patterns this program produces (bindings, docs-site assembly, KB restructure).

## Out of scope (for now)

Anything requiring a post-1.0 breaking change; new detector algorithms.
Stable-Rust SIMD was evaluated (`PERF-11`) and rejected: no stable backend
matches the nightly `std::simd` path without regressing aarch64 below scalar or
breaking the bit-exact pyramid, so `simd` stays nightly-only as an optional
high-performance path and the stable scalar/autovec build is the supported
portable baseline.
