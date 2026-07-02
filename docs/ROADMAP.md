# Roadmap

Milestones toward `v1.0.0`. Task IDs link to [`BACKLOG.md`](BACKLOG.md);
design detail lives in the linked `design/` docs.

**Sequencing:** `M1 → M2 → M3 (API surface) → M4 site → M5 C++/vcpkg → M6 hardening → tag 1.0.0`.
M3 froze the API surface; M6 re-opened it for a final coherence/tech-debt pass
before the freeze becomes a semver-locked contract. `SWEEP-*` (public-surface
dev-history cleanup) and `SOLID-*` (DRY/cohesion) run continuously, folded into
the M2/M3/M6 windows.

All milestones are **done**. The `1.0.0` **release act** (`API-09`: version
bump, `cargo-semver-checks` flip to blocking, tag → crates.io/PyPI/npm publish,
vcpkg registry finalization) is deferred by choice — there is no schedule
pressure, so the site, C++ bindings, and hardened surface ship together in the
first release.

## Milestones

| M | Goal | Tasks | Status | Outcome |
|---|------|-------|--------|---------|
| M1 | KB backbone: `docs/` knowledge base, algorithm index, design docs | `DOCS-01` | done | `docs/{README,ROADMAP,BACKLOG}` + `design/` landed; references swept. |
| M2 | Perf: bench every atomic hot path, profiling automation, CI regression gate | `PERF-01..12`, `SOLID-01` | done | Atomic benches + `tools/profile.sh` + CI bench gate (≤2% median drift, `bench-gate.yml`); baselines in `tools/perf/`. See [`design/perf-profiling.md`](design/perf-profiling.md). |
| M3 | Freeze a minimal semver-stable public surface | `API-01..07` | done | Fields dropped, config dedup, sealed traits, `#[non_exhaustive]`, MSRV stated; re-opened + re-frozen by M6. See [`design/api-v1.0.md`](design/api-v1.0.md). |
| M4 | GitHub Pages site: landing → book → API → demo → performance | `SITE-01..06` | done | `/`, `/book/`, `/api/`, `/demo/`, `/performance/` assembled by `docs.yml`; `scripts/build-site.sh` reproduces locally. See [`design/site-architecture.md`](design/site-architecture.md). |
| M5 | vcpkg-installable C/C++ binding | `CPP-01..07` | done | `chess-corners-capi` + cbindgen header + C++ header + CMake `find_package`; vcpkg port is a verified-local draft (registry finalize at release). See [`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md). |
| M6 | Design hardening before the freeze becomes semver-locked | `DEBT-01..05` | done | Deleted `unstable`/`low_level` escape hatches; config lowering exposed as `DetectorConfig` methods; argmax sentinel → `Option`; facade `config.rs` split. Detection bit-stable. |

## Release act (deferred)

`API-08` shipped advisory `cargo-semver-checks` in CI (baseline `v0.11.2`).
`API-09` is the release act only: bump `0.11.2 → 1.0.0`, flip semver-checks to
blocking, tag `v1.0.0`, publish to crates.io/PyPI/npm, and finalize the vcpkg
port (real tag + SHA512 + cross-platform `vcpkg install`, `CPP-05`). Awaiting
the go decision.

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
