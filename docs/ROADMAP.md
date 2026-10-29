# Roadmap

The completed path to `v1.0.0` (M1–M6) and the post-1.0 program. Task IDs
link to [`BACKLOG.md`](BACKLOG.md); design detail lives in the linked
`design/` docs.

**Sequencing:** `M1 → M2 → M3 (API surface) → M4 site → M5 C++/vcpkg → M6 hardening → tag 1.0.0`.
M3 froze the API surface; M6 re-opened it for a final coherence/tech-debt pass
before the freeze becomes a semver-locked contract. `SWEEP-*` (public-surface
dev-history cleanup) and `SOLID-*` (DRY/cohesion) run continuously, folded into
the M2/M3/M6 windows.

All milestones are **done**. The `1.0.0` **release act** (`API-09`) shipped:
`chess-corners`, `chess-corners-core`, and `box-image-pyramid` are published
on crates.io and PyPI at `1.0.0`. Two residuals carried forward: the npm
package `@vitavision/chess-corners` never got its `wasm-v1.0.0` tag and was
stranded at `0.11.2` (closed by `1.1.0`, `DEBT-11`), and the vcpkg port SHA512
was a placeholder (`CPP-05`). `1.1.0`, `1.2.0` (facade ROI detection) and
`1.3.0` (detector-config JSON Schema + WASM JSON config path) have since
shipped on every channel, followed by `1.3.1` (maintenance), `1.4.0`
(saddle-point position covariance, `UQ-00`), `1.5.0` (per-corner position
covariance on the detector output, `UQ-05`) `1.6.0` (per-localizer position
covariance behind `Uncertainty`, Radon refiner choice, public noise estimate,
Radon false-detection fixes) and `1.7.0` (default tier `Off`, frame noise σ in
every binding, covariance field deprecations), the current version on crates.io,
PyPI and npm
(`chess-corners-ml` stays at `0.13.0`). The only open release residual is
`CPP-05`: the vcpkg port still targets `1.1.0` and has never been
install-verified.

## Milestones

| M | Goal | Tasks | Status | Outcome |
|---|------|-------|--------|---------|
| M1 | KB backbone: `docs/` knowledge base, algorithm index, design docs | `DOCS-01` | done | `docs/{README,ROADMAP,BACKLOG}` + `design/` landed; references swept. |
| M2 | Perf: bench every atomic hot path, profiling automation, CI regression gate | `PERF-01..12`, `SOLID-01` | done | Atomic benches + `tools/profile.sh` + CI bench gate (≤2% median drift, `bench-gate.yml`); baselines in `tools/perf/`. See [`design/perf-profiling.md`](design/perf-profiling.md). |
| M3 | Freeze a minimal semver-stable public surface | `API-01..07` | done | Fields dropped, config dedup, sealed traits, `#[non_exhaustive]`, MSRV stated; re-opened + re-frozen by M6. See [`design/api-v1.0.md`](design/api-v1.0.md). |
| M4 | GitHub Pages site: landing → book → API → demo → performance | `SITE-01..06` | done | `/`, `/book/`, `/api/`, `/demo/`, `/performance/` assembled by `docs.yml`; `scripts/build-site.sh` reproduces locally. See [`design/site-architecture.md`](design/site-architecture.md). |
| M5 | vcpkg-installable C/C++ binding | `CPP-01..07` | done | `chess-corners-capi` + cbindgen header + C++ header + CMake `find_package`; vcpkg port carries a real SHA512 but still targets `1.1.0` (not bumped for `1.2.0`/`1.3.0`) and is not install-verified (`CPP-05`). See [`design/cpp-vcpkg-bindings.md`](design/cpp-vcpkg-bindings.md). |
| M6 | Design hardening before the freeze becomes semver-locked | `DEBT-01..05` | done | Deleted `unstable`/`low_level` escape hatches; config lowering exposed as `DetectorConfig` methods; argmax sentinel → `Option`; facade `config.rs` split. Detection bit-stable. |

## Release act

`API-09` (the `1.0.0` release act) shipped to crates.io and PyPI but left two
residuals: no `wasm-v1.0.0` tag was ever pushed, stranding npm at `0.11.2`
(`DEBT-11`), and the vcpkg portfile kept a placeholder `SHA512 0` (`CPP-05`).

**`1.1.0` is released on every channel** — `v1.1.0` and `wasm-v1.1.0` are
tagged; crates.io has `chess-corners`, `chess-corners-core` and
`box-image-pyramid` at `1.1.0` plus `chess-corners-ml` at `0.13.0`; PyPI has
`1.1.0` (3 wheels + sdist); npm `latest` is `1.1.0`, closing `DEBT-11`. The
release removes the `image` crate's default-features codec bundle (fixes #68,
`DEBT-09`), bumps the ONNX runtime and raises MSRV to `1.91` (`DEBT-10`), and
turns `cargo-semver-checks` into a blocking CI gate now that `1.0.0` is a
valid crates.io baseline (`API-08`) — the flip immediately exposed that the
job's `package:` filter had always been malformed, so it had never actually
checked anything.

**`1.2.0` and `1.3.0` are released on every channel** — `v1.2.0`,
`wasm-v1.2.0`, `v1.3.0` and `wasm-v1.3.0` are tagged, as are the releases
through `1.7.0`, the current `latest` on crates.io, PyPI and npm
(`chess-corners-ml` unchanged at `0.13.0`). See
[`changelog/1.2.0.md`](changelog/1.2.0.md),
[`changelog/1.3.0.md`](changelog/1.3.0.md),
[`changelog/1.3.1.md`](changelog/1.3.1.md),
[`changelog/1.4.0.md`](changelog/1.4.0.md),
[`changelog/1.5.0.md`](changelog/1.5.0.md),
[`changelog/1.6.0.md`](changelog/1.6.0.md) and
[`changelog/1.7.0.md`](changelog/1.7.0.md).

One residual stays open: **`CPP-05`**. The vcpkg port targets `1.1.0` with
the real source-tarball SHA512, so it is installable, but it was not bumped
for `1.2.0`/`1.3.0` and has never been exercised by a real `vcpkg install` on
any platform. Cross-platform
verification and the registry PR are tracked in
[`ports/README.md`](../ports/README.md).

## Post-1.0 program: accuracy-aware detection (`UQ-*`)

**Goal.** Every corner reports how far it can be trusted, and that report is
proven against truth. Downstream calibration (calibration-rs `UQ-FEATURE-COV`)
whitens its reprojection residuals with a per-corner 2×2 covariance and
expects the a-posteriori variance factor σ̂₀ ≈ 1 when the detector's model
holds.

**Error has two parts, and the program treats them separately.**
- *Precision* — scatter caused by image noise. A first-order covariance
  describes it; the saddle-point refiner reports one (per unit noise
  variance).
- *Bias* — systematic model error: a quadratic fitted to a blurred step
  corner, a centroid on an asymmetric patch. A noise covariance says nothing
  about it. SaddlePoint's clean-image mean error is ~0.11 px with **zero
  noise** (`reference/refiner-comparison.md`), so at typical noise levels bias
  may dominate. If it does, a noise-only covariance would make calibration
  overconfident, which is worse than uniform weighting.

**Evidence before API.** The public per-corner uncertainty API is designed
from a measured bias/precision split, not ahead of it.

| Release | Scope | Tasks |
|---------|-------|-------|
| `1.4.0` | Saddle-point position covariance in `RefineResult`; the refiner is also faster | `UQ-00` |
| next | Bias/precision benchmark per refiner; published noise-row numbers re-measured (the test noise helper produced 1.3× the stated σ before the fix) | `UQ-01..02` |
| decision | `UQ-01` found SaddlePoint bias (~0.12 px on sharp corners) far above noise scatter, removable by re-centred refitting: reduce bias first, then expose the covariance | — |
| `1.5.0` | SaddlePoint re-centred refit (default) with a validated covariance; per-image noise estimate (px² output); SaddlePoint covariance on `CornerDescriptor` and in the Python / WASM / C APIs and the JSON schema; CenterOfMass non-refinement fix; a book Part on uncertainty; camera-formed (lens blur before pixel integration) evidence fixtures and the floor tuned on rotated boards | `UQ-03`, `UQ-05`, `UQ-07`, `UQ-09`, `UQ-15` |
| `1.6.0` (shipped) | Position covariance for every localizer behind `Uncertainty { Off, Propagated }` (noise through each localizer's Jacobian plus a per-localizer systematic term with a real-data floor); opt-in SaddlePoint refinement of Radon corners (`RadonRefiner`; default stays the peak fit, `UQ-22`); per-image noise estimate public in every binding; robustness audit and real-data validation; book Part VIII rewritten on that evidence | `UQ-18`, `UQ-23`–`UQ-28`, `UQ-30` |
| `1.7.0` (shipped) | Evidence first, then hardening. The per-corner systematic model is kept: dropping it loses the ranking on real data (`UQ-36`). The floor is not transferable across cameras, and a cross-camera refit failed its pre-registered rule, so the shipped values stay with the limit documented (`UQ-37`, `UQ-44`). Weighting by Σ never helped, so `Off` is now the default (`UQ-38`). Also: honest docs for `Propagated` and orientation σ (`UQ-41`), two-sided σ̂₀² tests (`UQ-39`), the frame σₙ in every binding (`UQ-40`, C ABI 7), binding consistency (`UQ-42`), covariance code DRY (`SOLID-06`), a perf guard for the opt-in cost (`PERF-16`/`17`), and redundant tooling removed (`SOLID-07`) | `UQ-36`–`UQ-42`, `UQ-44`, `SOLID-06`/`07`, `PERF-16`/`17` |
| next | A second real dataset to settle the floor (`UQ-45`); end-to-end σ̂₀ in calibration-rs (`UQ-08`); one orientation σ definition, coordinated with calib-targets-rs, which gates and weights on it (`UQ-06`) | `UQ-45`, `UQ-08`, `UQ-06` |
| later | Per-corner systematic term that sees board rotation and Radon peak switching; cubic re-centring for SaddlePoint; a noise estimate for correlated noise | `UQ-16`, `UQ-32` |
| `2.0.0` | Uncertainty API: noise level inside the tier (`Propagated { noise }`), no public covariance mirror fields, no hidden public helpers (both deprecated in 1.7.0); refiner lifted to one stage | `UQ-43`, `UQ-19` |
| ongoing | Realistic ground truth (calib-synth / etendue renders; end-to-end σ̂₀ check in calibration-rs) | `UQ-08`, `ML-01`, `ML-02` |

**Known gaps.**
- Weighting a calibration by Σ has not beaten uniform weighting on real data,
  so the covariance is opt-in (`Off` by default from 1.7.0). The end-to-end
  σ̂₀ check in calibration-rs (`UQ-08`) has not been run.
- The real-data floor is fitted on one camera pair and does not transfer
  across cameras (held-out σ̂₀² 1.5–5.5); a second dataset decides it
  (`UQ-45`).
- The systematic term models a window-offset pull in an edge-width
  observable plus a constant floor fitted on one real calibration set; it
  cannot see board rotation (CenterOfMass and Förstner per-condition
  calibration) or the Radon peak fit's switching on flat-topped responses
  (σ̂₀² up to ~6 on blurred near-45° corners).
- Per-view errors common to all corners of an image (defocus, tilt: ~0.06 px
  per axis on the real set) are outside any per-corner covariance.
- The noise estimate misreads demosaiced images and declines on JPEG;
  users must supply σ there (`UQ-32`).
- Real-data evidence comes from one stereo ChArUco set; no repeated-capture
  or stage-truth data yet (`UQ-30`).
- Orientation σ is a fit-quality score, not a calibrated 1σ: RingFit scales a
  Gauss-Newton covariance by the fit residual plus a calibration factor;
  DiskFit is a heuristic. calib-targets-rs gates and weights on it (`UQ-06`).
- The detector drops refine score and status before `CornerDescriptor`
  (`detect/chess/detect.rs`, `refine_corners_on_image`); the covariance is
  carried (`UQ-05`).

## Continuous

- **`SWEEP-*`** — keep public surfaces (book/README/rustdoc/CHANGELOG) free of
  dev-history/lineage/origin references.
- **`SOLID-*`** — DRY/cohesion cleanup (shared test utils, dispatch, fixtures).
- **`SKILL-*`** — keep the reusable user-level skill set current with the
  patterns this program produces (bindings, docs-site assembly, KB restructure).

## Out of scope (for now)

Anything requiring a post-1.0 breaking change; new detector algorithms.
Additive accuracy work (`UQ-*`: new optional fields on `#[non_exhaustive]`
types, new refiner outputs) is in scope. The C ABI is not semver-checked, so a
`cc_corner` layout change bumps `cc_abi_version` instead.
Stable-Rust SIMD was evaluated (`PERF-11`) and rejected: no stable backend
matches the nightly `std::simd` path without regressing aarch64 below scalar or
breaking the bit-exact pyramid, so `simd` stays nightly-only as an optional
high-performance path and the stable scalar/autovec build is the supported
portable baseline.
