# Design: GitHub Pages site (book + API + demo + performance)

**Status:** Draft / design. **Workstream:** `SITE-*` (ROADMAP milestone **M4**,
depends on **M3** so the demo/docs target a frozen API). **Source pattern:**
ported from the sibling project `calib-targets-rs`
(`/Users/vitalyvorobyev/vision/calib-targets-rs`).

## Current state (what we are changing)

`.github/workflows/docs.yml` today publishes only two things and puts the
**book at the site root**:

```
public/         ← mdBook (book/book/)         = site root "/"
public/api/     ← rustdoc (target/doc/)
```

There is **no** landing page, **no** demo, and **no** performance page. A
stale `.github/pages/index.html` (titled `calib-targets-rs`, never deployed)
exists and must be reconciled.

## Target information architecture

A single landing page at `/` is the entry point, with four destinations:

```
/                ← landing page (4 cards)         (.github/pages/index.html)
/book/           ← mdBook guide                   (book/book/)
/api/            ← rustdoc                         (target/doc/)
/demo/           ← interactive WASM demo          (demo/dist/)
/performance/    ← perf report                    (.github/pages/performance/)
```

The four cards: **Guide** → `/book/`, **API reference** → `/api/`,
**Demo** → `/demo/`, **Performance** → `/performance/`. Dark theme, hero
title, project description, footer linking GitHub + vitavision.dev — matching
the calib-targets-rs look so the user's projects share an identity.

> **Breaking note:** the book moves from `/` to `/book/`. Update README badges
> and any absolute links; the landing page at `/` handles the old entry point.

## Build pipeline (`docs.yml` rework — `SITE-02`)

Extend the existing workflow's toolchain (it already has nightly Rust + mdBook)
with: `wasm32-unknown-unknown` target, `wasm-pack`, and `bun`. Build steps:

1. `cargo doc --workspace --all-features --no-deps` → `target/doc/`; write the
   redirect `index.html` (already done today).
2. `mdbook build book` → `book/book/`.
3. `scripts/build-wasm.sh` → wasm-pack build into `demo/pkg/`; then
   `bun install && bun run build` in `demo/` → `demo/dist/`.
4. `scripts/gen-perf-data.sh` → `.github/pages/performance/data.json` +
   overlay images.
5. Assemble `public/`:
   ```
   mkdir -p public/{api,book,demo}
   rsync -a .github/pages/ public/      # landing + performance/
   rsync -a target/doc/   public/api/
   rsync -a book/book/    public/book/
   rsync -a demo/dist/    public/demo/
   ```
6. Existing `deploy` job uploads `public/` (unchanged).

## Components to build

### Landing page — `SITE-01`

Rewrite `.github/pages/index.html` for chess-corners (title, description, the
four cards, correct links, vitavision.dev footer). Keep the calib-targets
visual design (radial-gradient dark theme, colored cards).

### WASM demo — `SITE-03`

- `demo/` is a Vite + React app built with Bun, depending on the built
  `@vitavision/chess-corners` package from `demo/pkg/` (consistent with the
  sibling's `@vitavision/calib-targets`).
- `scripts/build-wasm.sh`: `wasm-pack build crates/chess-corners-wasm --target
  web --release --out-dir demo/pkg`, set the npm package name, append any
  TypeScript extras (`typescript-extras.d.ts`) as the sibling does.
- Demo UX: upload/select an image, run detection in-browser, overlay detected
  corners + orientation axes, expose key config knobs (threshold, strategy,
  refiner, multiscale). This is the showcase for the WASM bindings.

### Performance page — `SITE-04`

- `scripts/gen-perf-data.sh` runs a timing binary on the public `testimages/`,
  emitting per-stage timings + corner counts to
  `.github/pages/performance/data.json` plus detection-overlay images. Reuse
  the logic in `tools/perf_bench.py` (and the baselines from the perf
  workstream — see [`perf-profiling.md`](perf-profiling.md), `PERF-09`).
- `.github/pages/performance/index.html` renders `data.json` as an interactive
  report (no criterion HTML; custom schema, like the sibling).

### Local build — `SITE-05`

`scripts/build-site.sh` runs steps 1–5 locally (cargo doc → mdBook → wasm →
bun → assemble `public/`) so the site is reproducible without CI.

### Cross-links — `SITE-06`

Update README + book for the `/book/` move and add the live-site/demo links;
wire the vitavision.dev linkage.

## Dependencies & sequencing

- **Depends on M3 (API freeze):** the demo and rustdoc should reflect the
  frozen surface (no `contrast`/`fit_rms`; final config shape) to avoid
  rebuilding the demo UI and docs after a breaking change.
- **Consumes M2 output:** the performance page is richer if `PERF-09`
  baselines exist, but can ship with a first data set generated from
  `tools/perf_bench.py`.

## Reference

Sibling files to mirror (read at implementation time):
`/Users/vitalyvorobyev/vision/calib-targets-rs/.github/workflows/docs.yml`,
`/.github/pages/index.html`, `/scripts/{build-wasm,gen-perf-data,build-site}.sh`,
`/demo/`.

---

Task list: [`../BACKLOG.md`](../BACKLOG.md) `SITE-*`.
