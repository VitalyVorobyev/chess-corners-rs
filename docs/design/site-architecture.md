# Design: GitHub Pages site (book + API + demo + performance)

**Status:** Implemented (M4). **Workstream:** `SITE-*` (ROADMAP milestone **M4**,
depends on **M3** so the demo/docs target a frozen API). Per-task status lives in
[`../BACKLOG.md`](../BACKLOG.md) `SITE-*`.

## Information architecture (as shipped)

A single landing page at `/` is the entry point, with four destinations:

```
/                ← landing page (4 cards)         (.github/pages/index.html)
/book/           ← mdBook guide                   (book/book/)
/api/            ← rustdoc                         (target/doc/)
/demo/           ← interactive WASM demo          (demo/dist/)
/performance/    ← perf report                    (.github/pages/performance/)
```

The book moved from `/` to `/book/`; the landing page now owns the root entry
point. The dark theme, hero, and footer (GitHub + vitavision.dev) match the
sibling `calib-targets-rs` so the projects share an identity.

## Build pipeline

`.github/workflows/docs.yml` (`SITE-02`) builds and assembles `public/`:

1. `cargo doc --workspace --all-features --no-deps` → `/api/` (+ `chess_corners`
   redirect).
2. `mdbook build book` → `/book/`.
3. `scripts/build-wasm.sh` (wasm-pack → `demo/pkg/`) + `bun run build` in `demo/`
   → `/demo/`.
4. `scripts/gen-perf-data.sh` → `performance/data.json` + overlay images.
5. `rsync` landing + performance + api + book + demo into `public/`; existing
   `deploy` job uploads it.

`scripts/build-site.sh` (`SITE-05`) reproduces the deployed tree locally.

## Components

- **Landing (`SITE-01`)** — `.github/pages/index.html`, four cards (Guide / API
  reference / Demo / Performance).
- **WASM demo (`SITE-03`)** — Vite + React + Bun over `@vitavision/chess-corners`:
  upload/select an image, detect in-browser (ChESS/Radon), overlay corners +
  orientation axes + σ wedges + response/Radon heatmap, live config controls.
- **Performance page (`SITE-04`)** — `gen_perf_data.py` + a `perf_overlay`
  example emit real per-stage timings + corner counts + overlays; the page
  renders `data.json` as an interactive report.

## Dependencies

- **Depends on M3 (API freeze):** the demo and rustdoc bind the frozen surface
  (no `contrast`/`fit_rms`).
- **Consumes M2 output:** the performance page uses the `PERF-09` baselines (see
  [`perf-profiling.md`](perf-profiling.md)).

---

Task list: [`../BACKLOG.md`](../BACKLOG.md) `SITE-*`.
