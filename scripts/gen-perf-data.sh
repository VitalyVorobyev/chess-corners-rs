#!/usr/bin/env bash
# Regenerate the PUBLIC performance-report data + overlays consumed by
# .github/pages/performance/index.html.
#
# Uses the PUBLIC testimages/ ONLY (small/mid/large). Everything it
# writes — .github/pages/performance/data.json and img/*.png — is
# committed and published, so it must stay public.
#
# Steps:
#   1. `perf_overlay` measures the single-scale ChESS pipeline per stage
#      (response / detection / refinement / orientation) on each image,
#      warmup + REPEATS iterations, reporting the p50, and writes a
#      detection-overlay PNG per image into the published img/ dir.
#   2. A guard aborts before the merge if no measurement was produced.
#   3. gen_perf_data.py merges the measured numbers + host metadata into
#      data.json, preserving the editorial prose. total_ms / throughput
#      are re-derived from the per-stage p50s there.
#
# Usage:
#   bash scripts/gen-perf-data.sh
#   REPEATS=200 WARMUP=20 FEATURES=simd bash scripts/gen-perf-data.sh
#
# `set -e` plus the guard are load-bearing: a failed `cargo run` must
# abort BEFORE the merge, otherwise the merger would treat the missing
# raw file as "no measurement" and leave a stale data.json while exiting
# successfully.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPEATS="${REPEATS:-60}"
WARMUP="${WARMUP:-8}"
FEATURES="${FEATURES:-simd}"
RAW="$(mktemp -d)"
trap 'rm -rf "$RAW"' EXIT

log() { printf '\n==== %s ====\n' "$*"; }

# ---- 1. Per-stage timing + detection overlays on the public images ----
out="$RAW/perf.json"
log "perf_overlay (small/mid/large; per-stage p50 + detection overlays)"
cargo run --release -q -p chess-corners --example perf_overlay --features "$FEATURES" -- \
  --repeats "$REPEATS" --warmup "$WARMUP" \
  --out "$out" \
  --overlay-dir .github/pages/performance/img

# ---- 2. Guard: never merge unless the measurement produced output ----
if [[ ! -s "$out" ]]; then
  echo "ERROR: missing or empty measurement: $out — aborting before merge." >&2
  exit 1
fi

# ---- 3. Host metadata (public) ----
if [[ "$FEATURES" == *rayon* ]]; then THREADS="multi"; else THREADS="single"; fi
export PERF_CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -mp)"
export PERF_RUSTC="$(rustc --version 2>/dev/null || echo unknown)"
export PERF_GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export PERF_FEATURES="$FEATURES"
export PERF_THREADS="$THREADS"
export PERF_REPEATS="$REPEATS"
export PERF_WARMUP="$WARMUP"

# ---- 4. Merge measured numbers + meta into data.json (editorial kept) ----
log "merge -> .github/pages/performance/data.json"
python3 scripts/gen_perf_data.py "$RAW" .github/pages/performance/data.json

echo "Done. Review the diff in .github/pages/performance/ before committing."
