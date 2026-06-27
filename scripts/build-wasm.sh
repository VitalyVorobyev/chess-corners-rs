#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building WASM package..."
wasm-pack build "$ROOT_DIR/crates/chess-corners-wasm" \
  --target web \
  --release \
  --out-dir "$ROOT_DIR/demo/pkg" \
  --out-name chess_corners_wasm

# Ship the crate README inside the npm package (it documents the JS API).
if [ -f "$ROOT_DIR/crates/chess-corners-wasm/README.md" ]; then
  cp "$ROOT_DIR/crates/chess-corners-wasm/README.md" "$ROOT_DIR/demo/pkg/README.md"
fi

# Append hand-written object-shape types to the auto-generated .d.ts so
# consumers of the npm package get typed result/parameter shapes (the
# wasm-bindgen output only types function signatures). Skipped when the
# crate does not ship one.
if [ -f "$ROOT_DIR/crates/chess-corners-wasm/typescript-extras.d.ts" ]; then
  cat "$ROOT_DIR/crates/chess-corners-wasm/typescript-extras.d.ts" \
    >> "$ROOT_DIR/demo/pkg/chess_corners_wasm.d.ts"
fi

# Override the published npm name (wasm-pack derives it from the Rust crate
# name; we ship as the scoped public package @vitavision/chess-corners).
(cd "$ROOT_DIR/demo/pkg" && npm pkg set name=@vitavision/chess-corners)

echo "WASM package built to demo/pkg/"
ls -lh "$ROOT_DIR/demo/pkg/"*.wasm
