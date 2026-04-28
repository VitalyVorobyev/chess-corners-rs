#!/usr/bin/env bash
# Wrapper around `cargo-flamegraph` and `samply` for the chess-corners
# detector hot paths. Builds the `profile_target` example with
# debug-info release flags and invokes the chosen profiler.
#
# Usage:
#   tools/profile.sh chess <image>           # ChESS multiscale
#   tools/profile.sh radon <image>           # whole-image Radon
#   tools/profile.sh refiner <kind> <image>  # ChESS multiscale + refiner
#   tools/profile.sh samply <chess|radon> <image>  # samply variant (UI)
#
#   tools/profile.sh -h
#
# Requires either `cargo-flamegraph` or `samply` on $PATH:
#   cargo install flamegraph
#   cargo install samply
#
# Outputs land in `testdata/out/profiles/` with timestamped filenames.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/testdata/out/profiles"
mkdir -p "$OUT_DIR"

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "profile.sh: $*" >&2
    exit 1
}

require_image() {
    [[ -n "${1:-}" ]] || die "missing <image> path"
    [[ -f "$1" ]] || die "image not found: $1"
}

build_target() {
    local features="${PROFILE_FEATURES:-simd,rayon,par_pyramid}"
    echo "==> building profile_target (features=$features)" >&2
    CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
        cargo build --release --features "$features" \
            --example profile_target -p chess-corners
}

target_bin() {
    echo "$ROOT/target/release/examples/profile_target"
}

timestamp() {
    date +%Y%m%dT%H%M%S
}

run_flamegraph() {
    local label="$1"; shift
    local output="$OUT_DIR/${label}-$(timestamp).svg"
    echo "==> cargo flamegraph -> $output" >&2
    # cargo-flamegraph needs cargo to drive the build/run; invoke it
    # against the same example to keep the symbol set consistent.
    CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
        cargo flamegraph --release \
            --features "${PROFILE_FEATURES:-simd,rayon,par_pyramid}" \
            --example profile_target -p chess-corners \
            -o "$output" \
            -- "$@"
    echo "wrote $output"
}

run_samply() {
    local label="$1"; shift
    local output="$OUT_DIR/${label}-$(timestamp).json.gz"
    build_target
    local bin
    bin="$(target_bin)"
    echo "==> samply record -> $output" >&2
    samply record --save-only -o "$output" -- "$bin" "$@"
    echo "wrote $output (open with: samply load $output)"
}

choose_profiler() {
    if command -v cargo-flamegraph >/dev/null 2>&1; then
        echo "flamegraph"
    elif command -v samply >/dev/null 2>&1; then
        echo "samply"
    else
        die "neither cargo-flamegraph nor samply is installed; install one (cargo install flamegraph / samply)"
    fi
}

cmd="${1:-}"
[[ -n "$cmd" ]] || { usage; exit 1; }
shift || true

case "$cmd" in
    -h|--help|help)
        usage
        ;;
    chess|radon)
        image="${1:-}"
        require_image "$image"
        label="${cmd}-$(basename "$image" .png)"
        case "$(choose_profiler)" in
            flamegraph)
                run_flamegraph "$label" --mode "$cmd" --image "$image" --iters 200
                ;;
            samply)
                run_samply "$label" --mode "$cmd" --image "$image" --iters 200
                ;;
        esac
        ;;
    refiner)
        kind="${1:-}"
        image="${2:-}"
        [[ -n "$kind" ]] || die "refiner requires <kind>"
        require_image "$image"
        label="refiner-${kind}-$(basename "$image" .png)"
        case "$(choose_profiler)" in
            flamegraph)
                run_flamegraph "$label" --mode chess --refiner "$kind" --image "$image" --iters 200
                ;;
            samply)
                run_samply "$label" --mode chess --refiner "$kind" --image "$image" --iters 200
                ;;
        esac
        ;;
    samply)
        sub="${1:-}"
        image="${2:-}"
        [[ "$sub" == "chess" || "$sub" == "radon" ]] || die "samply expects <chess|radon> <image>"
        require_image "$image"
        label="samply-${sub}-$(basename "$image" .png)"
        run_samply "$label" --mode "$sub" --image "$image" --iters 200
        ;;
    *)
        usage
        die "unknown command: $cmd"
        ;;
esac
