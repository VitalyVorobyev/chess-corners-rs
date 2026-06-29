#!/usr/bin/env python3
"""Merge measured PUBLIC perf numbers into the committed performance data.

Reads the raw per-image measurements produced by the `perf_overlay`
example (run by scripts/gen-perf-data.sh) and refreshes ONLY the measured
fields of .github/pages/performance/data.json. Editorial content — each
card's label, file and note, plus the meta note — is preserved.

Each card carries a `configs` array: the measured algorithm matrix
(ChESS × {refiner} × {orientation}, Radon × {refiner} × {orientation}),
keyed into the card by image basename. Per-config `total_ms` /
`throughput_mpix_s` are re-derived from the measured stage p50s here so
the published totals can never drift from the per-stage breakdown. The
per-image `width`, `height` and `overlays` (chess/radon preview paths)
are merged too.

Host metadata (cpu / rustc / git_sha / features / threads / repeats /
warmup) is read from environment variables exported by gen-perf-data.sh;
an unset variable leaves the existing value untouched. `generated` is set
to today (UTC).

Public data only — never read or emit private numbers.

Usage: python3 scripts/gen_perf_data.py <raw_dir> <data_json_path>
"""
import json
import os
import re
import sys
from datetime import datetime, timezone


def load(path):
    with open(path) as f:
        return json.load(f)


def measured_by_file(raw_dir):
    """Return {basename(file): image_obj} from <raw_dir>/perf.json, if present."""
    path = os.path.join(raw_dir, "perf.json")
    if not os.path.exists(path):
        return {}
    doc = load(path)
    return {os.path.basename(img["file"]): img for img in doc.get("images", [])}


def env(name):
    """Environment value, or None if unset/empty."""
    v = os.environ.get(name, "")
    return v if v else None


def apply_meta(meta):
    for key, var in (
        ("cpu", "PERF_CPU"),
        ("git_sha", "PERF_GIT_SHA"),
        ("features", "PERF_FEATURES"),
        ("threads", "PERF_THREADS"),
    ):
        val = env(var)
        if val is not None:
            meta[key] = val

    rustc = env("PERF_RUSTC")
    if rustc:
        m = re.search(r"\d+\.\d+\.\d+(?:-\S+)?", rustc)
        meta["rustc"] = m.group(0) if m else rustc

    for key, var in (("repeats", "PERF_REPEATS"), ("warmup", "PERF_WARMUP")):
        val = env(var)
        if val is not None:
            try:
                meta[key] = int(val)
            except ValueError:
                pass

    meta["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def merge_config(c, px):
    """Round a measured config's stages and re-derive total_ms / throughput.

    `px` is the image pixel count, used to recompute the per-config
    throughput from the re-derived total so totals never drift from the
    per-stage breakdown.
    """
    stages = {k: round(float(v), 4) for k, v in c["stages"].items()}
    total = round(sum(stages.values()), 4)
    throughput = round(px / (total * 1000.0), 2) if total > 0 else 0.0
    return {
        "id": c["id"],
        "detector": c["detector"],
        "refiner": c["refiner"],
        "orientation": c["orientation"],
        "corner_count": c["corner_count"],
        "stages": stages,
        "total_ms": total,
        "throughput_mpix_s": throughput,
    }


def main():
    raw_dir, data_path = sys.argv[1], sys.argv[2]
    data = load(data_path)
    measured = measured_by_file(raw_dir)

    apply_meta(data.setdefault("meta", {}))

    if not measured:
        print("WARNING: no perf.json measurements found — leaving image numbers as-is.")
    else:
        # Legacy single-config per-image fields, dropped in favour of `configs`.
        legacy_keys = ("img", "corner_count", "stages", "total_ms", "throughput_mpix_s")
        for card in data.get("images", []):
            m = measured.get(os.path.basename(card["file"]))
            if not m:
                continue
            card["width"] = m["width"]
            card["height"] = m["height"]
            card["overlays"] = m["overlays"]
            px = m["width"] * m["height"]
            card["configs"] = [merge_config(c, px) for c in m["configs"]]
            for k in legacy_keys:
                card.pop(k, None)

    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Updated {data_path}")


if __name__ == "__main__":
    main()
