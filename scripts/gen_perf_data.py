#!/usr/bin/env python3
"""Merge measured PUBLIC perf numbers into the committed performance data.

Reads the raw per-image measurements produced by the `perf_overlay`
example (run by scripts/gen-perf-data.sh) and refreshes ONLY the numeric
fields of .github/pages/performance/data.json. Editorial content — each
card's label, file, img and note, plus the meta note — is preserved.

Host metadata (cpu / rustc / git_sha / features / threads / repeats /
warmup) is read from environment variables exported by gen-perf-data.sh;
an unset variable leaves the existing value untouched. `generated` is set
to today (UTC).

`total_ms` and `throughput_mpix_s` are derived from the measured stage
p50s here so the published totals can never drift from the per-stage
breakdown.

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


def main():
    raw_dir, data_path = sys.argv[1], sys.argv[2]
    data = load(data_path)
    measured = measured_by_file(raw_dir)

    apply_meta(data.setdefault("meta", {}))

    if not measured:
        print("WARNING: no perf.json measurements found — leaving image numbers as-is.")
    else:
        for card in data.get("images", []):
            m = measured.get(os.path.basename(card["file"]))
            if not m:
                continue
            card["width"] = m["width"]
            card["height"] = m["height"]
            card["corner_count"] = m["corner_count"]
            stages = {k: round(float(v), 4) for k, v in m["stages"].items()}
            card["stages"] = stages
            total = round(sum(stages.values()), 4)
            card["total_ms"] = total
            px = m["width"] * m["height"]
            card["throughput_mpix_s"] = round(px / (total * 1000.0), 2) if total > 0 else 0.0

    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Updated {data_path}")


if __name__ == "__main__":
    main()
