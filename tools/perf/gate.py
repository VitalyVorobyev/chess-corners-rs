#!/usr/bin/env python3
"""Benchmark-regression gate for chess-corners-rs (PERF-08 / PERF-09).

This tool drives a *same-runner* comparison of a curated, low-variance subset
of the workspace's Criterion benches and fails CI when any gated bench's median
regresses beyond a budget (default 2%).

Why same-runner head-vs-merge-base
----------------------------------
Absolute, committed benchmark numbers compared across machines/runs are
hopelessly flaky at a 2% budget: GitHub-hosted runners differ by CPU model,
neighbour load and frequency scaling (often 10-30% run-to-run). The gate
therefore measures BOTH sides inside one CI job on one runner:

    gate.py run pr      # PR head        -> Criterion baseline "pr"
    <checkout merge-base sources>
    gate.py run base    # merge-base/main -> Criterion baseline "base"
    gate.py check base pr

Measuring base and pr back-to-back on the same physical CPU cancels the
machine-model and most of the steady-state-frequency variance; what remains is
sampling variance plus slow thermal drift between the two runs. The gated
subset (see gated-benches.json) was picked from measured no-op deltas so that
residual is comfortably under the budget.

Median, not p95
---------------
The pass/fail compares Criterion's MEDIAN point estimate -- the same statistic
`critcmp` reports. A true p95 would require parsing each
`target/criterion/<id>/new/sample.json`; the median is the pragmatic, standard
mechanism and is what critcmp/cargo-criterion users reason about.

Why critcmp --export rather than the critcmp table
--------------------------------------------------
`check` prints the human-readable `critcmp base pr` table into the CI log, but
enforces the threshold from `critcmp --export <baseline>` JSON. The export
carries the FULL-PRECISION median (`criterion_estimates_v1.median.point_estimate`)
keyed by `full_id`, whereas the printed table rounds its ratio to two decimals
-- too coarse to decide a 2% budget reliably. Same data source critcmp uses,
just without the display rounding. Robustness over cleverness.

Drift tolerance
---------------
A gated id is enforced only when it is present in BOTH baselines. An id missing
from `base` (a bench added in the PR) is reported and skipped; an id missing
from `pr` (a bench renamed/removed in the PR) is reported and skipped. This
keeps the gate from false-failing when a parallel task refactors the bench
fixtures, while still failing on genuine regressions of benches present on both
sides. If NOTHING comparable is found at all, the gate fails closed -- a broken
setup must not masquerade as green.

Subcommands
-----------
    run <baseline> [--strict]   Run the gated benches, save Criterion baseline.
    check <base> <pr>           Compare medians; exit non-zero on regression.
    snapshot <baseline> <out>   Write a committed metrics.json reference snapshot.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("gated-benches.json")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not cfg.get("targets"):
        raise SystemExit(f"error: no targets in {path}")
    return cfg


def expected_ids(cfg: dict) -> list[str]:
    ids: list[str] = []
    for target in cfg["targets"]:
        ids.extend(target["ids"])
    return ids


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    warmup = os.environ.get("PERF_WARMUP", str(cfg.get("warm_up_time_s", 2)))
    meas = os.environ.get("PERF_MEAS", str(cfg.get("measurement_time_s", 6)))

    failures: list[str] = []
    for target in cfg["targets"]:
        cmd = ["cargo", "bench", "-p", target["crate"]]
        if target.get("features"):
            cmd += ["--features", ",".join(target["features"])]
        cmd += ["--bench", target["bench"], "--"]
        if target.get("filter"):
            cmd.append(target["filter"])
        cmd += [
            "--save-baseline",
            args.baseline,
            "--warm-up-time",
            warmup,
            "--measurement-time",
            meas,
        ]
        print(f"\n==== {' '.join(cmd)} ====", flush=True)
        proc = subprocess.run(cmd, cwd=os.getcwd(), check=False)
        if proc.returncode != 0:
            failures.append(target["bench"])
            msg = f"bench target '{target['bench']}' exited {proc.returncode}"
            if args.strict:
                print(f"error: {msg}", file=sys.stderr)
            else:
                # Lenient: the merge-base may predate a bench that HEAD added;
                # the missing id is handled (skipped) by `check`.
                print(f"warning: {msg} (continuing; lenient run)", file=sys.stderr)

    if failures and args.strict:
        print(f"error: {len(failures)} bench target(s) failed under --strict", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# shared: read critcmp --export JSON
# --------------------------------------------------------------------------- #
def critcmp_export(baseline: str) -> dict[str, dict]:
    """Return {full_id: {median_ns, throughput}} for a saved Criterion baseline."""
    proc = subprocess.run(
        ["critcmp", "--export", baseline],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    out: dict[str, dict] = {}
    # `benchmarks` is a JSON object keyed by full name; iterate its values.
    for bench in data.get("benchmarks", {}).values():
        meta = bench["criterion_benchmark_v1"]
        est = bench["criterion_estimates_v1"]
        out[meta["full_id"]] = {
            "median_ns": est["median"]["point_estimate"],
            "throughput": meta.get("throughput"),
        }
    return out


def print_critcmp_table(base: str, pr: str) -> None:
    """Render the human-readable comparison into the CI log (best-effort)."""
    proc = subprocess.run(
        ["critcmp", base, pr, "--color", "never"],
        check=False,
        capture_output=True,
        text=True,
    )
    print("---- critcmp", base, pr, "----")
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    threshold = float(os.environ.get("REGRESSION_THRESHOLD_PCT", cfg.get("threshold_pct", 2.0)))
    ids = expected_ids(cfg)

    print_critcmp_table(args.base, args.pr)

    base = critcmp_export(args.base)
    pr = critcmp_export(args.pr)

    compared = 0
    regressions: list[tuple[str, float]] = []
    rows: list[str] = []
    for bid in ids:
        b = base.get(bid)
        p = pr.get(bid)
        if b is None and p is None:
            rows.append(f"  SKIP  {bid}: absent from both baselines (stale config?)")
            continue
        if b is None:
            rows.append(f"  SKIP  {bid}: new in PR (no merge-base baseline)")
            continue
        if p is None:
            rows.append(f"  SKIP  {bid}: absent from PR (bench renamed/removed?)")
            continue
        delta = p["median_ns"] / b["median_ns"] - 1.0
        compared += 1
        tag = "FAIL" if delta * 100.0 > threshold else "ok  "
        rows.append(
            f"  {tag}  {bid}: base {b['median_ns'] / 1e3:9.2f} us  "
            f"pr {p['median_ns'] / 1e3:9.2f} us  delta {delta * 100.0:+6.2f}%"
        )
        if delta * 100.0 > threshold:
            regressions.append((bid, delta * 100.0))

    print(f"\n---- regression gate (median, budget {threshold:.2f}%) ----")
    print("\n".join(rows))

    if compared == 0:
        print(
            "\nerror: 0 gated benches were comparable across both baselines; "
            "failing closed (broken setup?).",
            file=sys.stderr,
        )
        return 1
    if regressions:
        print(f"\nFAILED: {len(regressions)} bench(es) regressed > {threshold:.2f}%:", file=sys.stderr)
        for bid, pct in regressions:
            print(f"  {bid}: {pct:+.2f}%", file=sys.stderr)
        return 1
    print(f"\nPASSED: {compared} gated bench(es) within {threshold:.2f}% budget.")
    return 0


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #
def _throughput_summary(median_ns: float, throughput: dict | None) -> dict | None:
    if not throughput:
        return None
    for kind in ("Elements", "Bytes"):
        count = throughput.get(kind)
        if count:
            per_s = count / (median_ns * 1e-9)
            unit = "elem/s" if kind == "Elements" else "bytes/s"
            return {"kind": kind.lower(), "count": count, "per_second": round(per_s, 1), "unit": unit}
    return None


def cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = critcmp_export(args.baseline)
    ids = expected_ids(cfg)

    benches = []
    missing = []
    for bid in ids:
        rec = data.get(bid)
        if rec is None:
            missing.append(bid)
            continue
        benches.append(
            {
                "id": bid,
                "median_ns": round(rec["median_ns"], 1),
                "throughput": _throughput_summary(rec["median_ns"], rec["throughput"]),
            }
        )
    if missing:
        raise SystemExit(
            "error: baseline '%s' is missing gated ids: %s\n"
            "Run `gate.py run %s` first." % (args.baseline, ", ".join(missing), args.baseline)
        )

    def _git(*cmd: str) -> str:
        try:
            return subprocess.run(
                ["git", *cmd], check=True, capture_output=True, text=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            return "unknown"

    snapshot = {
        "note": (
            "Machine-stamped reference snapshot of the gated bench medians. The "
            "LIVE regression gate (bench-gate.yml) does NOT read this file -- it "
            "compares the PR head against its own merge-base on one runner. This "
            "file is for human tracking and may feed the perf page (SITE-04). It "
            "is distinct from .github/pages/performance/data.json."
        ),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "toolchain": cfg.get("toolchain", "nightly"),
        "features": sorted({f for t in cfg["targets"] for f in t.get("features", [])}),
        "threshold_pct": cfg.get("threshold_pct", 2.0),
        "host": {
            "cpu": _host_cpu(),
            "os": f"{platform.system()} {platform.release()}",
            "rustc": _rustc_version(),
        },
        "benches": benches,
    }
    out = Path(args.out)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(benches)} bench medians to {out}")
    return 0


def _host_cpu() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


def _rustc_version() -> str:
    try:
        return subprocess.run(
            ["rustc", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to gated-benches.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run gated benches and save a Criterion baseline")
    p_run.add_argument("baseline", help="Criterion baseline name (e.g. pr or base)")
    p_run.add_argument("--strict", action="store_true", help="fail if any bench target errors")
    p_run.set_defaults(func=cmd_run)

    p_check = sub.add_parser("check", help="compare two baselines and enforce the budget")
    p_check.add_argument("base")
    p_check.add_argument("pr")
    p_check.set_defaults(func=cmd_check)

    p_snap = sub.add_parser("snapshot", help="write a committed metrics.json snapshot")
    p_snap.add_argument("baseline")
    p_snap.add_argument("out")
    p_snap.set_defaults(func=cmd_snapshot)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
