from typing import Dict
import re
import json
# Duration helpers ---------------------------------------------------------

_DURATION_RE = re.compile(r"(?P<val>[\d.]+)(?P<unit>[mun]?s)")


def parse_duration_ms(raw: str) -> float:
    """Convert strings like '1.2ms', '42.1µs', '750ns' to milliseconds."""
    clean = raw.replace("µ", "u")
    m = _DURATION_RE.fullmatch(clean)
    if not m:
        raise ValueError(f"Unrecognized duration format: {raw}")
    val = float(m.group("val"))
    unit = m.group("unit")
    if unit == "s":
        return val * 1000.0
    if unit == "ms":
        return val
    if unit == "us":
        return val / 1000.0
    if unit == "ns":
        return val / 1_000_000.0
    raise ValueError(f"Unhandled duration unit in {raw}")

def parse_trace(stdout: str) -> Dict[str, float]:
    """Extract key timings from the INFO-level JSON trace output."""
    metrics: Dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        busy = evt.get("time.busy")
        if not busy:
            continue
        span = evt.get("span") or {}
        name = span.get("name")
        target = evt.get("target")
        try:
            ms = parse_duration_ms(busy)
        except ValueError:
            continue

        if target == "chess_corners::multiscale":
            if name == "find_chess_corners":
                metrics["levels"] = span.get("levels")
                metrics["min_size"] = span.get("min_size")
                metrics["total_ms"] = ms
            elif name in {"coarse", "refine", "merge"}:
                metrics[f"{name}_ms"] = ms
        elif target == "chess_corners::pyramid" and name == "build_pyramid":
            metrics["pyramid_ms"] = ms
        elif (
            target == "chess_corners_core::descriptor"
            and name == "corners_to_descriptors"
        ):
            metrics["descriptor_ms"] = ms

    return metrics
