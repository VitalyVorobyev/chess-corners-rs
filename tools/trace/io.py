from .runner import RunResult, FeatureCombo
from typing import Iterable

def combo_label(combo: FeatureCombo) -> str:
    return "none" if not combo else "+".join(combo)

def print_results(results: Iterable[RunResult], runs: int) -> None:
    rows = []
    for r in results:
        m = r.metrics
        rows.append(
            (
                r.config_label,
                combo_label(r.combo),
                r.image,
                m.get("levels"),
                m.get("total_ms"),
                m.get("pyramid_ms"),
                m.get("coarse_ms"),
                m.get("refine_ms"),
                m.get("merge_ms"),
                m.get("descriptor_ms"),
            )
        )

    print(f"\nPer-image timings averaged over {runs} run(s) (ms):")
    header = (
        "config",
        "features",
        "image",
        "levels",
        "total",
        "pyramid",
        "coarse",
        "refine",
        "merge",
        "descriptor",
    )
    print(
        "{:<10} {:<18} {:<30} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
            *header
        )
    )
    for row in rows:
        config, combo, img, lvl, tot, pyr, coarse, refine, merge, desc = row
        print(
            "{:<10} {:<18} {:<30} {:>6} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.3f}".format(
                config,
                combo,
                img.name,
                lvl if lvl is not None else "-",
                tot or 0.0,
                pyr or 0.0,
                coarse or 0.0,
                refine or 0.0,
                merge or 0.0,
                desc or 0.0,
            )
        )
