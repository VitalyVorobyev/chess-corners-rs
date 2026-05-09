# orientation_bench

Benchmark for the two-axis orientation fit inside `chess-corners`.
The bench renders synthetic patches and full chessboards, runs the
detector binding, and summarises per-cell accuracy, precision, sigma
calibration, and failure rates in a Markdown report.

## Prerequisites

```bash
pip install -r tools/orientation_bench/requirements.txt
maturin develop -m crates/chess-corners-py/pyproject.toml --release
```

The `--release` flag is recommended; debug builds are 30x slower for
the full grid. Tests that don't need the detector binding run without
maturin.

## Run

Smoke (≤30 s, useful for CI):

```bash
python -m orientation_bench sweep \
  --config tools/orientation_bench/configs/bench_smoke.yaml \
  --methods baseline disk_sector_rust disk_sector_py \
  --out out/orientation_bench
```

Full grid (≈10 min CPU on a laptop):

```bash
python -m orientation_bench sweep \
  --config tools/orientation_bench/configs/bench_default.yaml \
  --methods baseline sigma_correction_lut disk_sector_rust disk_sector_py adaptive_beta
```

Render the Markdown report:

```bash
python -m orientation_bench report --in out/orientation_bench/<timestamp>
```

## Outputs

```
out/orientation_bench/<timestamp>/
  metadata.json              # seed, config hash, git rev, methods, timestamp
  <sweep>/<method>/metrics.json
  plots/*.png                # CDF, error-vs-param, sigma calibration, etc.
  REPORT.md                  # rendered report (preserves manual sections)
```

The raw artefacts under `out/` are intentionally git-ignored. Only
`tools/orientation_bench/REPORT.md` is meant to be checked in (after
review).

## Determinism

- A single integer `seed` in the YAML config seeds every per-cell
  random number generator (`np.random.default_rng`).
- Per-cell sub-seeds are derived as `blake2s(seed | sweep_name |
  cell_idx)` so adding a new sweep does not perturb older cells.
- JSON output is written with sorted keys; floats are rounded to six
  significant figures before serialisation. Same seed + same config =
  bit-identical `metrics.json`.

## Methods

- `baseline`: legacy 16-ring Gauss-Newton fit.
- `sigma_correction_lut`: baseline angles with calibrated reported
  sigmas.
- `adaptive_beta`: failed diagnostic branch kept only for comparison.
- `disk_sector_py`: benchmark-only Python post-processor. It first
  runs `sigma_correction_lut` for centers/fallback sigmas, then
  conditionally replaces angles using a full-disk crossing-line score.
- `disk_sector_rust`: opt-in Rust port of the full-disk estimator. It
  uses the same conservative acceptance idea with sigma-LUT fallback.

## Adding a new method

`variants.py` maps method names to a builder function returning a
`chess_corners.ChessConfig`. Benchmark-only Python post-processors
should keep their extra logic in `runner.py` or a helper module under
`tools/orientation_bench`.

```python
def _build_my_variant():
    cfg = _build_baseline()
    cfg.orientation_method = chess_corners.OrientationMethod.AdaptiveBeta
    return cfg

VARIANTS["my_variant"] = _build_my_variant
```

Then run with `--methods baseline my_variant`. Phase 1 only ships the
baseline; non-registered names raise `KeyError`.

## Tests

```bash
pytest tools/orientation_bench/tests
```

The smoke test (`test_smoke.py`) skips automatically if
`chess_corners` is not importable.
