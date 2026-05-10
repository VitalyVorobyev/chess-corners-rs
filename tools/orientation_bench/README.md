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
  --methods ring_fit disk_fit \
  --out out/orientation_bench
```

Full grid (≈10 min CPU on a laptop):

```bash
python -m orientation_bench sweep \
  --config tools/orientation_bench/configs/bench_default.yaml \
  --methods ring_fit disk_fit
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

- `ring_fit`: 16-sample ring Gauss-Newton fit with calibrated σ
  (the workspace default).
- `disk_fit`: full-disk crossing-line estimator with `ring_fit`
  fallback when the disk model is weak or invalid.
- `disk_sector_py`: benchmark-only Python post-processor. It first
  runs `ring_fit` for centers/fallback sigmas, then conditionally
  replaces angles using a full-disk crossing-line score.

## Adding a new method

`variants.py` maps method names to a builder function returning a
`chess_corners.ChessConfig`. Benchmark-only Python post-processors
should keep their extra logic in `runner.py` or a helper module under
`tools/orientation_bench`.

```python
def _build_my_variant():
    cfg = _build_permissive_cfg()
    cfg.orientation_method = chess_corners.OrientationMethod.RING_FIT
    return cfg

VARIANTS["my_variant"] = _build_my_variant
```

Then run with `--methods ring_fit my_variant`. Non-registered names
raise `KeyError`.

## Tests

```bash
pytest tools/orientation_bench/tests
```

The smoke test (`test_smoke.py`) skips automatically if
`chess_corners` is not importable.
