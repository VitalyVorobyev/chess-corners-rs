# Tools

This directory contains helper scripts used to benchmark and visualize the
`chess-corners` CLI. They are **not** required to use the Rust crates, but are
useful when you want to compare detectors or inspect performance.

Most scripts assume:

- Python 3.10+,
- NumPy,
- OpenCV (`opencv-python`),
- Matplotlib.

A minimal setup with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python matplotlib
```

From there:

- `accuracy_bench.py` – compares ChESS corners against OpenCV’s Harris and
  `findChessboardCornersSB` on stereo chessboard datasets, producing
  histograms and summary statistics.
- `perf_bench.py` – runs the CLI under different feature combinations
  (`rayon`, `simd`, etc.) and summarizes timing metrics from tracing output.
- `trace/` – helpers for running the CLI with JSON tracing enabled and
  parsing/plotting timing information.

Scripts expect the `chess-corners` CLI binary to be built first:

```bash
cargo build -p chess-corners --release
```

See the script docstrings and the `config/` JSON files for the exact CLI
arguments and expected directory layout under `testdata/`.

