# Tools

Helper scripts for regenerating the book/README figures, benchmarking, and
research harnesses. None of them is required to use the Rust crates.

Most Python scripts assume Python 3.10+ with NumPy, Matplotlib and OpenCV
(`pip install -r tools/requirements.txt`); scripts that call the detector
directly need the Python bindings
(`(cd crates/chess-corners-py && maturin develop --release)`), and scripts
that drive the CLI need it built first (`cargo build -p chess-corners --release`).

| Path | Purpose |
|------|---------|
| `book/` | Part IX benchmark plots and fixture panels (`opencv_subpix_sweep.py`, `synth_examples.py`, `plot_benchmark.py`); see the book's §9.12. |
| `render_book_overlays.py` (+ `plot_output.py`) | `book/src/img/{small,mid,large}_chess.png` detector overlays. |
| `render_orientation_overlays.py` | Part VI orientation figures. |
| `render_radon_saddle_overlay.py` | Part IV Radon figure. |
| `render_uncertainty_overlays.py` | Part VIII uncertainty figures. |
| `render_readme_perf.py`, `render_readme_warp_overlays.py` | README timing table and warp overlay. |
| `synthimages.py` | Synthetic chessboard dataset with ground truth (below). |
| `detect.py` | Single-image overlay of the CLI's corners, optionally next to OpenCV Harris / `findChessboardCornersSB` (below). |
| `perf/` | Criterion bench-regression gate and committed baseline; see `perf/README.md`. |
| `profile.sh` | `cargo-flamegraph` / `samply` wrapper for hot-path profiling. |
| `check_doc_versions.py` | Checks install-snippet versions in the docs against the workspace version (CI; `--fix` rewrites). |
| `orientation_bench/` | Orientation-fit benchmark; see its README. |
| `refiner_robustness/` | Refiner-robustness and uncertainty research harnesses; results in `docs/reference/`. |
| `orientation_sigma/` | Orientation-σ study: `synth.py` grid and real-data truth for the `orientation_sigma_study` example, and the z / Spearman analysis; results in `docs/reference/orientation-sigma.md`. |
| `detection/`, `plotting/` | Internal modules used by `detect.py` and `plot_output.py`. |

## Synthetic dataset generation

`synthimages.py` writes `dataset.json` (metadata + per-image ground truth
`gt_corners_uv`) and rendered grayscale `images/000000.png`, …:

```bash
python3 tools/synthimages.py \
  --out testdata/synthetic_sigma_3 \
  --num 200 --seed 1 \
  --inner 9 6 --square_size 0.03 --pps 90 \
  --img_size 1280 960 --fx 1100 --fy 1100 \
  --yaw 35 --pitch 30 --roll 15 \
  --z_range 0.7 1.6 \
  --noise_sigma 3.0 --blur_sigma 0.8 \
  --gamma_range 0.9 1.2 --contrast_range 0.9 1.1 --brightness_range -10 10 \
  --vignetting 0.15
```

## Single-image visualization

```bash
python3 tools/detect.py testdata/synthetic_sigma_3/images/000000.png \
  --config config/chess_cli_config_example.json \
  --harris --chessboard --pattern-size 9 6 \
  --out /tmp/overlay.png
```

For details, check each script's docstring and `--help`.
