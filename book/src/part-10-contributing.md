# Part X: Contributing

The workspace is a practical reference implementation rather than a
one-off experiment. Contributions are welcome in three forms.

**Bug reports and feature requests.** Issues with reproduction steps
and sample data are the most useful. Small synthetic fixtures that
expose a failure mode are especially helpful because they can be
added directly to the test suite.

**Tests, benchmarks, and datasets.** The Python helpers under `tools/`
and the images in `testdata/` are designed for rerunning accuracy and
performance experiments after code changes. Extending them, or
contributing new datasets, is a good way to validate improvements.
The benchmark fixture in [Part VIII §7.1](part-08-benchmarks.md#71-the-benchmark-fixture)
and the ML refiner pipeline in `tools/ml_refiner/` have their own
knobs and regression gates.

**Algorithms.** ChESS and the Duda-Frese Radon detector are two
points in the design space. New detectors, refiners, or descriptor
variants can be added behind the same trait surfaces
(`CornerRefiner`, `DetectorConfig` presets, `DenseDetector` implementations) and benchmarked against the
shipped pipelines using `crates/chess-corners/examples/bench_sweep.rs`.
Proposals go in `docs/`, following the existing `proposal-*.md`
templates.

Pre-PR quality gates (also enforced in CI):

```bash
python3 tools/check_doc_versions.py
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
mdbook build book
```

See `CLAUDE.md` for the full set of workspace conventions, including
the layering rules between the core, facade, and pyramid crates.
