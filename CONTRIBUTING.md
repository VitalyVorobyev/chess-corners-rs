# Contributing

Thanks for your interest in improving `chess-corners-rs`!

## Development workflow

This repository is a small Rust workspace:

- `crates/chess-corners-core` – low-level ChESS response + detector.
- `crates/chess-corners` – ergonomic facade with optional `image` support and a CLI.

Useful commands:

- Run tests on stable defaults:

  ```bash
  cargo test --workspace
  ```

- Run clippy and fmt (mirrors CI):

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

- Exercise feature combinations (similar to CI):

  ```bash
  # core
  cargo test -p chess-corners-core --tests
  cargo test -p chess-corners-core --tests --features rayon

  # facade (no simd on stable)
  cargo test -p chess-corners --lib
  cargo test -p chess-corners --lib --features rayon
  ```

SIMD (`simd` feature) is tested on nightly in CI; if you have a nightly toolchain installed you can run:

```bash
cargo test -p chess-corners-core --tests --features "simd"
cargo test -p chess-corners-core --tests --features "rayon,simd"
```

## Docs and book

To build the Rust API docs and the mdBook locally:

```bash
cargo doc --workspace --all-features --no-deps
mdbook build book
```

The GitHub Pages site published at
`https://vitalyvorobyev.github.io/chess-corners-rs/` is built from these
artifacts in CI.

## Submitting changes

- Prefer small, focused pull requests.
- Include tests when fixing bugs or adding user-visible behavior.
- Keep the public API surface stable; if you believe a breaking change is
  needed, please open an issue first to discuss the design.

If you are unsure about anything, opening a draft PR or a discussion issue is
very welcome.
