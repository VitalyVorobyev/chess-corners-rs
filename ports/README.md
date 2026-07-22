# chess-corners vcpkg overlay port

This directory is a local [vcpkg](https://vcpkg.io) **overlay port** for the
C/C++ bindings of `chess-corners` (the Rust ChESS chessboard-corner detector).
It packages the `crates/chess-corners-capi` CMake build so C/C++ projects can
consume the library via `find_package(chess-corners CONFIG)`.

> **Status: installable, not yet install-verified.** The port targets `1.1.0`,
> the `v1.1.0` tag exists, and `portfile.cmake` carries the real `SHA512` of
> its source tarball — so `vcpkg install` has everything it needs. It has
> still never been exercised by an actual `vcpkg install` on any platform;
> that verification is TODO 1 below and is a prerequisite for a registry
> submission.

## Layout

```
ports/
  README.md                    (this file)
  chess-corners/
    vcpkg.json                 (manifest: name, version, license, host deps)
    portfile.cmake             (fetch + configure + install steps)
```

## Build-time requirement: a Rust toolchain

vcpkg does **not** provide cargo. The packaged CMake runs
`cargo build --release -p chess-corners-capi`, so the machine building this
port must have a Rust toolchain (`cargo` / `rustc`) on `PATH`, plus network
access for crates.io. This is the known wrinkle of distributing a Rust library
through vcpkg, and it is the main risk for eventual registry acceptance (vcpkg
CI prefers hermetic, network-free builds). The optional `simd` cargo feature
(a future port feature — see TODO 4) would additionally require a **nightly**
toolchain.

## Testing the overlay locally

From any directory, with vcpkg available and a Rust toolchain on `PATH`:

```sh
# the initial port builds with the crate's default cargo features
vcpkg install chess-corners --overlay-ports=/abs/path/to/ports

# linkage is selected by the triplet
vcpkg install chess-corners:x64-linux           # static (default)
vcpkg install chess-corners:x64-osx             # static (default)
vcpkg install chess-corners:x64-windows         # dynamic (default)
vcpkg install chess-corners:x64-windows-static  # static
```

To exercise the portfile against the branch tip instead of the release tag,
`--head` builds from `main` HEAD and skips the `SHA512` check:

```sh
vcpkg install chess-corners --head --overlay-ports=/abs/path/to/ports
```

A consuming CMake project then uses:

```cmake
find_package(chess-corners CONFIG REQUIRED)
target_link_libraries(app PRIVATE chess-corners::chess-corners)
```

Non-CMake consumers can use the installed pkg-config file:
`pkg-config --cflags --libs chess-corners`.

## Keeping the port current

On every release, bump `version` in `vcpkg.json` and recompute the `SHA512` in
`portfile.cmake` — the portfile's `REF` is `v${VERSION}`, so both must move
together:

```sh
curl -sSL -o src.tar.gz \
  "https://github.com/VitalyVorobyev/chess-corners-rs/archive/v<VERSION>.tar.gz"
shasum -a 512 src.tar.gz
```

## Remaining TODOs

1. **Run a real `vcpkg install` on all three desktop OSes** — Linux, macOS,
   Windows — in **both linkages** (e.g. `x64-linux`, `x64-osx`,
   `x64-windows`, and `x64-windows-static`), with a Rust toolchain present.
   Confirm: `find_package(chess-corners CONFIG)` succeeds, linking
   `chess-corners::chess-corners` works, the macOS dylib loads via `@rpath`,
   and `pkg-config chess-corners` resolves. (Mobile/UWP/emscripten triplets
   are untested and would need cargo cross-compilation setup.)
2. **(Optional) Add feature support.** The initial port exposes no vcpkg
   features — it builds the crate's default cargo features. To offer
   `rayon` / `simd` / `ml-refiner` as vcpkg features, make three changes
   together and verify each with a real install:
   - add pass-through cargo features (`rayon` / `simd` / `ml-refiner`) to the
     `chess-corners-capi` crate, forwarding to the `chess-corners` facade;
   - teach `crates/chess-corners-capi/CMakeLists.txt` to read a
     `CHESS_CORNERS_CARGO_FEATURES` cache var and append `--features ...` to
     the cargo build; and
   - declare matching `features` in `vcpkg.json` and map them in
     `portfile.cmake`.

   Verify each feature actually changes the produced library (symbol/behavior
   diff; note `simd` requires a nightly toolchain).
3. **Submit a registry PR** (overlay → official vcpkg registry) only after the
   above pass on all three platforms.

## Notes

- Linkage follows `VCPKG_LIBRARY_LINKAGE` (the packaged CMake reads it).
- The port builds **release-only** (`VCPKG_BUILD_TYPE release`) because the
  Rust library is always built with `cargo --release`; a vcpkg debug tree
  would otherwise carry a release library mislabeled as debug.
- On macOS the dylib's install name is rewritten to `@rpath/...` by the
  packaged CMake, so the installed library is relocatable.
- **No `vcpkg install` has ever been run against this port.** vcpkg is not
  installed on the authoring machine, so only static checks have been
  performed. The `SHA512` was computed directly from the release tarball
  rather than read back from a vcpkg download.
