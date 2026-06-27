# chess-corners vcpkg overlay port (DRAFT)

This directory is a local [vcpkg](https://vcpkg.io) **overlay port** for the
C/C++ bindings of `chess-corners` (the Rust ChESS chessboard-corner detector).
It packages the `crates/chess-corners-capi` CMake build so C/C++ projects can
consume the library via `find_package(chess-corners CONFIG)`.

> **Status: release-ready DRAFT.** It cannot be installed as-is. The `v1.0.0`
> tag does not exist yet and the `SHA512` in `portfile.cmake` is a `0`
> placeholder. See "Release-time TODOs" below. The port version tracks the
> planned `1.0.0` release; the workspace crate is currently `0.11.2`.

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

## Testing the overlay locally (once vcpkg + the tag exist)

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

Before the `v1.0.0` tag exists you can still exercise the portfile mechanics
against the branch tip — `--head` builds from `main` HEAD and skips the
`SHA512` check:

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

## Release-time TODOs

1. **Tag `v1.0.0`** on `VitalyVorobyev/chess-corners-rs`. The portfile's
   `REF` is `v${VERSION}`, and `version` in `vcpkg.json` is `1.0.0`.
2. **Real `SHA512`.** Run an install once with `SHA512 0`; vcpkg downloads the
   release tarball and prints the true hash. Paste it into `portfile.cmake`,
   replacing the `0`.
3. **Run a real `vcpkg install` on all three desktop OSes** — Linux, macOS,
   Windows — in **both linkages** (e.g. `x64-linux`, `x64-osx`,
   `x64-windows`, and `x64-windows-static`), with a Rust toolchain present.
   Confirm: `find_package(chess-corners CONFIG)` succeeds, linking
   `chess-corners::chess-corners` works, the macOS dylib loads via `@rpath`,
   and `pkg-config chess-corners` resolves. (Mobile/UWP/emscripten triplets
   are untested and would need cargo cross-compilation setup.)
4. **(Optional) Add feature support.** The initial port exposes no vcpkg
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
5. **Submit a registry PR** (overlay → official vcpkg registry) only after the
   above pass on all three platforms.

## Notes

- Linkage follows `VCPKG_LIBRARY_LINKAGE` (the packaged CMake reads it).
- The port builds **release-only** (`VCPKG_BUILD_TYPE release`) because the
  Rust library is always built with `cargo --release`; a vcpkg debug tree
  would otherwise carry a release library mislabeled as debug.
- On macOS the dylib's install name is rewritten to `@rpath/...` by the
  packaged CMake, so the installed library is relocatable.
- **No `vcpkg install` was run while authoring this port:** vcpkg is not
  installed on the authoring machine and the `v1.0.0` tag does not exist yet.
  Only the static checks below were performed.
```
