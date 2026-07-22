# ===========================================================================
# chess-corners — vcpkg overlay portfile
# ===========================================================================
# Packages the C/C++ bindings of the Rust `chess-corners` workspace (crate
# `chess-corners-capi`) behind a CMake package config and a pkg-config file,
# so C/C++ consumers can use:
#
#     find_package(chess-corners CONFIG REQUIRED)
#     target_link_libraries(app PRIVATE chess-corners::chess-corners)
#
# BUILD-TIME REQUIREMENT (the known Rust-via-vcpkg wrinkle): vcpkg does NOT
# provide cargo. The packaged CMake invokes
# `cargo build --release -p chess-corners-capi`, so a Rust toolchain
# (cargo / rustc) must be installed and on PATH at build time, with network
# access for crates.io. The `simd` feature additionally needs NIGHTLY Rust.
# ===========================================================================

# The packaged CMake always builds the Rust staticlib/cdylib in release mode
# (`cargo build --release`). A vcpkg "debug" tree would therefore hold a
# release library mislabeled as debug; build release-only to avoid that.
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO VitalyVorobyev/chess-corners-rs
    REF "v${VERSION}"
    # SHA512 of https://github.com/VitalyVorobyev/chess-corners-rs/archive/v1.1.0.tar.gz
    # Recompute whenever `version` in vcpkg.json changes.
    SHA512 b86d687459ae04a5f4266d07d7d5140e6cfcaa89dbf700e8c5ed199ab47e5c74e6d714541535387a3bb5bb8ab11231a922af6b36c1d81bd9fac8c6c551e92c20
    HEAD_REF main
)

# This initial port builds the crate with its default cargo features. Optional
# cargo features (rayon / simd / ml-refiner) are intentionally NOT exposed as
# vcpkg features yet: wiring them is a single release-phase task that adds
# pass-through features to the `chess-corners-capi` crate, teaches the packaged
# CMake to forward `--features` to cargo, and declares matching vcpkg features
# here — all verified together with a real `vcpkg install` (see ports/README.md).
set(_cc_configure_options "-DCHESS_CORNERS_BUILD_TESTS=OFF")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/crates/chess-corners-capi"
    OPTIONS
        ${_cc_configure_options}
    # Linkage: the packaged CMake reads VCPKG_LIBRARY_LINKAGE directly to pick
    # the static vs. dynamic artifact, so no extra option is needed here.
)

vcpkg_cmake_install()

# The packaged CMake installs its config to lib/cmake/chess-corners (under
# vcpkg, CMAKE_INSTALL_LIBDIR is always `lib`). Files installed there:
# chess-corners-config.cmake and chess-corners-config-version.cmake.
vcpkg_cmake_config_fixup(
    PACKAGE_NAME chess-corners
    CONFIG_PATH lib/cmake/chess-corners
)

# The packaged CMake installs chess-corners.pc into lib/pkgconfig.
vcpkg_fixup_pkgconfig()

# vcpkg layout: a single copy of the headers under <prefix>/include; never a
# debug/include duplicate. (A no-op while VCPKG_BUILD_TYPE is release-only,
# but kept per vcpkg convention.)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# macOS relocatability: the Rust cdylib is produced with an absolute
# build-tree install name; the packaged CMake rewrites it to `@rpath/<lib>`
# via `install_name_tool -id` during install, so the port ships a
# relocatable dylib with no extra work here.

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
