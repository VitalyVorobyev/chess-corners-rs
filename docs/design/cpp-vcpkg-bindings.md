# Design: C++ bindings via vcpkg

**Status:** Implemented (`CPP-01..07`); the vcpkg overlay port is verified
locally, with registry finalization (real `v1.0.0` tag + SHA512 + cross-platform
`vcpkg install`) deferred to the 1.0 release. **Workstream:** `CPP-*` (ROADMAP
milestone **M5**, depends on **M3** API freeze). **Approach:** a C ABI generated
with `cbindgen`, wrapped by a thin hand-written C++ header, packaged with CMake
and shipped as a vcpkg port.

## Why C-ABI + `cbindgen` (not `cxx`)

| | C-ABI + cbindgen (chosen) | `cxx` bridge |
|--|---------------------------|--------------|
| ABI surface | stable `extern "C"`, FFI-able from C, C++, any language | C++-specific, tied to the `cxx` runtime |
| vcpkg packaging | clean: lib + header + CMake config | harder: must ship `cxx`-generated glue + runtime |
| C++ ergonomics | provided by our own thin header (RAII, `std::vector`) | richer out of the box |
| Build coupling | Rust stays pure; no C++ compiled inside the Rust build | C++ compiled during the Rust build |

A C ABI is the portable lowest common denominator and the standard shape for a
vcpkg library; C++ ergonomics come from a small hand-written header on top.

## Crate: `chess-corners-capi`

`crates/chess-corners-capi` (`crate-type = ["staticlib", "cdylib", "rlib"]`,
`publish = false`) is a thin `extern "C"` layer over the `chess-corners` facade.
It owns **no** algorithm logic — it only marshals types across the boundary, and
every boundary function is panic-trapped.

### C type mapping (frozen surface)

| Rust | C |
|------|---|
| `CornerDescriptor { x, y, response, axes: Option<[AxisEstimate;2]> }` | `cc_corner { float x, y, response; cc_axis axes[2]; uint8_t has_orientation; }` |
| `AxisEstimate { angle, sigma }` | `cc_axis { float angle, sigma; }` |
| `DetectorConfig` (+ nested enums) | `cc_config` flat struct of scalars + `uint32_t` enum tags (no discriminant UB) |
| `ChessError` | `cc_status` enum (`CC_OK=0`, `CC_ERR_*`) |

`axes` is optional in Rust; the C struct carries a `has_orientation` flag that is
`0` (with `axes` zeroed) when the fit is skipped. The mapping tracks the frozen
`CornerDescriptor` (no `contrast`/`fit_rms` — see [`api-v1.0.md`](api-v1.0.md)),
which is why M5 depends on M3.

### C function surface

```c
cc_config   cc_config_default(void);
cc_config   cc_config_chess_multiscale(void);        // mirror Rust presets
cc_status   cc_detect_u8(const uint8_t* pixels, uint32_t w, uint32_t h,
                         const cc_config* cfg, cc_result* out);
void        cc_result_free(cc_result*);              // frees the lib-owned array
const char* cc_status_str(cc_status);
uint32_t    cc_abi_version(void);                    // header/lib ABI guard
```

`cc_result { cc_corner* corners; size_t len; }`; the lib allocates, the caller
frees. No callbacks, no global state — one detect call is self-contained and
reentrant. (A reusable detector handle for cross-frame buffer reuse is a scoped
post-1.0 follow-up.)

## Packaging & CI (as shipped)

- **Header:** `cbindgen.toml` + a `generate-ffi-header` bin (with `--check` drift
  mode) emit the committed `include/chess_corners.h`; CI fails on drift.
- **C++ convenience header:** `include/chess_corners.hpp` (C++17) — value types +
  presets, exception-safe `ResultGuard` RAII, `detect()` → `std::vector<Corner>`
  throwing `chess_corners::Error`, `check_abi()` per call. Clean under
  `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.
- **CMake:** `find_package(chess-corners CONFIG)` → `chess-corners::chess-corners`;
  honours `BUILD_SHARED_LIBS`/`VCPKG_LIBRARY_LINKAGE`, links `native-static-libs`,
  fixes the macOS dylib install-name, emits a relocatable pkg-config `.pc`.
- **vcpkg port:** overlay port `ports/chess-corners/` (`vcpkg.json` +
  `portfile.cmake`) — `vcpkg_from_github` → configure/install/config_fixup, honours
  `VCPKG_LIBRARY_LINKAGE`. Develop against the overlay first; submit to the
  registry only after the release tag exists.
- **CI (`.github/workflows/cpp.yml`):** static+shared matrix — header-drift gate →
  build → ctest → install → build an example via `find_package`. Marshalling
  parity (`cc_detect_u8` corner-by-corner vs `Detector::detect_u8`) is a Rust test.

## Resolved / deferred

1. **`cc_config` versioning.** Runtime-guarded by `cc_abi_version()` (the C++
   header checks it every call). The flat struct is frozen for 1.0;
   field-evolution policy (reserved padding vs versioned structs) is a post-1.0
   ABI concern.
2. **Threading.** `cc_detect_u8` is reentrant and stateless; `rayon` parallelism
   is internal, controlled by the build feature.
3. **Buffer reuse.** v1 ships stateless; a reusable detector handle is a scoped
   post-1.0 follow-up.
4. **Minimum versions.** CMake ≥ 3.20 and a C++17 compiler.

---

Book coverage: Part IX "C++ bindings" (`CPP-07`). Task list:
[`../BACKLOG.md`](../BACKLOG.md) `CPP-*`.
