# Design: C++ bindings via vcpkg

**Status:** Implemented (CPP-01..07); the vcpkg overlay port is verified
locally, with registry finalization (real `v1.0.0` tag + SHA512 + cross-platform
`vcpkg install`) deferred to the 1.0 release. **Workstream:** `CPP-*` (ROADMAP
milestone **M5**, depends on **M3** API freeze). **Approach (locked):** C ABI generated with
`cbindgen`, wrapped by a thin hand-written C++ convenience header, packaged
with CMake and shipped as a vcpkg port.

## Why C-ABI + `cbindgen` (not `cxx`)

| | C-ABI + cbindgen (chosen) | `cxx` bridge |
|--|---------------------------|--------------|
| ABI surface | stable `extern "C"`, trivially FFI-able from C, C++, and any other language | C++-specific, tied to the `cxx` runtime |
| vcpkg packaging | clean: a static/dynamic lib + a header + CMake config | harder: must ship `cxx`-generated glue + runtime |
| C++ ergonomics | provided by our own thin header (RAII, `std::vector`) | richer out of the box |
| Build coupling | Rust stays pure; no C++ compiled inside the Rust build | C++ compiled during the Rust build |

A C ABI is the portable lowest common denominator and the standard shape for
a vcpkg library. We keep C++ ergonomics by hand-writing a small header on top.

## Crate: `chess-corners-capi`

New crate `crates/chess-corners-capi`, `crate-type = ["staticlib", "cdylib"]`,
a thin `extern "C"` layer over the `chess-corners` facade. It owns **no**
algorithm logic — it only marshals types across the boundary.

### C type mapping (post-API-freeze surface)

| Rust | C |
|------|---|
| `CornerDescriptor { x, y, response, axes: Option<[AxisEstimate;2]> }` | `cc_corner { float x, y, response; cc_axis axes[2]; uint8_t has_orientation; }` |
| `AxisEstimate { angle, sigma }` | `cc_axis { float angle, sigma; }` |
| `DetectorConfig.threshold: f32` | `cc_config.threshold` — a single `float`, no kind tag |
| `DetectorConfig` (+ nested enums) | `cc_config` flat struct of scalars + `int` enum tags |
| `ChessError` | `cc_status` enum (`CC_OK=0`, `CC_ERR_*`) |

> The mapping must track the **frozen** `CornerDescriptor` (no `contrast` /
> `fit_rms` — see [`api-v1.0.md`](api-v1.0.md)). `axes` is optional in Rust
> (`Option<[AxisEstimate;2]>`); the C struct carries a `has_orientation`
> flag that is `0` (with `axes` zeroed) when the per-corner fit is skipped.
> This is why M5 depends on M3.

### C function surface (sketch)

```c
cc_config   cc_config_default(void);                 // sane defaults
cc_config   cc_config_chess_multiscale(void);        // mirror Rust presets
cc_status   cc_detect_u8(const uint8_t* pixels, uint32_t w, uint32_t h,
                         const cc_config* cfg, cc_result* out);
void        cc_result_free(cc_result*);              // frees the corner array
const char* cc_status_str(cc_status);                // human-readable error
uint32_t    cc_abi_version(void);                    // header/lib ABI guard
```

`cc_result { cc_corner* corners; size_t len; }`. Ownership: the lib allocates
`corners`; the caller must call `cc_result_free`. No callbacks, no global
state; one detect call is self-contained. (A later iteration may add a reusable
detector handle for buffer reuse across frames — out of scope for first ship.)

### Header generation

`cbindgen.toml` → `include/chess_corners.h` (committed and regenerated in CI;
CI fails if the committed header drifts from the source). The hand-written
`include/chess_corners.hpp` wraps the C API:

```cpp
namespace chess_corners {
  std::vector<Corner> detect(std::span<const uint8_t>, uint32_t w, uint32_t h,
                             const Config& = {});   // throws on cc_status != OK
}
```

RAII for `cc_result` (free in destructor), `std::vector<Corner>` return,
exceptions (or `std::expected`) for errors.

## Packaging

### CMake

`CMakeLists.txt` builds nothing Rust itself; instead the port invokes `cargo
build --release -p chess-corners-capi` and installs the artifact + headers,
then generates a CMake **package config** so consumers do:

```cmake
find_package(chess-corners CONFIG REQUIRED)
target_link_libraries(app PRIVATE chess-corners::chess-corners)
```

Also emit a `pkg-config` `.pc` for non-CMake consumers.

### vcpkg port

- `vcpkg.json` manifest (name `chess-corners`, version tracking the crate).
- `portfile.cmake`: fetch the source at the released git tag, run the Rust
  build via vcpkg's cargo support / a custom build step, install lib + headers
  + CMake config; respect `VCPKG_LIBRARY_LINKAGE` (static vs dynamic).
- **Develop against an overlay port first** (`--overlay-ports`) for local
  iteration; submit to the vcpkg registry only once stable.
- Feature flags exposed as vcpkg port features (e.g. `rayon`, `simd`,
  `ml-refiner`) mapping to cargo features.

## CI

A dedicated job (Linux first, then macOS/Windows):
1. `cargo build -p chess-corners-capi --release`.
2. Regenerate the header and assert no diff vs the committed one.
3. Compile a C smoke test and a C++ example against the built lib.
4. Run the smoke test: detect on a fixture image, assert corner count/positions
   match the Rust reference within tolerance (parity test).
5. (Later) `vcpkg install chess-corners --overlay-ports=...` from the overlay.

## Testing

- **Parity test:** C harness runs `cc_detect_u8` on a `testimages/` fixture
  and compares against a golden produced by the Rust CLI — guards the
  marshalling.
- **ABI guard:** `cc_abi_version()` checked by the C++ header at compile/run
  time so a mismatched header/lib pair fails loudly.

## Resolved / deferred

1. **`cc_config` versioning.** Guarded at runtime by `cc_abi_version()` — the
   C++ header checks it on every call so a mismatched header/lib pair fails
   loudly. The flat struct is frozen for 1.0; field-evolution policy (reserved
   padding vs versioned structs) is a post-1.0 ABI-change concern.
2. **Threading.** `cc_detect_u8` is reentrant and carries no global state;
   `rayon` parallelism is internal and controlled by the build feature
   (documented in the header and book Part IX).
3. **Buffer reuse.** v1 ships stateless (one self-contained detect call); a
   reusable detector handle is a scoped post-1.0 follow-up.
4. **Minimum versions.** CMake ≥ 3.20 and a C++17 compiler (the convenience
   header is C++17; the example and smoke test pin `cxx_std_17`).

---

Book coverage: Part IX "C++ bindings" (CPP-07). Task list:
[`../BACKLOG.md`](../BACKLOG.md) `CPP-*`.
