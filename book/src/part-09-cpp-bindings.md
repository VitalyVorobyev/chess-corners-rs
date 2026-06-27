# Part IX: C++ bindings

The detector core is written in Rust, but most calibration and
robotics codebases that consume chessboard corners are C or C++. This
chapter shows how to call the detector from both languages, what to
install, and the ownership rules that keep the language boundary safe.

## 9.1 Why a C ABI with a thin C++ header

C++ has no stable ABI across compilers, standard-library versions, or
even build flags, so a Rust library cannot hand C++ types across the
link boundary directly. The portable lowest common denominator is a
**C ABI**: flat structs, integer status codes, and explicit ownership.
Every language and every native package manager already knows how to
link against C. Exposing the detector through a C header
(`chess_corners.h`) therefore reaches the widest set of consumers and
keeps the Rust side self-contained — the boundary stays small, stable,
and easy to audit.

A raw C ABI is, by design, spartan: you pass pointers, check an
integer status, and free whatever the library allocated. To recover
C++ ergonomics without giving up that portable core, the package also
ships a header-only C++17 wrapper (`chess_corners.hpp`). It adds
`std::vector` returns, value-type corners, exceptions instead of
status codes, and RAII cleanup — and it carries no detection logic of
its own. Every call marshals straight to the C ABI, so the C++ layer
cannot drift from the C one.

In short:

- **Link from anything** — the C ABI is the universal contract.
- **Write idiomatic C++** — the header adds vectors, exceptions, RAII.
- **Pure Rust build** — no C++ toolchain is required to build the
  library itself.

## 9.2 Installing

The library builds to a normal native library (shared or static) with
a CMake package config and a pkg-config file, so you integrate it the
same way you integrate any C/C++ dependency.

**CMake (`find_package`).** Point `CMAKE_PREFIX_PATH` at the install
prefix and link the imported target:

```cmake
find_package(chess-corners CONFIG REQUIRED)

add_executable(detect detect.cpp)
target_compile_features(detect PRIVATE cxx_std_17)   # the C++ header is C++17
target_link_libraries(detect PRIVATE chess-corners::chess-corners)
```

The imported target carries its own include directories, so both
`chess_corners.h` and `chess_corners.hpp` land on the include path as
soon as you link it.

**pkg-config.** For non-CMake build systems, a `chess-corners`
pkg-config module is installed alongside the library:

```sh
cc detect.c $(pkg-config --cflags --libs chess-corners) -o detect
```

**vcpkg.** An overlay port under `ports/` integrates the library into
vcpkg-based projects; it becomes a fully supported port at the 1.0
release. Until then, consume it as an overlay (`--overlay-ports`) from
a vcpkg manifest.

## 9.3 Using the library from C++

Include `chess_corners.hpp`, pick a `Config` preset, and call
`detect`. The image is 8-bit, row-major grayscale — exactly
`width * height` bytes.

```cpp
#include "chess_corners.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    std::uint32_t width = /* image width  */;
    std::uint32_t height = /* image height */;
    std::vector<std::uint8_t> pixels = /* width * height grayscale bytes */;

    try {
        // Presets: chess(), chess_multiscale(), radon(),
        // radon_multiscale(), default_(). The flat fields are public,
        // so you can tweak a preset before detecting:
        chess_corners::Config config = chess_corners::Config::chess();
        // config.orientation_method = CC_ORIENTATION_DISK_FIT;  // optional

        std::vector<chess_corners::Corner> corners =
            chess_corners::detect(pixels, width, height, config);

        for (const chess_corners::Corner& c : corners) {
            std::printf(
                "(%.2f, %.2f) response=%.3f  "
                "axis0=%.3f rad (sigma %.3f)  axis1=%.3f rad (sigma %.3f)\n",
                c.x, c.y, c.response,
                c.axes[0].angle, c.axes[0].sigma,
                c.axes[1].angle, c.axes[1].sigma);
        }
    } catch (const chess_corners::Error& err) {
        std::fprintf(stderr, "detection failed (status %d): %s\n",
                     static_cast<int>(err.status()), err.what());
        return 1;
    }
    return 0;
}
```

A few things the wrapper does for you:

- **RAII, no manual free.** `detect` returns a `std::vector<Corner>`
  of value-type corners. The heap buffer the C ABI allocated is
  released by an internal guard before `detect` returns — even if
  marshalling throws partway — so there is nothing for you to free.
- **Exceptions instead of status codes.** A non-success status from
  the C ABI becomes a `chess_corners::Error`, whose `what()` is the
  library's own description and whose `status()` returns the
  underlying `cc_status`. Catch `chess_corners::Error` for detection
  failures, or the broader `std::runtime_error` to also catch an ABI
  mismatch (see §9.5).
- **Two `detect` overloads.** The `std::vector<std::uint8_t>` overload
  shown here validates the buffer length for you; a
  `const std::uint8_t*` overload takes a raw pointer plus dimensions
  when you already own the buffer.

## 9.4 Using the library from plain C

The C path is the same pipeline without the conveniences: you build a
`cc_config`, call `cc_detect_u8`, read the result, and free it.

```c
#include "chess_corners.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    uint32_t width = /* image width  */;
    uint32_t height = /* image height */;
    const uint8_t *pixels = /* width * height grayscale bytes */;

    cc_config cfg = cc_config_chess();   /* or cc_config_radon(), ... */

    cc_result result;
    cc_status status = cc_detect_u8(pixels, width, height, &cfg, &result);
    if (status != CC_OK) {
        /* On any error status, *out is left untouched: nothing to free. */
        fprintf(stderr, "detection failed: %s\n", cc_status_str(status));
        return 1;
    }

    for (size_t i = 0; i < result.len; ++i) {
        const cc_corner *c = &result.corners[i];
        printf("(%.2f, %.2f) response=%.3f\n", c->x, c->y, c->response);
    }

    /* The library owns result.corners. Release it exactly once. */
    cc_result_free(&result);
    return 0;
}
```

The ownership contract is the part to get right:

- On `CC_OK`, `result` owns a heap-allocated corner array. You must
  release it with `cc_result_free` exactly once. `cc_result_free` is
  null-safe and idempotent — after it runs, `corners` is null and
  `len` is zero.
- On any error status, `*out` is left untouched, so there is nothing
  to free; handle the error and return.
- `cc_status_str` returns a static, NUL-terminated description that
  you must **not** free.

## 9.5 Notes

**Reentrancy and threading.** `cc_detect_u8` is stateless and
reentrant: it keeps no global state and no hidden caches between calls,
and each call allocates and returns its own result. You can run
detections concurrently from multiple threads on independent inputs
without any external locking.

**ABI-version guard.** The library reports its ABI revision through
`cc_abi_version()`, and the C++ header pins the revision it was
written against (`chess_corners::CHESS_CORNERS_ABI_VERSION`).
`chess_corners::detect` calls `check_abi()` before every detection and
throws `std::runtime_error` on a mismatch, so a header that no longer
matches the linked library fails loudly instead of reading a stale
struct layout. You can also call `chess_corners::check_abi()` once at
startup. In plain C, compare `cc_abi_version()` against the value you
built against and refuse to proceed if they differ.

**Corner fields.** `cc_corner` (and the C++ `Corner`) mirror the Rust
`CornerDescriptor`: a subpixel `x`, `y` in full-resolution input
pixels, a raw detector `response`, and two local grid `axes`, each
carrying an `angle` and its 1σ angular uncertainty `sigma`, both in
radians. The polarity convention is identical on every binding:
`axes[0].angle` lies in `[0, π)` and `axes[1].angle` in
`(axes[0].angle, axes[0].angle + π)`, with the counter-clockwise arc
between them crossing a dark sector. See
[Part I §1.4](part-01-orientation.md#14-the-cornerdescriptor-output)
for the field semantics and
[Part III §3.4](part-03-chess-detector.md#34-corner-descriptors) for
the fit math.

---

Next: [Part X: Contributing](part-10-contributing.md) — how to file
issues, add tests and datasets, and propose new algorithms.
