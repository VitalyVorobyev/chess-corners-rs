# chess-corners-capi

Stable C ABI for the [`chess-corners`](../chess-corners) chessboard-corner
detector, packaged as a normal native library (shared or static) with a
CMake package config and a pkg-config file. A thin, header-only C++17
wrapper (`chess_corners.hpp`) sits on top of the flat C header
(`chess_corners.h`) for idiomatic `std::vector` / exception-based usage;
every call marshals straight through to the C ABI, so the two layers
cannot drift.

This crate is not published to crates.io — it ships to C/C++ consumers
via CMake / vcpkg / pkg-config, not `cargo add`.

## Install

**CMake (`find_package`):**

```cmake
find_package(chess-corners CONFIG REQUIRED)

add_executable(detect detect.cpp)
target_compile_features(detect PRIVATE cxx_std_17)
target_link_libraries(detect PRIVATE chess-corners::chess-corners)
```

**vcpkg:** an overlay port under [`ports/`](../../ports) integrates the
library into vcpkg-based projects (`--overlay-ports`).

**pkg-config:** for non-CMake build systems, a `chess-corners` module is
installed alongside the library:

```sh
cc detect.c $(pkg-config --cflags --libs chess-corners) -o detect
```

## C++ example

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
        std::vector<chess_corners::Corner> corners =
            chess_corners::detect(pixels, width, height, chess_corners::Config::chess());
        std::printf("detected %zu corners\n", corners.size());
    } catch (const chess_corners::Error& err) {
        std::fprintf(stderr, "detection failed (status %d): %s\n",
                     static_cast<int>(err.status()), err.what());
        return 1;
    }
    return 0;
}
```

A complete, buildable version of this example (self-contained, no
image-decode dependency) lives at
[`examples/cpp/detect.cpp`](examples/cpp/detect.cpp).

See [book Part IX: C++ bindings](../../book/src/part-09-cpp-bindings.md)
for the full guide — ownership rules, presets, error handling, and the
ABI-version safety check.

## License

MIT
