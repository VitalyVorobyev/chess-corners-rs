// Self-contained C++ consumer of the chess-corners library.
//
// Synthesizes a checkerboard in memory (no image-decode dependency), runs the
// detector through the chess_corners convenience header, and prints how many
// corners were found. Doubles as a marshalling smoke test: a clean board with
// 8 squares per side has 7 x 7 = 49 interior corners.

#include "chess_corners.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

int main() {
    constexpr std::uint32_t kSquarePx = 16;
    constexpr std::uint32_t kSquares = 8;
    constexpr std::uint32_t width = kSquarePx * kSquares;   // 128
    constexpr std::uint32_t height = kSquarePx * kSquares;  // 128

    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) * height, 0u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            if (((x / kSquarePx) + (y / kSquarePx)) % 2u == 0u) {
                image[static_cast<std::size_t>(y) * width + x] = 255u;
            }
        }
    }

    try {
        const std::vector<chess_corners::Corner> corners =
            chess_corners::detect(image, width, height, chess_corners::Config::chess());
        std::printf("chess-corners: detected %zu corners\n", corners.size());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "detection failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
