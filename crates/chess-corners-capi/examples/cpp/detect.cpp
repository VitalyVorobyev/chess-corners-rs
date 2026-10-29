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
#include <optional>
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

        // Every localizer reports each corner's position covariance (px^2)
        // under `Uncertainty::propagated` (the default is off). This
        // synthetic board has no noise to estimate, so supply the sensor
        // noise level.
        chess_corners::Config saddle = chess_corners::Config::chess();
        saddle.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
        saddle.uncertainty = chess_corners::Uncertainty::propagated(1.0f);
        std::size_t with_cov = 0;
        std::optional<float> noise_sigma;
        for (const chess_corners::Corner& c :
             chess_corners::detect(image, width, height, saddle, &noise_sigma)) {
            if (c.position_covariance) {
                ++with_cov;
            }
        }
        std::printf("chess-corners: %zu corners with a position covariance\n", with_cov);
        // The noise level behind the covariances: here the supplied one.
        if (with_cov == 0 || noise_sigma != saddle.uncertainty.noise_sigma()) {
            return 1;
        }

        // Radon keeps its peak fit by default; the same image refiner
        // refines its peaks on request. Switching the uncertainty off drops
        // the covariances and leaves positions unchanged.
        chess_corners::Config radon = chess_corners::Config::radon();
        radon.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
        radon.uncertainty = chess_corners::Uncertainty::propagated(1.0f);
        const std::vector<chess_corners::Corner> radon_corners =
            chess_corners::detect(image, width, height, radon);
        std::printf("chess-corners: Radon detected %zu corners\n", radon_corners.size());

        chess_corners::Config quiet = saddle;
        quiet.uncertainty = chess_corners::Uncertainty::off();
        for (const chess_corners::Corner& c :
             chess_corners::detect(image, width, height, quiet)) {
            if (c.position_covariance) {
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "detection failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
