#ifndef CHESS_CORNERS_HPP
#define CHESS_CORNERS_HPP

#pragma once

// Header-only C++17 convenience wrapper over the chess-corners C ABI
// (chess_corners.h). It adds RAII ownership of the detector result,
// std::vector returns, value-type corners, and exceptions instead of status
// codes. It contains no detection logic: every call marshals to the C ABI.

#include "chess_corners.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace chess_corners {

/// ABI revision this header was written against.
///
/// Checked against `cc_abi_version()` before each detection (see
/// `check_abi()`); a mismatch means the header and the linked library are
/// incompatible and detection throws rather than reading a stale layout.
inline constexpr std::uint32_t CHESS_CORNERS_ABI_VERSION = 4;

/// One local grid-axis direction with its 1-sigma angular uncertainty.
///
/// `angle` and `sigma` are in radians, matching `cc_axis`.
struct Axis {
    float angle = 0.0f;
    float sigma = 0.0f;
};

/// One detected corner in full-resolution image pixels.
///
/// `axes` follow the joint polarity convention of the C ABI: `axes[0].angle`
/// lies in `[0, pi)` and `axes[1].angle` in `(axes[0].angle, axes[0].angle +
/// pi)`. They are only meaningful when `has_orientation` is `true`; when the
/// orientation fit was skipped (`CC_ORIENTATION_NONE`) `has_orientation` is
/// `false` and `axes` is zeroed.
struct Corner {
    float x = 0.0f;
    float y = 0.0f;
    float response = 0.0f;
    std::array<Axis, 2> axes{};
    bool has_orientation = true;
};

/// Exception thrown when a C entry point returns a non-`CC_OK` status.
///
/// Carries the originating `cc_status` and, as the `what()` message, the
/// library's own `cc_status_str` description. An ABI-version mismatch is
/// reported separately as a plain `std::runtime_error` (it is a linkage fault,
/// not a detection status); both derive from `std::runtime_error`, so a single
/// `catch (const std::runtime_error&)` covers either.
class Error : public std::runtime_error {
public:
    explicit Error(cc_status status)
        : std::runtime_error(describe(status)), status_(status) {}

    /// The underlying C status code.
    [[nodiscard]] cc_status status() const noexcept { return status_; }

private:
    static std::string describe(cc_status status) {
        const char* text = cc_status_str(status);
        return text != nullptr ? std::string(text) : std::string("unknown status");
    }

    cc_status status_;
};

/// Flat detector configuration mirroring `cc_config`.
///
/// Start from a preset (`Config::chess()`, `Config::radon_multiscale()`, ...)
/// and tweak the public members. The default-constructed `Config` matches
/// `Config::chess()`.
///
/// `chess_ring` applies to the ChESS strategy only; `ray_radius`,
/// `image_upsample`, `response_blur_radius`, and `peak_fit` apply to the
/// Radon strategy only. Fields that do not apply to the active `strategy`
/// are ignored.
struct Config {
    cc_strategy_t strategy;
    float threshold;
    std::uint32_t nms_radius;
    std::uint32_t min_cluster_size;
    cc_refiner_t refiner;
    cc_orientation_method_t orientation_method;
    std::uint32_t multiscale;
    float merge_radius;
    std::uint32_t upscale_factor;
    cc_chess_ring_t chess_ring;
    std::uint32_t ray_radius;
    std::uint32_t image_upsample;
    std::uint32_t response_blur_radius;
    cc_peak_fit_t peak_fit;

    /// Defaults to the single-scale ChESS preset.
    Config() noexcept : Config(from_c(cc_config_chess())) {}

    /// Wrap a C-level configuration.
    static Config from_c(const cc_config& c) noexcept {
        Config out{Uninit{}};
        out.strategy = c.strategy;
        out.threshold = c.threshold;
        out.nms_radius = c.nms_radius;
        out.min_cluster_size = c.min_cluster_size;
        out.refiner = c.refiner;
        out.orientation_method = c.orientation_method;
        out.multiscale = c.multiscale;
        out.merge_radius = c.merge_radius;
        out.upscale_factor = c.upscale_factor;
        out.chess_ring = c.chess_ring;
        out.ray_radius = c.ray_radius;
        out.image_upsample = c.image_upsample;
        out.response_blur_radius = c.response_blur_radius;
        out.peak_fit = c.peak_fit;
        return out;
    }

    /// Lower to the C-level configuration consumed by `cc_detect_u8`.
    [[nodiscard]] cc_config to_c() const noexcept {
        cc_config c;
        c.strategy = strategy;
        c.threshold = threshold;
        c.nms_radius = nms_radius;
        c.min_cluster_size = min_cluster_size;
        c.refiner = refiner;
        c.orientation_method = orientation_method;
        c.multiscale = multiscale;
        c.merge_radius = merge_radius;
        c.upscale_factor = upscale_factor;
        c.chess_ring = chess_ring;
        c.ray_radius = ray_radius;
        c.image_upsample = image_upsample;
        c.response_blur_radius = response_blur_radius;
        c.peak_fit = peak_fit;
        return c;
    }

    /// Default configuration (single-scale ChESS).
    static Config default_() noexcept { return from_c(cc_config_default()); }
    /// Single-scale ChESS preset.
    static Config chess() noexcept { return from_c(cc_config_chess()); }
    /// Three-level coarse-to-fine ChESS preset.
    static Config chess_multiscale() noexcept { return from_c(cc_config_chess_multiscale()); }
    /// Single-scale Radon preset.
    static Config radon() noexcept { return from_c(cc_config_radon()); }
    /// Three-level coarse-to-fine Radon preset.
    static Config radon_multiscale() noexcept { return from_c(cc_config_radon_multiscale()); }

private:
    // Tag used by the default constructor to build an uninitialized aggregate
    // before copying preset fields in, without recursing into `Config()`.
    struct Uninit {};
    explicit Config(Uninit) noexcept {}
};

/// Throw unless the linked library's ABI matches this header.
///
/// Called automatically by `detect`; exposed so callers can validate linkage
/// explicitly (e.g. at startup). Throws `std::runtime_error` on mismatch.
inline void check_abi() {
    const std::uint32_t lib = cc_abi_version();
    if (lib != CHESS_CORNERS_ABI_VERSION) {
        throw std::runtime_error(
            "chess_corners: ABI version mismatch (header " +
            std::to_string(CHESS_CORNERS_ABI_VERSION) + ", library " +
            std::to_string(lib) + ")");
    }
}

namespace detail {

/// RAII owner of a `cc_result`; frees it (null-safe, idempotent) on scope exit,
/// including during stack unwinding.
class ResultGuard {
public:
    ResultGuard() noexcept : result_{nullptr, 0} {}
    ~ResultGuard() { cc_result_free(&result_); }

    ResultGuard(const ResultGuard&) = delete;
    ResultGuard& operator=(const ResultGuard&) = delete;
    ResultGuard(ResultGuard&&) = delete;
    ResultGuard& operator=(ResultGuard&&) = delete;

    cc_result* ptr() noexcept { return &result_; }
    [[nodiscard]] const cc_result& get() const noexcept { return result_; }

private:
    cc_result result_;
};

}  // namespace detail

/// Detect corners in an 8-bit, row-major grayscale image.
///
/// `pixels` must point to at least `width * height` readable bytes. Throws
/// `chess_corners::Error` if the C ABI returns a non-`CC_OK` status, or
/// `std::runtime_error` on an ABI-version mismatch.
inline std::vector<Corner> detect(const std::uint8_t* pixels,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  const Config& config = Config::chess()) {
    check_abi();

    const cc_config c = config.to_c();
    detail::ResultGuard guard;
    const cc_status status = cc_detect_u8(pixels, width, height, &c, guard.ptr());
    if (status != CC_OK) {
        throw Error(status);
    }

    const cc_result& result = guard.get();
    std::vector<Corner> corners;
    corners.reserve(result.len);
    for (std::size_t i = 0; i < result.len; ++i) {
        const cc_corner& src = result.corners[i];
        Corner dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.response = src.response;
        dst.axes[0] = Axis{src.axes[0].angle, src.axes[0].sigma};
        dst.axes[1] = Axis{src.axes[1].angle, src.axes[1].sigma};
        dst.has_orientation = src.has_orientation != 0;
        corners.push_back(dst);
    }
    return corners;
}

/// Detect corners in a contiguous grayscale buffer.
///
/// Throws `chess_corners::Error` (`CC_ERR_DIMENSION_MISMATCH`) if `pixels`
/// holds fewer than `width * height` bytes.
inline std::vector<Corner> detect(const std::vector<std::uint8_t>& pixels,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  const Config& config = Config::chess()) {
    if (pixels.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw Error(CC_ERR_DIMENSION_MISMATCH);
    }
    return detect(pixels.data(), width, height, config);
}

}  // namespace chess_corners

#endif  // CHESS_CORNERS_HPP
