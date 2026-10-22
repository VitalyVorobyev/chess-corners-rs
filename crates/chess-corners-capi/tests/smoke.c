/*
 * Minimal C smoke test for the chess-corners C ABI.
 *
 * Not built by Cargo. Compile and run it against the built cdylib, e.g. on
 * macOS from the workspace root:
 *
 *   cargo build -p chess-corners-capi --release
 *   clang crates/chess-corners-capi/tests/smoke.c \
 *       -I crates/chess-corners-capi/include \
 *       -L target/release -lchess_corners_capi \
 *       -o target/release/cc_smoke
 *   DYLD_LIBRARY_PATH=target/release target/release/cc_smoke
 *
 * On Linux substitute LD_LIBRARY_PATH for DYLD_LIBRARY_PATH.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "chess_corners.h"

int main(void) {
    const uint32_t w = 128, h = 128;
    uint8_t *img = (uint8_t *)calloc((size_t)w * (size_t)h, 1);
    if (img == NULL) {
        return 2;
    }
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            if (((x / 16u) + (y / 16u)) % 2u == 0u) {
                img[(size_t)y * (size_t)w + (size_t)x] = 255;
            }
        }
    }

    cc_config cfg = cc_config_chess();
    cc_result out = {NULL, 0};
    cc_status st = cc_detect_u8(img, w, h, &cfg, &out);
    if (st != CC_OK) {
        fprintf(stderr, "cc_detect_u8 failed: %s\n", cc_status_str(st));
        free(img);
        return 1;
    }

    printf("abi=%u detected=%zu corners\n", cc_abi_version(), out.len);
    if (out.len == 0 || out.corners == NULL) {
        fprintf(stderr, "expected a non-empty detection on the checkerboard\n");
        cc_result_free(&out);
        free(img);
        return 1;
    }

    cc_result_free(&out);
    if (out.corners != NULL || out.len != 0) {
        fprintf(stderr, "cc_result_free did not clear the result\n");
        free(img);
        return 3;
    }

    /* Invalid values in the new flat fields must map to a clean status
     * (not a panic or UB). An out-of-range upscale factor is rejected with
     * CC_ERR_UPSCALE; an unknown ring tag with CC_ERR_INVALID_CONFIG. */
    cc_config bad_upscale = cc_config_chess();
    bad_upscale.upscale_factor = 5; /* only 0 (off) and 2..4 are valid */
    cc_result ignored = {NULL, 0};
    if (cc_detect_u8(img, w, h, &bad_upscale, &ignored) != CC_ERR_UPSCALE) {
        fprintf(stderr, "invalid upscale factor was not rejected\n");
        free(img);
        return 4;
    }

    cc_config bad_ring = cc_config_chess();
    bad_ring.chess_ring = 99; /* unknown tag */
    if (cc_detect_u8(img, w, h, &bad_ring, &ignored) != CC_ERR_INVALID_CONFIG) {
        fprintf(stderr, "unknown ring tag was not rejected\n");
        free(img);
        return 5;
    }

    /* The saddle-point refiner reports a position covariance. This board is
     * saturated (0/255), so its noise cannot be estimated: supply it. */
    cc_config saddle = cc_config_chess();
    saddle.refiner = CC_REFINER_SADDLE_POINT;
    saddle.has_noise_sigma = 1;
    saddle.noise_sigma = 1.0f;
    cc_result cov_out = {NULL, 0};
    if (cc_detect_u8(img, w, h, &saddle, &cov_out) != CC_OK) {
        fprintf(stderr, "saddle-point detection failed\n");
        free(img);
        return 6;
    }
    size_t with_cov = 0;
    for (size_t i = 0; i < cov_out.len; ++i) {
        const cc_corner *c = &cov_out.corners[i];
        if (c->has_position_covariance && c->position_covariance.xx > 0.0f &&
            c->position_covariance.yy > 0.0f) {
            ++with_cov;
        }
    }
    cc_result_free(&cov_out);
    if (with_cov == 0) {
        fprintf(stderr, "no corner carried a position covariance\n");
        free(img);
        return 7;
    }

    saddle.noise_sigma = -1.0f; /* must be finite and >= 0 */
    if (cc_detect_u8(img, w, h, &saddle, &ignored) != CC_ERR_INVALID_CONFIG) {
        fprintf(stderr, "negative noise_sigma was not rejected\n");
        free(img);
        return 8;
    }

    free(img);
    printf("smoke OK\n");
    return 0;
}
