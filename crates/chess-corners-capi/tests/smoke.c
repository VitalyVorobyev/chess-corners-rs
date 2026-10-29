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
    cc_result out = {NULL, 0, 0, 0.0f};
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
    cc_result ignored = {NULL, 0, 0, 0.0f};
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

    /* Every localizer reports a position covariance; check the saddle-point
     * image refiner. This board is saturated (0/255), so its noise cannot be
     * estimated: supply it. */
    cc_config saddle = cc_config_chess();
    saddle.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
    saddle.uncertainty = CC_UNCERTAINTY_PROPAGATED;
    saddle.has_noise_sigma = 1;
    saddle.noise_sigma = 1.0f;
    cc_result cov_out = {NULL, 0, 0, 0.0f};
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

    /* The presets and a zero-initialised config select CC_UNCERTAINTY_OFF.
     * Propagated reports a covariance for the center of mass too;
     * CC_UNCERTAINTY_OFF reports none, and an unknown tag is rejected. */
    if (cc_config_default().uncertainty != CC_UNCERTAINTY_OFF ||
        cc_config_chess().uncertainty != CC_UNCERTAINTY_OFF ||
        cc_config_radon().uncertainty != CC_UNCERTAINTY_OFF) {
        fprintf(stderr, "preset does not default to uncertainty off\n");
        free(img);
        return 10;
    }
    cc_config zeroed = {0};
    if (zeroed.uncertainty != CC_UNCERTAINTY_OFF) {
        fprintf(stderr, "zero-initialised config is not uncertainty off\n");
        free(img);
        return 10;
    }
    cc_config plain = cc_config_chess();
    plain.uncertainty = CC_UNCERTAINTY_PROPAGATED;
    plain.has_noise_sigma = 1;
    plain.noise_sigma = 1.0f;
    cc_result plain_out = {NULL, 0, 0, 0.0f};
    if (cc_detect_u8(img, w, h, &plain, &plain_out) != CC_OK) {
        fprintf(stderr, "center-of-mass detection failed\n");
        free(img);
        return 10;
    }
    size_t plain_cov = 0;
    for (size_t i = 0; i < plain_out.len; ++i) {
        plain_cov += plain_out.corners[i].has_position_covariance ? 1u : 0u;
    }
    /* The result carries the supplied noise level behind the covariances. */
    const int plain_sigma_ok =
        plain_out.has_noise_sigma == 1 && plain_out.noise_sigma == 1.0f;
    cc_result_free(&plain_out);
    if (plain_cov == 0) {
        fprintf(stderr, "center of mass carried no position covariance\n");
        free(img);
        return 10;
    }
    if (!plain_sigma_ok) {
        fprintf(stderr, "result did not carry the supplied noise sigma\n");
        free(img);
        return 10;
    }
    /* Under CC_UNCERTAINTY_OFF the noise level is not read. */
    plain.uncertainty = CC_UNCERTAINTY_OFF;
    plain.noise_sigma = -1.0f;
    cc_result off_out = {NULL, 0, 0, 0.0f};
    if (cc_detect_u8(img, w, h, &plain, &off_out) != CC_OK) {
        fprintf(stderr, "uncertainty-off detection failed\n");
        free(img);
        return 11;
    }
    if (off_out.has_noise_sigma != 0) {
        fprintf(stderr, "uncertainty off still reported a noise sigma\n");
        cc_result_free(&off_out);
        free(img);
        return 11;
    }
    for (size_t i = 0; i < off_out.len; ++i) {
        if (off_out.corners[i].has_position_covariance) {
            fprintf(stderr, "uncertainty off still reported a covariance\n");
            cc_result_free(&off_out);
            free(img);
            return 11;
        }
    }
    cc_result_free(&off_out);
    plain.uncertainty = 99;
    if (cc_detect_u8(img, w, h, &plain, &ignored) != CC_ERR_INVALID_CONFIG) {
        fprintf(stderr, "unknown uncertainty tag was not rejected\n");
        free(img);
        return 12;
    }

    saddle.noise_sigma = -1.0f; /* must be finite and >= 0 */
    if (cc_detect_u8(img, w, h, &saddle, &ignored) != CC_ERR_INVALID_CONFIG) {
        fprintf(stderr, "negative noise_sigma was not rejected\n");
        free(img);
        return 8;
    }

    /* Every preset and a zero-initialised config run no image refiner; the
     * saddle-point refiner applies to Radon too, and an unknown tag is
     * rejected. */
    cc_config radon = cc_config_radon();
    if (radon.refiner != CC_IMAGE_REFINER_NONE ||
        cc_config_chess().refiner != CC_IMAGE_REFINER_NONE ||
        zeroed.refiner != CC_IMAGE_REFINER_NONE) {
        fprintf(stderr, "default config runs an image refiner\n");
        free(img);
        return 9;
    }
    radon.refiner = CC_IMAGE_REFINER_SADDLE_POINT;
    cc_result radon_out = {NULL, 0, 0, 0.0f};
    if (cc_detect_u8(img, w, h, &radon, &radon_out) != CC_OK || radon_out.len == 0) {
        fprintf(stderr, "Radon saddle-point detection failed\n");
        cc_result_free(&radon_out);
        free(img);
        return 9;
    }
    cc_result_free(&radon_out);
    radon.refiner = 99;
    if (cc_detect_u8(img, w, h, &radon, &ignored) != CC_ERR_INVALID_CONFIG) {
        fprintf(stderr, "unknown image-refiner tag was not rejected\n");
        free(img);
        return 10;
    }

    /* The saturated board has no structure-free area: no noise estimate. */
    float sigma = -1.0f;
    uint32_t has_sigma = 7;
    if (cc_estimate_noise_sigma_u8(img, w, h, &sigma, &has_sigma) != CC_OK ||
        has_sigma != 0) {
        fprintf(stderr, "noise estimate on a saturated board\n");
        free(img);
        return 11;
    }

    free(img);
    printf("smoke OK\n");
    return 0;
}
