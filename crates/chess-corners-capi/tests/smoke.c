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

    free(img);
    printf("smoke OK\n");
    return 0;
}
