// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void cast_bfloat16_to_int8(bfloat16* input, int8_t* output) {
    constexpr int N = 256;
    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        int xi = int(x);
        int floor_x;
        if (x >= 0.0f) {
            floor_x = xi;
        } else {
            if (float(xi) == x) {
                floor_x = xi;
            } else {
                floor_x = xi - 1;
            }
        }
        float frac = x - float(floor_x);
        int r;
        if (frac < 0.5f) {
            r = floor_x;
        } else if (frac > 0.5f) {
            r = floor_x + 1;
        } else {
            r = (floor_x & 1) == 0 ? floor_x : (floor_x + 1);
        }
        if (r > 127)      r = 127;
        else if (r < -128) r = -128;
        output[i] = int8_t(r);
    }
}