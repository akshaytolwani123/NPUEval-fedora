// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void rsqrt_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    constexpr float THREE_HALFS = 1.5f;
    constexpr float HALF = 0.5f;

    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        if (x <= 0.0f) {
            output[i] = bfloat16(0.0f);
            continue;
        }

        union { uint32_t u; float f; } fu{ .f = x };
        union { uint32_t u; float f; } gu{ .u = 0x5f3759dfu - (fu.u >> 1) };
        float y = gu.f;

        y = y * (THREE_HALFS - HALF * x * y * y);
        y = y * (THREE_HALFS - HALF * x * y * y);

        output[i] = bfloat16(y);
    }
}