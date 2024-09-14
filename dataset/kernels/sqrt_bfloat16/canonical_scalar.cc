// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void sqrt_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;

    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        if (x <= 0.0f) {
            output[i] = bfloat16(0.0f);
            continue;
        }

        union { uint32_t u; float f; } fu{ .f = x };
        union { uint32_t u; float f; } gu{ .u = (fu.u >> 1) + 0x1F800000u };
        float y = gu.f;

        y = 0.5f * (y + x / y);
        y = 0.5f * (y + x / y);

        output[i] = bfloat16(y);
    }
}