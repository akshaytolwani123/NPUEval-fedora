// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void tan_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    constexpr float PI      = 3.14159265358979323846f;
    constexpr float PI_2    = 1.57079632679489661923f;
    constexpr float PI_4    = 0.78539816339744830962f;
    constexpr float INV_PI  = 0.31830988618379067154f;

    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        float z = x * INV_PI;
        int k_int = int(z + (z >= 0.0f ? 0.5f : -0.5f));
        float y = x - float(k_int) * PI;

        bool recip = false;
        if (y > PI_4) {
            y = PI_2 - y;
            recip = true;
        } else if (y < -PI_4) {
            y = -PI_2 - y;
            recip = true;
        }

        float y2 = y * y;
        float y3 = y2 * y;
        float y5 = y3 * y2;
        float y7 = y5 * y2;
        float t  = y
                 + y3 * (1.0f / 3.0f)
                 + y5 * (2.0f / 15.0f)
                 + y7 * (17.0f / 315.0f);

        float result = recip ? (1.0f / t) : t;
        output[i] = bfloat16(result);
    }
}