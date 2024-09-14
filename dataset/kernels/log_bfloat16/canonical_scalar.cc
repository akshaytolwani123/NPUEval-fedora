// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void log_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    static const float LUT[17] = {
        0.00000000f, 0.06062462f, 0.11778304f, 0.17185026f,
        0.22314355f, 0.27193372f, 0.31845373f, 0.36290549f,
        0.40546511f, 0.44628710f, 0.48550782f, 0.52324814f,
        0.55961579f, 0.59470711f, 0.62860866f, 0.66139848f,
        0.69314718f
    };
    union { uint32_t u; float f; } uinf{ .u = 0xFF800000u };
    constexpr float LN2 = 0.69314718056f;

    for (int i = 0; i < N; ++i) {
        union { uint32_t u; float f; } fu{ .f = float(input[i]) };
        float x = fu.f;

        if (x <= 0.0f) {
            output[i] = bfloat16(uinf.f);
            continue;
        }
        int E = int((fu.u >> 23) & 0xFF) - 127;
        fu.u = (fu.u & 0x7FFFFFu) | 0x3F800000u;
        float m = fu.f - 1.0f;
        float t = m * 16.0f;
        int idx = int(t);
        if (idx >= 16) idx = 15;
        float frac = t - float(idx);
        float y_m = LUT[idx] + frac * (LUT[idx+1] - LUT[idx]);
        float y = float(E) * LN2 + y_m;
        output[i] = bfloat16(y);
    }
}