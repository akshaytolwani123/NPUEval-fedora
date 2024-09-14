// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void log10_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    static const float LUT[17] = {
        0.00000000f, 0.02632894f, 0.05115252f, 0.07463362f,
        0.09691001f, 0.11809931f, 0.13830270f, 0.15760785f,
        0.17609126f, 0.19382003f, 0.21085337f, 0.22724378f,
        0.24303805f, 0.25827802f, 0.27300127f, 0.28724171f,
        0.30103000f
    };

    union { uint32_t u; float f; } uinf{ .u = 0xFF800000u };

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

        constexpr float LOG10_2 = 0.30102999566f;
        float y = float(E) * LOG10_2 + y_m;

        output[i] = bfloat16(y);
    }
}