// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void log2_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    static const float LUT[17] = {
        0.0f, 0.08746284f, 0.16992500f, 0.24792751f,
        0.32192809f, 0.39231742f, 0.45943162f, 0.52356196f,
        0.58496250f, 0.64385619f, 0.70043972f, 0.75488750f,
        0.80735492f, 0.85798100f, 0.90689060f, 0.95419631f,
        1.0f
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
        float y = float(E) + y_m;
        output[i] = bfloat16(y);
    }
}