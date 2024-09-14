// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void cumsum_bfloat16(bfloat16* in, bfloat16* out) {
    constexpr int N = 256;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += float(in[i]);
        out[i] = bfloat16(sum);
    }
}