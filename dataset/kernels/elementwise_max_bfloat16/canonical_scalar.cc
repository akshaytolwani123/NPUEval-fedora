// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

void elementwise_max_bfloat16(bfloat16* input0, bfloat16* input1, bfloat16* output) {
    constexpr int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float a = float(input0[i]);
        float b = float(input1[i]);
        output[i] = bfloat16(a > b ? a : b);
    }
}