// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <math.h>

void tanh_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];
        if (x > 4.97f) {
            output_vector[i] = (bfloat16)1.0f;
            continue;
        }
        if (x < -4.97f) {
            output_vector[i] = (bfloat16)-1.0f;
            continue;
        }
        float abs_x = x < 0 ? -x : x;
        float x2 = abs_x * abs_x;
        float numerator = abs_x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float denominator = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        float result = numerator / denominator;
        result = x < 0 ? -result : result;
        output_vector[i] = (bfloat16)result;
    }
}