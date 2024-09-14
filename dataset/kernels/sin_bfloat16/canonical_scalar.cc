// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <math.h>

void sin_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    constexpr float PI = 3.14159265359f;

    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];

        while (x > PI) x -= 2.0f * PI;
        while (x < -PI) x += 2.0f * PI;

        float x2 = x * x;
        float x3 = x2 * x;
        float x5 = x3 * x2;
        float x7 = x5 * x2;
        float x9 = x7 * x2;
        float x11 = x9 * x2;

        float result = x 
                    - x3 * 0.166666666667f
                    + x5 * 0.00833333333333f
                    - x7 * 0.000198412698413f
                    + x9 * 0.00000275573192f
                    - x11 * 0.0000000250521084f;

        output_vector[i] = (bfloat16)result;
    }
}