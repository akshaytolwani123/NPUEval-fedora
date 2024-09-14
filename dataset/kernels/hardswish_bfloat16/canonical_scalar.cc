// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void hardswish_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    bfloat16 zero_value = 0.0f;
    bfloat16 three_value = 3.0f;
    bfloat16 six_value = 6.0f;

    for (uint32_t i = 0; i < vector_size; i++) {
        bfloat16 x = *input_vector;
        input_vector++;
        if (x < -three_value) {
            output_vector[i] = zero_value;
        } else if (x > three_value) {
            output_vector[i] = x;
        } else {
            output_vector[i] = x * (x + three_value) / six_value;
        }
    }
}