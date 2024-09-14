// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void argmax_bfloat16(bfloat16 *input_vector, uint32_t *result) {
    constexpr int32_t vector_size = 64;
    uint32_t max_index = 0;
    bfloat16 max_value = input_vector[0];

    for (uint32_t i = 1; i < vector_size; i++) {
        if (input_vector[i] > max_value) {
            max_value = input_vector[i];
            max_index = i;
        }
    }

    *result = max_index;
}