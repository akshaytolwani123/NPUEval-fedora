// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void argmin_bfloat16(bfloat16 *input_vector, uint32_t *result) {
    constexpr int32_t vector_size = 256;
    uint32_t min_index = 0;
    bfloat16 min_value = input_vector[0];
    for (uint32_t i = 1; i < vector_size; i++) {
        if (input_vector[i] < min_value) {
            min_value = input_vector[i];
            min_index = i;
        }
    }
    *result = min_index;
}