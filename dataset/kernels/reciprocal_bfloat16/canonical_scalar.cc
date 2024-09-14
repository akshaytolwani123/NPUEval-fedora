// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void reciprocal_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];
        output_vector[i] = (bfloat16)(1.0f / x);
    }
}