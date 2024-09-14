// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void negate_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr uint32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        output_vector[i] = (bfloat16)(-((float)input_vector[i]));
    }
}