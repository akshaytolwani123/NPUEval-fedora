// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void ceil_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        float val = (float)input_vector[i];
        int32_t int_part = (int32_t)val;
        if (val == (float)int_part || val < 0.0f) {
            output_vector[i] = (bfloat16)int_part;
        } else {
            output_vector[i] = (bfloat16)(int_part + 1);
        }
    }
}