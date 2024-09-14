// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void sign_int8(int8_t* input_vector, int8_t* output_vector) {
    constexpr int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        int8_t val = input_vector[i];
        if (val > 0)
            output_vector[i] = 1;
        else if (val < 0)
            output_vector[i] = -1;
        else
            output_vector[i] = 0;
    }
}