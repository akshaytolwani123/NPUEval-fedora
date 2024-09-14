// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void negate_int8(int8_t* input_vector, int8_t* output_vector) {
    constexpr uint32_t VECTOR_SIZE = 256;
    for (uint32_t i = 0; i < VECTOR_SIZE; ++i) {
        output_vector[i] = -input_vector[i];
    }
}