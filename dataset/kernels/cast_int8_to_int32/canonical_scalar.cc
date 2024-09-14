// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void cast_int8_to_int32(int8_t* input_vector, int32_t* output_vector) {
    constexpr int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        output_vector[i] = (int32_t)input_vector[i];
    }
}