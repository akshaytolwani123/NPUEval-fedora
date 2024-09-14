// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void pad1d_int32(int32_t *input_vector, int32_t *padded_vector, uint32_t pad_size, int32_t pad_value) {
    constexpr int32_t vector_size = 256;
    uint32_t padded_length = vector_size + 2 * pad_size;

    for (uint32_t i = 0; i < padded_length; i++) {
        padded_vector[i] = pad_value;
    }

    for (uint32_t i = 0; i < vector_size; i++) {
        padded_vector[i + pad_size] = input_vector[i];
    }
}