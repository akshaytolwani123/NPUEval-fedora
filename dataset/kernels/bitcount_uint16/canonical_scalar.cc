// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void bitcount_uint16(uint16_t *input_vector, uint16_t *output_vector) {
    constexpr int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        uint16_t val = input_vector[i];
        uint16_t count = 0;
        while (val) {
            count += val & 1;
            val >>= 1;
        }
        output_vector[i] = count;
    }
}