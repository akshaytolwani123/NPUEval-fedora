// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void bitcount_uint8(uint8_t *input_vector, uint8_t *output_vector) {
    constexpr int32_t vector_size = 256;
    for (int i = 0; i < vector_size; i++) {
        uint8_t x = input_vector[i];
        uint8_t count = 0;
        for (int b = 0; b < 8; b++) {
            count += (x >> b) & 1;
        }
        output_vector[i] = count;
    }
}