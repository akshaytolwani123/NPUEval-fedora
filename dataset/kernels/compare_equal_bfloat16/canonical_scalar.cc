// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void compare_equal_bfloat16(bfloat16 *input_a, bfloat16 *input_b, uint8_t *output_mask) {
    constexpr uint32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        output_mask[i] = (input_a[i] == input_b[i]) ? 1 : 0;
    }
}