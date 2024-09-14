// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void reducemin_bfloat16(bfloat16 *in_buffer, bfloat16* out_buffer) {
    constexpr int32_t num_elements = 64;
    bfloat16 min_value = in_buffer[0];
    for (int i = 1; i < num_elements; i++) {
        if (min_value > in_buffer[i]) {
            min_value = in_buffer[i];
        }
    }
    *out_buffer = min_value;
}