// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void hardsigmoid_bfloat16(bfloat16 *in_buffer, bfloat16 *out_buffer) {
    constexpr int32_t num_elements = 256;
    bfloat16 zero_value = 0.0f;
    bfloat16 one_value = 1.0f;
    bfloat16 slope_value = 0.2f;
    bfloat16 offset_value = 0.5f;

    for (uint32_t i = 0; i < num_elements; i++) {
        bfloat16 input_value = *in_buffer;
        in_buffer++;

        bfloat16 result = input_value * slope_value + offset_value;

        if (result > one_value) {
            result = one_value;
        }
        if (result < zero_value) {
            result = zero_value;
        }

        out_buffer[i] = result;
    }
}