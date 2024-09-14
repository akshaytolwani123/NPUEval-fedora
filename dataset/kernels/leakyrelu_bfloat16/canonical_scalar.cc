// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void leaky_relu_bfloat16(bfloat16 *in_buffer, bfloat16 *out_buffer) {
    constexpr int32_t num_elements = 256;
    bfloat16 zero_value = 0.0f;
    bfloat16 alpha_value = 0.01f;

    for (uint32_t i = 0; i < num_elements; ++i) {
        bfloat16 input_value = *in_buffer;
        in_buffer++;

        bfloat16 relu_out = input_value > zero_value ? input_value : zero_value;
        bfloat16 leaky_out = input_value < zero_value ? input_value * alpha_value : input_value;
        bfloat16 output_value = input_value < zero_value ? leaky_out : relu_out;

        *out_buffer = output_value;
        out_buffer++;
    }
}