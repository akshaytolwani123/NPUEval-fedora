// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void conv1d_k2_s1_bias_relu_bfloat16(bfloat16* in_buffer, bfloat16* kernel, bfloat16* out_buffer, bfloat16 bias) {
    constexpr int32_t VECTOR_SIZE = 256;
    constexpr int32_t KERNEL_SIZE = 2;
    constexpr int32_t STRIDE = 1;
    uint32_t num_windows = (VECTOR_SIZE - KERNEL_SIZE) / STRIDE + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += (float)in_buffer[i * STRIDE + j] * (float)kernel[j];
        }
        acc += (float)bias;
        float relu = acc > 0.0f ? acc : 0.0f;
        out_buffer[i] = (bfloat16)relu;
    }
}