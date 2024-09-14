// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void conv1d_int32(int32_t *in_buffer, int32_t* kernel, int32_t* out_buffer, uint32_t stride) {
    constexpr int32_t vector_size = 256;
    constexpr int32_t kernel_size = 3;
    uint32_t num_windows = (vector_size - kernel_size) / stride + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        int32_t conv_sum = 0;
        for (uint32_t j = 0; j < kernel_size; j++) {
            conv_sum += in_buffer[i * stride + j] * kernel[j];
        }
        out_buffer[i] = conv_sum;
    }
}