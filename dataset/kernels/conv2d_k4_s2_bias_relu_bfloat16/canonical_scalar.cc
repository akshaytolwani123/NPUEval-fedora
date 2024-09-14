// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void conv2d_k4_s2_bias_relu_bfloat16(bfloat16* in_buffer, bfloat16* kernel, bfloat16* out_buffer, bfloat16 bias, uint32_t stride) {
    constexpr int32_t IN_ROWS = 16;
    constexpr int32_t IN_COLS = 16;
    constexpr int32_t KERNEL_SIZE = 4;

    uint32_t out_rows = (IN_ROWS - KERNEL_SIZE) / stride + 1;
    uint32_t out_cols = (IN_COLS - KERNEL_SIZE) / stride + 1;

    for (uint32_t i = 0; i < out_rows; i++) {
        for (uint32_t j = 0; j < out_cols; j++) {
            bfloat16 acc = 0;
            for (uint32_t ki = 0; ki < KERNEL_SIZE; ki++) {
                for (uint32_t kj = 0; kj < KERNEL_SIZE; kj++) {
                    acc += in_buffer[(i * stride + ki) * IN_COLS + (j * stride + kj)] * kernel[ki * KERNEL_SIZE + kj];
                }
            }
            acc += bias;
            bfloat16 relu = acc > 0.0f ? acc : 0.0f;
            out_buffer[i * out_cols + j] = relu;
        }
    }
}