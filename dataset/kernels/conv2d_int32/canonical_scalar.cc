// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void conv2d_int32(int32_t *input_matrix, int32_t* kernel, int32_t* output_matrix, uint32_t stride) {
    constexpr int32_t input_rows = 16;
    constexpr int32_t input_cols = 16;
    constexpr int32_t kernel_size = 3;
    uint32_t output_rows = (input_rows - kernel_size) / stride + 1;
    uint32_t output_cols = (input_cols - kernel_size) / stride + 1;

    for (uint32_t i = 0; i < output_rows; i++) {
        for (uint32_t j = 0; j < output_cols; j++) {
            int32_t conv_sum = 0;
            for (uint32_t ki = 0; ki < kernel_size; ki++) {
                for (uint32_t kj = 0; kj < kernel_size; kj++) {
                    conv_sum += input_matrix[(i * stride + ki) * input_cols + (j * stride + kj)] * kernel[ki * kernel_size + kj];
                }
            }
            output_matrix[i * output_cols + j] = conv_sum;
        }
    }
}