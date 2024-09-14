// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void avgpool1d_bfloat16(bfloat16 *in_buffer, bfloat16 *out_buffer, 
                       uint32_t window_size, uint32_t stride) {
    constexpr uint32_t num_elements = 1024;
    uint32_t num_windows = (num_elements - window_size) / stride + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < window_size; j++) {
            float current_val = (float)*(in_buffer + i * stride + j);
            sum += current_val;
        }
        float avg = sum / window_size;
        out_buffer[i] = (bfloat16)avg;
    }
}