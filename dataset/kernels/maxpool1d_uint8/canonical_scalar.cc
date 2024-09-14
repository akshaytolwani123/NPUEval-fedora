// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void maxpool1d_uint8(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t window_size, uint32_t stride) {
    constexpr uint32_t nbytes = 1024;
    uint32_t num_windows = (nbytes - window_size) / stride + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        uint8_t max_val = 0;
        for (uint32_t j = 0; j < window_size; j++) {
            uint8_t current_val = *(in_buffer + i * stride + j);
            if (current_val > max_val) {
                max_val = current_val;
            }
        }
        out_buffer[i] = max_val;
    }
}