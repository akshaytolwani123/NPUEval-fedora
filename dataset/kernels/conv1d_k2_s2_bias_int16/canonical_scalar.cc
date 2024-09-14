// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdint>

void conv1d_k2_s2_bias_int16(int16_t* in_buffer, int16_t* kernel, int16_t* out_buffer, int16_t bias) {
    constexpr int32_t VECTOR_SIZE = 128;
    constexpr int32_t KERNEL_SIZE = 2;
    constexpr int32_t STRIDE = 2;
    uint32_t num_windows = (VECTOR_SIZE - KERNEL_SIZE) / STRIDE + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        int16_t acc = bias;
        for (uint32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += in_buffer[i * STRIDE + j] * kernel[j];
        }
        if (acc >  32767) acc =  32767;
        if (acc < -32768) acc = -32768;
        out_buffer[i] = acc;
    }
}