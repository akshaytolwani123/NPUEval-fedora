// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void elementwise_max_int8(int8_t* in_buffer1, int8_t* in_buffer2, int8_t* out_buffer) {
    constexpr int32_t VECTOR_SIZE = 512;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        int8_t a = in_buffer1[i];
        int8_t b = in_buffer2[i];
        out_buffer[i] = (a > b) ? a : b;
    }
}