// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectoradd_int16(int16_t *in_buffer1, int16_t *in_buffer2, int16_t *out_buffer) {
    constexpr int32_t vector_size = 256;
    for (int32_t i = 0; i < vector_size; ++i) {
        out_buffer[i] = in_buffer1[i] + in_buffer2[i];
    }
}