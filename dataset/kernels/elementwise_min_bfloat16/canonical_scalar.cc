// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void elementwise_min_bfloat16(bfloat16* in_buffer_a, bfloat16* in_buffer_b, bfloat16* out_buffer) {
    constexpr int32_t VEC_SIZE = 256;
    for (int i = 0; i < VEC_SIZE; ++i) {
        float a = (float)in_buffer_a[i];
        float b = (float)in_buffer_b[i];
        out_buffer[i] = (a < b) ? in_buffer_a[i] : in_buffer_b[i];
    }
}