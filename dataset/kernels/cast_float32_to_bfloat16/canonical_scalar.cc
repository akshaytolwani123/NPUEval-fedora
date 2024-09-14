// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void cast_float32_to_bfloat16(float *in_buffer, bfloat16 *out_buffer) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; ++i) {
        out_buffer[i] = bfloat16(in_buffer[i]);
    }
}