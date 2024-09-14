// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void cast_bfloat16_to_float32(bfloat16* in_buffer, float* out_buffer) {
    constexpr int32_t N = 512;
    for (int i = 0; i < N; i++) {
        out_buffer[i] = float(in_buffer[i]);
    }
}