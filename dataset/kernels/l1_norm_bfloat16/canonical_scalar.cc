// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void l1_norm_bfloat16(bfloat16* in_buffer, bfloat16* out_buffer) {
    constexpr int32_t N = 256;
    bfloat16 sum = 0;
    for (int i = 0; i < N; ++i) {
        bfloat16 val = in_buffer[i];
        sum += (val >= 0.0f) ? val : -val;
    }
    *out_buffer = sum;
}