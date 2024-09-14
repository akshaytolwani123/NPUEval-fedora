// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void relu6_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        float y = x > 0.0f ? x : 0.0f;
        y = y < 6.0f ? y : 6.0f;
        output[i] = bfloat16(y);
    }
}