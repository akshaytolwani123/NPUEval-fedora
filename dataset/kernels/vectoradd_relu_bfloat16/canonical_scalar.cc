// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectoradd_relu_bfloat16(bfloat16* in0, bfloat16* in1, bfloat16* out) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; i++) {
        bfloat16 sum = in0[i] + in1[i];
        bfloat16 relu = sum > 0 ? sum : 0;
        out[i] = relu;
    }
}