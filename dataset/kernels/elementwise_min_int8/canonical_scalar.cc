// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void elementwise_min_int8(int8_t* input1, int8_t* input2, int8_t* output) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; ++i) {
        output[i] = (input1[i] < input2[i]) ? input1[i] : input2[i];
    }
}