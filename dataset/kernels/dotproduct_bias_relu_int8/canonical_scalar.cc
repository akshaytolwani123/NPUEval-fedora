// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void dotproduct_bias_relu_int8(int8_t* in0, int8_t* in1, int8_t* out, int8_t bias) {
    constexpr int32_t N = 256;
    int32_t acc = 0;
    for (int i = 0; i < N; i++) {
        acc += (int32_t)in0[i] * (int32_t)in1[i];
    }
    acc += bias;
    if (acc < 0) acc = 0;
    if (acc > 127) acc = 127;
    if (acc < -128) acc = -128;
    *out = (int8_t)acc;
}