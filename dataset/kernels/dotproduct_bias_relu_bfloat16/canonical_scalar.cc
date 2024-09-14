// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void dotproduct_bias_relu_bfloat16(bfloat16* in0, bfloat16* in1, bfloat16* out, bfloat16 bias, uint32_t vector_size) {
    bfloat16 acc = 0;
    for (uint32_t i = 0; i < vector_size; i++) {
        acc += in0[i] * in1[i];
    }
    acc += bias;
    acc = acc > 0 ? acc : 0;
    *out = acc;
}