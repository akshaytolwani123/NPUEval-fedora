// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void divide_bfloat16(bfloat16 *a, bfloat16 *b, bfloat16 *out) {
    constexpr int32_t vector_size = 256;
    for (int i = 0; i < vector_size; i++) {
        if (b[i] == (bfloat16)0.0f) {
            out[i] = (bfloat16)0.0f;
        } else {
            out[i] = (bfloat16)((float)a[i] / (float)b[i]);
        }
    }
}