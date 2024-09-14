// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectorsubtract_bfloat16(bfloat16* input_a, bfloat16* input_b, bfloat16* output) {
    constexpr int32_t NUM_ELEMENTS = 256;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        output[i] = input_a[i] - input_b[i];
    }
}