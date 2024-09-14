// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void compare_lt_int8(int8_t* input_a, int8_t* input_b, int8_t* output) {
    constexpr int32_t NUM_ELEMENTS = 256;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        output[i] = (input_a[i] < input_b[i]) ? 1 : 0;
    }
}