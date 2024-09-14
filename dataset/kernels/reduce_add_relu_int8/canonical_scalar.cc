// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void reduce_add_relu_int8(int8_t *in_buffer, int8_t* out_buffer) {
    constexpr int32_t NUM_ELEMENTS = 128;
    int32_t sum = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        sum += in_buffer[i];
    }
    if (sum < 0)
        sum = 0;
    if (sum > 127) sum = 127;
    if (sum < -128) sum = -128;
    *out_buffer = (int8_t)sum;
}