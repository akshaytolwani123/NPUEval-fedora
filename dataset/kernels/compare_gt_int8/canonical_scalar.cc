// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void compare_gt_int8(int8_t* in_buffer0, int8_t* in_buffer1, int8_t* out_buffer) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; i++) {
        out_buffer[i] = in_buffer0[i] > in_buffer1[i] ? 1 : 0;
    }
}