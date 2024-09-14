// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void compare_equal_int32(int32_t* a, int32_t* b, uint8_t* out) {
    constexpr int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        out[i] = (a[i] == b[i]) ? 1 : 0;
    }
}