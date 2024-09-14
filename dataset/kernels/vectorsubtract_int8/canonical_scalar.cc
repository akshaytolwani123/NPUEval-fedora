// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectorsubtract_int8(int8_t *in_buffer1, int8_t *in_buffer2, int8_t *out_buffer) {
    constexpr int32_t nbytes = 512;
    for (int32_t i = 0; i < nbytes; ++i) {
        out_buffer[i] = in_buffer1[i] - in_buffer2[i];
    }
}