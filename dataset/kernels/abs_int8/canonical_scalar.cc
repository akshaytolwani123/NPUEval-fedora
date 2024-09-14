// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void abs_int8(int8_t *in_buffer, int8_t* out_buffer) {
    constexpr int32_t nbytes = 1024;
    for (uint32_t i = 0; i < nbytes; i++) {
        int8_t value = in_buffer[i];
        if (value < 0) {
            out_buffer[i] = -value;
        } else {
            out_buffer[i] = value;
        }
    }
}