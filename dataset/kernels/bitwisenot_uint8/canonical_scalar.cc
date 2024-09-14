// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void bitwisenot_uint8(uint8_t *in_buffer1, uint8_t *out_buffer) {
    constexpr uint32_t nbytes = 1024;
    for (uint32_t i = 0; i < nbytes; ++i) {
        out_buffer[i] = ~in_buffer1[i];
    }
}