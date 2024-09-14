// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void add_offset_uint8(uint8_t *in_buffer, uint8_t* out_buffer, uint8_t offset) {
    constexpr int32_t nbytes = 1024;
    for(int j=0; j<nbytes; j++) {
        out_buffer[j] = in_buffer[j] + offset;
    }
}