// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void inverse_uint8(uint8_t *in_buffer, uint8_t* out_buffer) {
    constexpr uint32_t nbytes = 1024;
    constexpr uint32_t num_elements = nbytes / sizeof(uint8_t);

    for (uint32_t i = 0; i < num_elements; i++) {
        out_buffer[i] = 255 - in_buffer[i];
    }
}