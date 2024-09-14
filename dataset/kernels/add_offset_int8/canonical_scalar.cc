// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void add_offset_int8(int8_t* in_buffer, int8_t* out_buffer, int8_t offset) {
    constexpr int32_t NUM_ELEMENTS = 256;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        int16_t tmp = static_cast<int16_t>(in_buffer[i]) + static_cast<int16_t>(offset);
        out_buffer[i] = static_cast<int8_t>(tmp);
    }
}