// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void relu_bfloat16(bfloat16 *in_buffer, bfloat16* out_buffer) {
    constexpr int32_t num_elements = 256;
    for (uint32_t i = 0; i < num_elements; ++i) {
        out_buffer[i] = in_buffer[i] < 0 ? 0 : in_buffer[i];
    }
}