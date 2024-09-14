// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void gather_bfloat16_int32idx(bfloat16* input_data, int32_t* indices, bfloat16* output_data) {
    constexpr int32_t GATHER_N = 256;
    for (int i = 0; i < GATHER_N; i++) {
        int32_t idx = indices[i];
        output_data[i] = input_data[idx];
    }
}