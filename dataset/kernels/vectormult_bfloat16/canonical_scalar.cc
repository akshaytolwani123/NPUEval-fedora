// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectormult_bfloat16(bfloat16 *in_buffer1, bfloat16 *in_buffer2, bfloat16 *out_buffer, int32_t vector_size) {
    for (int32_t i = 0; i < vector_size; ++i) {
        out_buffer[i] = in_buffer1[i] * in_buffer2[i];
    }
}