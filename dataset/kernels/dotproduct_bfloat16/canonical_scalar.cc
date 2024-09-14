// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT


void dotproduct_bfloat16(bfloat16 *vector1, bfloat16 *vector2, bfloat16 *result, uint32_t vector_size) {
    *result = 0;
    for (uint32_t i = 0; i < vector_size; i++) {
        *result += vector1[i] * vector2[i];
    }
}