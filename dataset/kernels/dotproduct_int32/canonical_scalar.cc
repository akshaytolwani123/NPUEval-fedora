// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void dotproduct_int32(int32_t *vector1, int32_t *vector2, int32_t *result) {
    constexpr int32_t vector_size = 256;
    *result = 0;
    for (uint32_t i = 0; i < vector_size; i++) {
        *result += vector1[i] * vector2[i];
    }
}