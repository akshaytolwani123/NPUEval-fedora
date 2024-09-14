// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void vectormatrix_mult_int32(int32_t *vector, int32_t *matrix, int32_t *result) {
    constexpr uint32_t vector_size = 16;
    constexpr uint32_t matrix_cols = 16;
    for (uint32_t i = 0; i < matrix_cols; i++) {
        result[i] = 0;
    }
    for (uint32_t i = 0; i < matrix_cols; i++) {
        for (uint32_t j = 0; j < vector_size; j++) {
            result[i] += vector[j] * matrix[j * matrix_cols + i];
        }
    }
}