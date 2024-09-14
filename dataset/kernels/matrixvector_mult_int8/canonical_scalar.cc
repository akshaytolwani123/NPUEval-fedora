// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void matrixvector_mult_int8(int8_t* matrix, int8_t* vector, int32_t* result) {
    constexpr int ROWS = 16;
    constexpr int COLS = 16;
    for (int i = 0; i < ROWS; ++i) {
        int32_t acc = 0;
        for (int j = 0; j < COLS; ++j) {
            acc += static_cast<int32_t>(matrix[i * COLS + j]) * static_cast<int32_t>(vector[j]);
        }
        result[i] = acc;
    }
}