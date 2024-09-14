// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void pad2d_int32(int32_t *input_matrix, int32_t *padded_matrix, uint32_t pad_size, int32_t pad_value) {
    constexpr uint32_t input_rows = 16;
    constexpr uint32_t input_cols = 16;
    uint32_t padded_rows = input_rows + 2 * pad_size;
    uint32_t padded_cols = input_cols + 2 * pad_size;

    for (uint32_t i = 0; i < padded_rows; i++) {
        for (uint32_t j = 0; j < padded_cols; j++) {
            padded_matrix[i * padded_cols + j] = pad_value;
        }
    }

    for (uint32_t i = 0; i < input_rows; i++) {
        for (uint32_t j = 0; j < input_cols; j++) {
            padded_matrix[(i + pad_size) * padded_cols + (j + pad_size)] = input_matrix[i * input_cols + j];
        }
    }
}