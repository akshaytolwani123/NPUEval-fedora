// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void maxpool2d_relu_int8(int8_t* input, int8_t* output) {
    constexpr int32_t ROWS = 16;
    constexpr int32_t COLS = 16;
    constexpr int32_t WINDOW_SIZE = 2;
    constexpr int32_t STRIDE = 2;

    for (int i = 0; i < ROWS; i += STRIDE) {
        for (int j = 0; j < COLS; j += STRIDE) {
            int8_t max_val = input[i * COLS + j];
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
                    int row_idx = i + wi;
                    int col_idx = j + wj;
                    if (row_idx < ROWS && col_idx < COLS) {
                        int8_t val = input[row_idx * COLS + col_idx];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
            if (max_val < 0) max_val = 0;
            output[(i / STRIDE) * (COLS / STRIDE) + (j / STRIDE)] = max_val;
        }
    }
}