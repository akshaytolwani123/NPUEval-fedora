// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void maxpool2d_relu_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int ROWS = 32;
    constexpr int COLS = 32;
    constexpr int WINDOW_SIZE = 2;
    constexpr int STRIDE = 2;

    for (int i = 0; i < ROWS; i += STRIDE) {
        for (int j = 0; j < COLS; j += STRIDE) {
            bfloat16 max_val = input[i * COLS + j];
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
                    bfloat16 current_val = input[(i + wi) * COLS + (j + wj)];
                    if (current_val > max_val) {
                        max_val = current_val;
                    }
                }
            }
            float out_val = float(max_val) > 0.0f ? float(max_val) : 0.0f;
            output[(i/STRIDE) * (COLS/STRIDE) + (j/STRIDE)] = (bfloat16)out_val;
        }
    }
}