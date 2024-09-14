// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void avgpool2d_relu_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int ROWS = 32;
    constexpr int COLS = 32;
    constexpr int WINDOW_SIZE = 2;
    constexpr int STRIDE = 2;

    for (int i = 0; i < ROWS; i += STRIDE) {
        for (int j = 0; j < COLS; j += STRIDE) {
            float sum = 0.0f;
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
                    sum += (float)input[(i + wi) * COLS + (j + wj)];
                }
            }
            float avg = sum / (WINDOW_SIZE * WINDOW_SIZE);
            float relu = avg > 0.0f ? avg : 0.0f;
            output[(i/STRIDE) * (COLS/STRIDE) + (j/STRIDE)] = (bfloat16)relu;
        }
    }
}