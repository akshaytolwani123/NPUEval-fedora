// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void hardsigmoid_int8(const int8_t *input_vector, int8_t *output_vector) {
    constexpr int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float x = (float)input_vector[i];
        float y = (x / 6.0f) + 0.5f;
        if (y < 0.0f) y = 0.0f;
        if (y > 1.0f) y = 1.0f;
        int32_t out = (int32_t)(y * 127.0f + 0.5f);
        if (out > 127) out = 127;
        if (out < 0) out = 0;
        output_vector[i] = (int8_t)out;
    }
}