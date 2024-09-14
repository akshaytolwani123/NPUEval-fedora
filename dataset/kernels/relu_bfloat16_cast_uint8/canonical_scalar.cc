//
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void relu_bfloat16_cast_uint8(bfloat16* input_vector, uint8_t* output_vector) {
    constexpr int32_t VECTOR_SIZE = 256;
    for (uint32_t i = 0; i < VECTOR_SIZE; i++) {
        float x        = (float)input_vector[i];
        float relu_val = x > 0.0f ? x : 0.0f;

        int   ip   = (int)relu_val;
        float frac = relu_val - (float)ip;
        if (frac > 0.5f) {
            ip += 1;
        }
        else if (frac == 0.5f && (ip & 1)) {
            ip += 1;
        }

        if      (ip <  0)   ip = 0;
        else if (ip > 255)  ip = 255;

        output_vector[i] = (uint8_t)ip;
    }
}