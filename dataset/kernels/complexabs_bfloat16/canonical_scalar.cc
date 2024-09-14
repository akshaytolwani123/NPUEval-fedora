// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void complexabs_bfloat16(bfloat16 *real_vector, bfloat16 *imag_vector, 
                        bfloat16 *output_vector) {
    constexpr int32_t vector_size = 64;
    for (uint32_t i = 0; i < vector_size; i++) {
        float real = (float)real_vector[i];
        float imag = (float)imag_vector[i];
        float x = real * real + imag * imag;
        if (x == 0.0f) {
            output_vector[i] = (bfloat16)0.0f;
            continue;
        }
        float xhalf = 0.5f * x;
        int32_t j = *(int32_t*)&x;
        j = 0x5f3759df - (j >> 1); // fast inverse square root
        float y = *(float*)&j;
        y = y * (1.5f - xhalf * y * y);
        float result = x * y;
        output_vector[i] = (bfloat16)result;
    }
}