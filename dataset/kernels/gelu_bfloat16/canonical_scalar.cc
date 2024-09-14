// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <math.h>

void gelu_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);

        float inner_squared = inner * inner;
        float tanh_num = inner * (135135.0f + inner_squared * (17325.0f + inner_squared * (378.0f + inner_squared)));
        float tanh_den = 135135.0f + inner_squared * (62370.0f + inner_squared * (3150.0f + inner_squared * 28.0f));
        float tanh_approx = tanh_num / tanh_den;

        float cdf = 0.5f * (1.0f + tanh_approx);
        output_vector[i] = (bfloat16)(x * cdf);
    }
}