// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void softmax_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 64;
    float max_val = (float)input_vector[0];
    for (uint32_t i = 1; i < vector_size; i++) {
        float val = (float)input_vector[i];
        if (val > max_val) {
            max_val = val;
        }
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i] - max_val;
        int32_t ix = (int32_t)(x * 1.442695040888963f);
        float fx = x * 1.442695040888963f - ix;
        ix = (ix + 127) << 23;
        float pow2_ix;
        memcpy(&pow2_ix, &ix, sizeof(float));
        float pow2_fx = 1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;
        float result = pow2_ix * pow2_fx;
        output_vector[i] = (bfloat16)result;
        sum += result;
    }
    constexpr float eps = 1e-7f;
    sum = sum + eps;
    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vector_size; i++) {
        float val = (float)output_vector[i] * inv_sum;
        output_vector[i] = (bfloat16)val;
    }
}