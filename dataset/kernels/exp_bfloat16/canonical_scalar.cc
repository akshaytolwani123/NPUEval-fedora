// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void exp_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr float log2e = 1.442695040888963f;
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float c1 = ln2;
    constexpr float c2 = 0.2401598148889220f;
    constexpr float c3 = 0.0558801115849865f;
    constexpr int32_t vector_size = 64;
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];
        int32_t ix = (int32_t)(x * log2e);
        float fx = x * log2e - ix;
        ix = (ix + 127) << 23;
        float pow2_ix;
        memcpy(&pow2_ix, &ix, sizeof(float));
        float fx2 = fx * fx;
        float fx3 = fx2 * fx;
        float pow2_fx = 1.0f +
                        c1 * fx +
                        c2 * fx2 +
                        c3 * fx3;
        float result = pow2_ix * pow2_fx;
        output_vector[i] = (bfloat16)result;
    }
}