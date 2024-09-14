// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>
#include <cstdint>

// Computes variance of input of length 256 (bfloat16), writes scalar output (bfloat16).
void variance_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < N; ++i) {
        float val = float(input[i]);
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / float(N);
    float var = (sum_sq / float(N)) - (mean * mean);

    // Clamp to zero if round-off yields small negatives:
    if (var < 0.0f) var = 0.0f;

    *output = bfloat16(var);
}