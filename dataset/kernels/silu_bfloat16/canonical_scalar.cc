// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void silu_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    // constants for exp2-approximation
    const float LOG2E = 1.4426950408889634f;  // log2(e)
    const float C1    = 0.6931471805599453f;  // ln2
    const float C2    = 0.2402265069591010f;  // (ln2)^2/2
    const float C3    = 0.05550410866482158f; // (ln2)^3/6
    
    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        float sigmoid;
        
        // Calculate sigmoid(x) = 1/(1+e^(-x))
        if (x > 16.0f) {
            // For very large x, sigmoid(x) ≈ 1
            sigmoid = 1.0f;
        }
        else if (x < -16.0f) {
            // For very negative x, sigmoid(x) ≈ 0
            sigmoid = 0.0f;
        }
        else {
            // Calculate e^(-x) using the same exp approximation as in softplus
            float t = -x * LOG2E;
            int   ip = int(t);
            float fp = t - float(ip);
            unsigned bits = unsigned(ip + 127) << 23;
            union { unsigned u; float f; } pun; pun.u = bits;
            float p2i = pun.f;
            float p2f = 1.0f
                      + C1 * fp
                      + C2 * fp * fp
                      + C3 * fp * fp * fp;
            float exp_neg_x = p2i * p2f;
            
            // sigmoid(x) = 1/(1+e^(-x))
            sigmoid = 1.0f / (1.0f + exp_neg_x);
        }
        
        // SiLU(x) = x * sigmoid(x)
        output[i] = bfloat16(x * sigmoid);
    }
}