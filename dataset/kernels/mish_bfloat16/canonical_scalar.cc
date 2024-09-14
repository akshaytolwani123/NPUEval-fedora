// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void mish_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    const float LOG2E = 1.44269504f;   // log2(e)
    const float LN2   = 0.69314718f;   // ln(2)
    const float C1    = LN2;           // exp approx 1st-order
    const float C2    = 0.24022651f;   // exp approx 2nd-order

    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);

        // --- softplus: ln(1 + exp(x)) ---
        float sp;
        if (x > 16.0f) {
            sp = x;
        } else if (x < -16.0f) {
            // exp_approx(x):
            {
                float xl = x * LOG2E;
                int   ip = int(xl);
                float fp = xl - float(ip);
                unsigned bits = unsigned(ip + 127) << 23;
                union { unsigned u; float f; } pun; pun.u = bits;
                float p2i = pun.f;
                float p2f = 1.0f + C1 * fp + C2 * fp * fp;
                sp = p2i * p2f;
            }
        } else if (x > 0.0f) {
            // ln(1+e^x) = x + ln1p(e^{-x})
            float z;
            // exp_approx(-x):
            {
                float xl = -x * LOG2E;
                int   ip = int(xl);
                float fp = xl - float(ip);
                unsigned bits = unsigned(ip + 127) << 23;
                union { unsigned u; float f; } pun; pun.u = bits;
                float p2i = pun.f;
                float p2f = 1.0f + C1 * fp + C2 * fp * fp;
                z = p2i * p2f;
            }
            // ln1p_poly(z): ≈ z*(1 – z*(½ – z/3))
            sp = x + z * (1.0f - z * (0.5f - z * (1.0f/3.0f)));
        } else {
            // ln(1+e^x) for x<=0: ln1p(e^x)
            float z;
            // exp_approx(x):
            {
                float xl = x * LOG2E;
                int   ip = int(xl);
                float fp = xl - float(ip);
                unsigned bits = unsigned(ip + 127) << 23;
                union { unsigned u; float f; } pun; pun.u = bits;
                float p2i = pun.f;
                float p2f = 1.0f + C1 * fp + C2 * fp * fp;
                z = p2i * p2f;
            }
            sp = z * (1.0f - z * (0.5f - z * (1.0f/3.0f)));
        }

        // --- tanh(sp) via rational approx ---
        float tanh_sp;
        if (sp > 4.97f) {
            tanh_sp = 1.0f;
        } else {
            float y2  = sp * sp;
            float num = sp * (135135.0f
                           + y2 * (17325.0f
                           + y2 * (378.0f
                           + y2)));
            float den = 135135.0f
                      + y2 * (62370.0f
                      + y2 * (3150.0f
                      + y2 * 28.0f));
            tanh_sp = num / den;
        }

        // --- output Mish ---
        output[i] = bfloat16(x * tanh_sp);
    }
}
